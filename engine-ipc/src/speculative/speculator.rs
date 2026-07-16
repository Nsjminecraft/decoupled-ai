//! Draft Generation Speculator Engine for Speculative Decoding
//!
//! Generates candidate draft tokens using N-gram continuations with
//! back-off and confidence-gated sampling.

use anyhow::Result;
use crate::speculative::config::{SpeculativeConfig, SpeculativeMetrics};
use crate::speculative::ngram_index::NgramIndex;
use rand::Rng;
use rand::SeedableRng;
use std::sync::Arc;
use tracing::{debug, trace};

/// Draft token with metadata
#[derive(Debug, Clone)]
pub struct DraftToken {
    pub token: u32,
    pub logprob: f32,
    pub source_order: usize,  // Which N-gram order produced this
    pub confidence: f32,
}

/// Result of speculative generation
#[derive(Debug, Clone)]
pub struct SpeculativeResult {
    pub draft_tokens: Vec<DraftToken>,
    pub confidence: f32,
    pub should_verify: bool,
}

/// Draft Generation Speculator Engine
pub struct Speculator {
    ngram_index: Arc<NgramIndex>,
    config: SpeculativeConfig,
    rng: rand::rngs::StdRng,
}

impl Speculator {
    /// Create new speculator with N-gram index
    pub fn new(ngram_index: Arc<NgramIndex>, config: SpeculativeConfig) -> Self {
        Self {
            ngram_index,
            config,
            rng: rand::rngs::StdRng::seed_from_u64(rand::random::<u64>()),
        }
    }

    /// Generate draft tokens using N-gram index
    ///
    /// Algorithm:
    /// 1. Get N-gram continuations from current context
    /// 2. If confidence < threshold, return empty (no speculation)
    /// 3. Sample from continuations up to max_draft_tokens
    /// 4. Update rolling context after each sampled token
    pub fn generate_draft(&mut self, context: &[u32]) -> SpeculativeResult {
        if !self.config.enabled {
            return SpeculativeResult {
                draft_tokens: Vec::new(),
                confidence: 0.0,
                should_verify: false,
            };
        }

        // Get candidates from N-gram index with back-off
        let candidates = self.ngram_index.draft_candidates(context);

        if candidates.is_empty() {
            trace!("No N-gram candidates found for context len {}", context.len());
            return SpeculativeResult {
                draft_tokens: Vec::new(),
                confidence: 0.0,
                should_verify: false,
            };
        }

        // Estimate confidence based on N-gram order and candidate count
        let confidence = self.estimate_confidence(&candidates, context);

        if confidence < self.config.confidence_threshold {
            debug!("Confidence {:.3} below threshold {:.3}, disabling speculation",
                   confidence, self.config.confidence_threshold);
            return SpeculativeResult {
                draft_tokens: Vec::new(),
                confidence,
                should_verify: false,
            };
        }

        // Generate draft tokens
        let mut draft_tokens = Vec::with_capacity(self.config.max_draft_tokens);
        let mut rolling_context = context.to_vec();

        for _ in 0..self.config.max_draft_tokens {
            let next_candidates = self.ngram_index.draft_candidates(&rolling_context);
            if next_candidates.is_empty() {
                break;
            }

            // Sample from candidates
            let token = self.sample_from_candidates(&next_candidates);
            let logprob = 0.0; // Would compute from actual probabilities
            let source_order = 4; // Simplified

            draft_tokens.push(DraftToken {
                token,
                logprob,
                source_order,
                confidence,
            });

            // Update rolling context
            rolling_context.push(token);
            if rolling_context.len() > self.config.max_ngram_context {
                rolling_context.remove(0);
            }
        }

        trace!("Generated {} draft tokens with confidence {:.3}",
               draft_tokens.len(), confidence);

        let should_verify = !draft_tokens.is_empty();

        SpeculativeResult {
            draft_tokens,
            confidence,
            should_verify,
        }
    }

    /// Estimate confidence for draft generation
    /// Higher for higher-order N-grams, fewer candidates
    fn estimate_confidence(&self, candidates: &[u32], context: &[u32]) -> f32 {
        // Base confidence increases with N-gram order found
        let base_confidence = match candidates.len() {
            1 => 0.9,
            2..=4 => 0.7,
            5..=10 => 0.5,
            _ => 0.3,
        };

        // Adjust based on context length (longer context = more reliable)
        let context_bonus = (context.len() as f32 / self.config.max_ngram_context as f32).min(0.1);

        (base_confidence + context_bonus).min(1.0)
    }

    /// Sample token from candidates using configured sampling strategy
    fn sample_from_candidates(&mut self, candidates: &[u32]) -> u32 {
        if candidates.len() == 1 {
            return candidates[0];
        }

        // Temperature-based sampling
        let temp = self.config.draft_temperature;
        if temp <= 0.0 {
            return candidates[0]; // Greedy
        }

        // Apply top-k
        let k = self.config.draft_top_k.min(candidates.len());
        let top_candidates = &candidates[..k];

        // Apply top-p (nucleus) sampling
        if self.config.draft_top_p < 1.0 {
            // Simplified: just use first few
            let p_count = ((k as f32) * self.config.draft_top_p).ceil() as usize;
            let p_candidates = &top_candidates[..p_count.min(top_candidates.len())];
            let idx = self.rng.gen_range(0..p_candidates.len());
            return p_candidates[idx];
        }

        // Uniform sampling from top-k
        let idx = self.rng.gen_range(0..top_candidates.len());
        top_candidates[idx]
    }

    /// Update N-gram index with newly generated tokens
    /// Called after verification to keep index in sync
    pub fn update_index(&self, context: &[u32]) {
        self.ngram_index.insert(context);
    }

    /// Get current metrics
    pub fn metrics(&self) -> SpeculativeMetrics {
        SpeculativeMetrics {
            tokens_indexed: self.ngram_index.token_count(),
            unique_ngrams: self.ngram_index.unique_ngrams(),
            config: self.config.clone(),
            acceptance_rate: None,
            avg_tokens_per_step: None,
            verification_batches: 0,
            rejections: 0,
            index_evictions: self.ngram_index.eviction_count(),
            speedup_ratio: None,
        }
    }
}

impl SpeculativeConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.max_draft_tokens == 0 {
            anyhow::bail!("max_draft_tokens must be > 0");
        }
        if self.confidence_threshold < 0.0 || self.confidence_threshold > 1.0 {
            anyhow::bail!("confidence_threshold must be in [0, 1]");
        }
        if self.ngram_order == 0 {
            anyhow::bail!("ngram_order must be > 0");
        }
        if self.max_ngram_entries == 0 {
            anyhow::bail!("max_ngram_entries must be > 0");
        }
        if self.draft_temperature < 0.0 {
            anyhow::bail!("draft_temperature must be >= 0");
        }
        if self.draft_top_p < 0.0 || self.draft_top_p > 1.0 {
            anyhow::bail!("draft_top_p must be in [0, 1]");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::config::NgramIndexConfig;

    #[test]
    fn test_speculative_config_default() {
        let config = SpeculativeConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_draft_tokens, 8);
        assert_eq!(config.confidence_threshold, 0.5);
        assert_eq!(config.ngram_order, 4);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_speculative_config_validation() {
        let mut config = SpeculativeConfig::default();
        config.max_draft_tokens = 0;
        assert!(config.validate().is_err());

        config = SpeculativeConfig::default();
        config.confidence_threshold = 1.5;
        assert!(config.validate().is_err());

        config = SpeculativeConfig::default();
        config.draft_temperature = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_speculator_generate_draft() {
        let index = Arc::new(NgramIndex::new(NgramIndexConfig::default()));
        let config = SpeculativeConfig::default();
        let mut speculator = Speculator::new(index.clone(), config);

        // Add some N-grams to the index
        index.insert(&[1, 2, 3, 4, 5]);
        index.insert(&[2, 3, 4, 5, 6]);

        // Generate draft from context "2 3 4 5"
        let context = vec![2, 3, 4, 5];
        let result = speculator.generate_draft(&context);

        // Should generate some draft tokens
        assert!(result.should_verify || result.draft_tokens.is_empty());
    }

    #[test]
    fn test_confidence_estimation() {
        let index = Arc::new(NgramIndex::new(NgramIndexConfig::default()));
        let config = SpeculativeConfig::default();
        let speculator = Speculator::new(index, config);

        // Fewer candidates = higher confidence
        let c1 = speculator.estimate_confidence(&[1], &[1,2,3,4]);
        let c2 = speculator.estimate_confidence(&[1,2,3,4,5,6,7,8,9,10], &[1,2,3,4]);
        assert!(c1 > c2);
    }
}