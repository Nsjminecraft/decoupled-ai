//! Target Verification & KV-Cache Mask Adjuster for Speculative Decoding
//!
//! Implements batched forward pass verification with parallel token acceptance
//! and KV-cache mask adjustment for correct attention masking.

use anyhow::Result;
use crate::{ComputeBackend, GenerateRequest, InferenceEngine, ModelInstance};
use half::f16;
use std::sync::Arc;
use super::{SpeculativeConfig, SpeculativeResult, DraftToken};
use tracing::{debug, trace};

/// Result of token verification
#[derive(Debug, Clone)]
pub struct TokenVerification {
    pub position: usize,
    pub draft_token: u32,
    pub target_token: u32,
    pub target_logprob: f32,
    pub accepted: bool,
    pub finish_reason: Option<String>,
}

/// Result of batched verification
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Tokens verified (prefix of draft that was accepted)
    pub accepted_tokens: Vec<u32>,
    /// All verification results (for metrics)
    pub verifications: Vec<TokenVerification>,
    /// Number of tokens accepted
    pub acceptance_count: usize,
    /// Whether generation should continue (all draft tokens accepted)
    pub continue_generation: bool,
    /// Finish reason if stopped
    pub finish_reason: Option<String>,
}

/// Configuration for verifier
#[derive(Debug, Clone)]
pub struct VerifierConfig {
    pub acceptance_threshold: f32,
    pub max_batch_size: usize,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            acceptance_threshold: 0.5,
            max_batch_size: 16,
        }
    }
}

/// Metrics tracked by the verifier
#[derive(Debug, Default, Clone)]
pub struct VerifierMetrics {
    pub total_verifications: u64,
    pub total_accepted: u64,
    pub total_rejected: u64,
    pub verification_batches: u64,
}

impl VerifierMetrics {
    pub fn acceptance_rate(&self) -> Option<f32> {
        if self.total_verifications == 0 {
            None
        } else {
            Some(self.total_accepted as f32 / self.total_verifications as f32)
        }
    }

    pub fn avg_tokens_per_step(&self) -> Option<f32> {
        if self.verification_batches == 0 {
            None
        } else {
            Some(self.total_accepted as f32 / self.verification_batches as f32)
        }
    }
}

/// Batched verifier for speculative decoding
pub struct Verifier {
    backend: Arc<dyn ComputeBackend>,
    config: VerifierConfig,
    metrics: std::sync::Mutex<VerifierMetrics>,
}

impl Verifier {
    pub fn new(backend: Arc<dyn ComputeBackend>, config: VerifierConfig) -> Self {
        Self { backend, config, metrics: std::sync::Mutex::new(VerifierMetrics::default()) }
    }

    pub fn metrics(&self) -> VerifierMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Verify draft tokens using a single batched forward pass
    ///
    /// Algorithm:
    /// 1. Concatenate prompt + all draft tokens into single sequence
    /// 2. Run ONE forward pass to get logits for all positions
    /// 3. For each draft position, compare target logits with draft token
    /// 4. Accept tokens while they match; stop at first mismatch
    /// 5. If mismatch, target model's token replaces draft token
    pub async fn verify_draft(
        &self,
        model: &ModelInstance,
        prompt_tokens: &[u32],
        draft_tokens: &[DraftToken],
        temperature: f32,
        top_p: f32,
        stop_tokens: &[u32],
    ) -> Result<VerificationResult> {
        if draft_tokens.is_empty() {
            return Ok(VerificationResult {
                accepted_tokens: Vec::new(),
                verifications: Vec::new(),
                acceptance_count: 0,
                continue_generation: true,
                finish_reason: None,
            });
        }

        let draft_token_ids: Vec<u32> = draft_tokens.iter().map(|d| d.token).collect();
        let num_draft = draft_token_ids.len();

        // Build combined sequence: prompt + draft tokens
        let mut combined = prompt_tokens.to_vec();
        combined.extend_from_slice(&draft_token_ids);

        // Run batched forward pass to get logits for draft positions
        // In a real implementation, we'd compute logits for ALL positions
        // but we only need the last `num_draft` positions
        let logits = self.compute_logits_for_sequence(model, &combined)?;

        // Extract logits for draft positions
        let prompt_len = prompt_tokens.len();
        let mut verifications = Vec::with_capacity(num_draft);
        let mut accepted_tokens = Vec::with_capacity(num_draft);
        let mut acceptance_count = 0;

        for (i, &draft_token) in draft_token_ids.iter().enumerate() {
            let pos = prompt_len + i;

            // Get target model's logits at this position
            let position_logits = &logits[pos];

            // Sample from target distribution (with temperature/top-p)
            let target_token = self.sample_token(position_logits, temperature, top_p)?;

            // Get target token's probability
            let target_logprob = self.logprob_of_token(position_logits, draft_token, temperature);

            let accepted = target_token == draft_token;

            // Check stop tokens
            let finish_reason = if stop_tokens.contains(&target_token) {
                Some("stop".to_string())
            } else {
                None
            };

            verifications.push(TokenVerification {
                position: pos,
                draft_token,
                target_token,
                target_logprob,
                accepted,
                finish_reason: finish_reason.clone(),
            });

            if accepted {
                accepted_tokens.push(draft_token);
                acceptance_count += 1;

                if finish_reason.is_some() {
                    return Ok(VerificationResult {
                        accepted_tokens,
                        verifications,
                        acceptance_count,
                        continue_generation: false,
                        finish_reason,
                    });
                }
            } else {
                // Mismatch: use target model's token, stop accepting further
                accepted_tokens.push(target_token);
                trace!("Mismatch at pos {}: draft={}, target={}, logprob={:.4}",
                       pos, draft_token, target_token, target_logprob);
                break;
            }
        }

        let continue_generation = acceptance_count == num_draft && verifications.last().map(|v| v.finish_reason.is_none()).unwrap_or(true);

        // Update metrics
        {
            let mut m = self.metrics.lock().unwrap();
            m.total_verifications += verifications.len() as u64;
            m.total_accepted += acceptance_count as u64;
            m.total_rejected += (verifications.len() - acceptance_count) as u64;
            m.verification_batches += 1;
        }

        Ok(VerificationResult {
            accepted_tokens,
            verifications,
            acceptance_count,
            continue_generation,
            finish_reason: None,
        })
    }

    /// Compute logits for entire sequence using batched forward pass
    /// This is a simplified version - real implementation uses the model's forward pass
    fn compute_logits_for_sequence(
        &self,
        model: &ModelInstance,
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        // In real implementation, this would call the model's forward pass
        // For now, return dummy logits
        let vocab_size = 32000;
        let seq_len = tokens.len();
        let mut logits = Vec::with_capacity(seq_len);

        for _ in 0..seq_len {
            // Uniform logits for testing
            logits.push(vec![0.0f32; vocab_size]);
        }

        Ok(logits)
    }

    /// Sample token from logits with temperature and top-p
    fn sample_token(&self, logits: &[f32], temperature: f32, top_p: f32) -> Result<u32> {
        // Simplified: argmax
        let max_idx = logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(max_idx as u32)
    }

    /// Get log-probability of a specific token
    fn logprob_of_token(&self, logits: &[f32], token: u32, temperature: f32) -> f32 {
        // Simplified: return 0 for testing
        0.0
    }
}

/// KV-Cache Mask Adjuster for Speculative Decoding
///
/// When speculative tokens are rejected, the KV-cache mask must be adjusted
/// so that attention only attends to accepted tokens.
pub struct KvCacheMaskAdjuster {
    max_seq_len: usize,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
}

impl KvCacheMaskAdjuster {
    pub fn new(max_seq_len: usize, num_layers: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            max_seq_len,
            num_layers,
            num_heads,
            head_dim,
        }
    }

    /// Compute adjusted causal mask for speculative decoding
    ///
    /// Args:
    /// - prompt_len: Length of original prompt
    /// - accepted_len: Number of accepted draft tokens (including any corrected token)
    ///
    /// Returns: Flat causal mask [seq_len, seq_len] where seq_len = prompt_len + accepted_len
    pub fn adjusted_mask(&self, prompt_len: usize, accepted_len: usize) -> Vec<f32> {
        let total_len = prompt_len + accepted_len;
        let mut mask = vec![0.0f32; total_len * total_len];

        for i in 0..total_len {
            for j in 0..=i {
                mask[i * total_len + j] = 1.0;
            }
            // Positions > i remain 0 (causal masking)
        }

        mask
    }

    /// Advance write position in KV-cache after accepting tokens
    pub fn advance_write_pos(&self, current_pos: usize, accepted_len: usize) -> usize {
        current_pos + accepted_len
    }

    /// Handle rejection: adjust KV-cache for corrected token
    ///
    /// When a draft token is rejected at position `rejection_pos`,
    /// we need to:
    /// 1. Invalidate KV entries for rejected token and all subsequent draft tokens
    /// 2. Write the corrected (target) token at rejection_pos
    /// 3. Continue from rejection_pos + 1
    pub fn handle_rejection(
        &self,
        draft_tokens: &[u32],
        verifications: &[TokenVerification],
        current_kv_len: usize,
    ) -> (Vec<u32>, usize) {
        // Find first rejection
        let rejection_idx = verifications.iter()
            .position(|v| !v.accepted)
            .unwrap_or(verifications.len());

        // Build corrected sequence: accepted draft tokens + target token at rejection
        let mut corrected = Vec::with_capacity(rejection_idx + 1);
        for v in &verifications[..rejection_idx] {
            corrected.push(v.draft_token);
        }
        if rejection_idx < verifications.len() {
            corrected.push(verifications[rejection_idx].target_token);
        }

        // New KV length = prompt_len + rejected_position + 1 (for corrected token)
        let new_kv_len = current_kv_len + rejection_idx + 1;

        (corrected, new_kv_len)
    }
}

/// Full speculative verifier combining batched verification and mask adjustment
pub struct SpeculativeVerifier {
    pub verifier: Verifier,
    pub mask_adjuster: KvCacheMaskAdjuster,
}

impl SpeculativeVerifier {
    pub fn new(
        backend: Arc<dyn ComputeBackend>,
        verifier_config: VerifierConfig,
        max_seq_len: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            verifier: Verifier::new(backend, verifier_config),
            mask_adjuster: KvCacheMaskAdjuster::new(max_seq_len, num_layers, num_heads, head_dim),
        }
    }

    pub fn verifier_metrics(&self) -> VerifierMetrics {
        self.verifier.metrics()
    }

    pub async fn verify_and_adjust(
        &self,
        model: &ModelInstance,
        prompt_tokens: &[u32],
        draft_tokens: &[DraftToken],
        temperature: f32,
        top_p: f32,
        stop_tokens: &[u32],
    ) -> Result<(VerificationResult, Vec<f32>)> {
        let verification = self.verifier.verify_draft(
            model,
            prompt_tokens,
            draft_tokens,
            temperature,
            top_p,
            stop_tokens,
        ).await?;

        // Compute adjusted mask
        let accepted_len = verification.accepted_tokens.len();
        let mask = self.mask_adjuster.adjusted_mask(prompt_tokens.len(), accepted_len);

        Ok((verification, mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjusted_mask() {
        let adjuster = KvCacheMaskAdjuster::new(10, 1, 1, 1);
        let mask = adjuster.adjusted_mask(5, 3); // 5 prompt + 3 accepted = 8 total

        assert_eq!(mask.len(), 64); // 8x8

        // Check causal structure
        for i in 0..8 {
            for j in 0..8 {
                let expected = if j <= i { 1.0 } else { 0.0 };
                assert_eq!(mask[i * 8 + j], expected, "Position ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_advance_write_pos() {
        let adjuster = KvCacheMaskAdjuster::new(10, 1, 1, 1);
        assert_eq!(adjuster.advance_write_pos(5, 3), 8);
        assert_eq!(adjuster.advance_write_pos(0, 0), 0);
    }

    #[test]
    fn test_handle_rejection() {
        let adjuster = KvCacheMaskAdjuster::new(10, 1, 1, 1);
        let draft = vec![10, 20, 30, 40];

        let verifications = vec![
            TokenVerification { position: 0, draft_token: 10, target_token: 10, target_logprob: 1.0, accepted: true, finish_reason: None },
            TokenVerification { position: 1, draft_token: 20, target_token: 25, target_logprob: 0.5, accepted: false, finish_reason: None },
            TokenVerification { position: 2, draft_token: 30, target_token: 30, target_logprob: 1.0, accepted: true, finish_reason: None },
            TokenVerification { position: 3, draft_token: 40, target_token: 40, target_logprob: 1.0, accepted: true, finish_reason: None },
        ];

        let (corrected, new_len) = adjuster.handle_rejection(&draft, &verifications, 5);
        // Should correct position 1 to 25, stop after that
        assert_eq!(corrected[0], 10);
        assert_eq!(corrected[1], 25); // Corrected
        assert_eq!(new_len, 7); // 5 prompt + 2 valid (positions 0,1)
    }
}