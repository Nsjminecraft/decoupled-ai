//! Configuration for Speculative Decoding

use serde::{Deserialize, Serialize};

/// Configuration for N-gram index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramIndexConfig {
    pub max_order: usize,           // 4 (4-gram base)
    pub max_entries: usize,         // Cap on total entries (~4M for 16MB)
    pub vocab_size: usize,          // Vocabulary size for hash packing
}

impl Default for NgramIndexConfig {
    fn default() -> Self {
        Self {
            max_order: 4,
            max_entries: 4_000_000,
            vocab_size: 65536,      // u16 vocab
        }
    }
}

/// Configuration for speculative decoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    /// Enable speculative decoding
    pub enabled: bool,
    /// Maximum number of draft tokens per round
    pub max_draft_tokens: usize,
    /// Minimum confidence threshold for accepting draft tokens (0.0-1.0)
    pub confidence_threshold: f32,
    /// Maximum N-gram order (default 4)
    pub ngram_order: usize,
    /// Maximum entries in N-gram index
    pub max_ngram_entries: usize,
    /// Temperature for draft token sampling
    pub draft_temperature: f32,
    /// Top-k for draft sampling
    pub draft_top_k: usize,
    /// Top-p for draft sampling
    pub draft_top_p: f32,
    /// Maximum context length for N-gram index
    pub max_ngram_context: usize,
    /// Verification acceptance threshold
    pub verification_threshold: f32,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_draft_tokens: 8,
            confidence_threshold: 0.5,
            ngram_order: 4,
            max_ngram_entries: 4_000_000,
            draft_temperature: 0.1,
            draft_top_k: 50,
            draft_top_p: 0.9,
            max_ngram_context: 256,
            verification_threshold: 0.5,
        }
    }
}

/// Configuration for verifier
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Metrics for speculative decoding
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpeculativeMetrics {
    pub tokens_indexed: usize,
    pub unique_ngrams: usize,
    pub config: SpeculativeConfig,
    // Verifier metrics
    pub acceptance_rate: Option<f32>,
    pub avg_tokens_per_step: Option<f32>,
    pub verification_batches: u64,
    pub rejections: u64,
    pub index_evictions: u64,
    pub speedup_ratio: Option<f32>,
}