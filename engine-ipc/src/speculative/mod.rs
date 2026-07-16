//! Speculative Decoding Module
//!
//! Implements draftless N-gram speculative decoding with:
//! - N-gram sliding window hash index
//! - Draft generation with confidence gating
//! - Batched target verification with KV-cache mask adjustment
//! - Real-time metrics collection

pub mod ngram_index;
pub mod speculator;
pub mod verifier;
pub mod config;

// Re-export public types
pub use config::{SpeculativeConfig, NgramIndexConfig, VerifierConfig, SpeculativeMetrics};
pub use ngram_index::{NgramIndex, ThreadLocalNgramIndex};
pub use speculator::{Speculator, DraftToken, SpeculativeResult};
pub use verifier::{Verifier, VerificationResult, TokenVerification, KvCacheMaskAdjuster, SpeculativeVerifier};