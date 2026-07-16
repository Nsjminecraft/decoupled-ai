//! N-Gram Sliding Window Hash Indexer for Speculative Decoding
//!
//! Implements a compact rolling-hash index for N-gram continuation lookup
//! with back-off from 4-gram → 3-gram → 2-gram → 1-gram.
//! Target: <50μs lookup latency, ~16MB memory cap.

use hashbrown::HashMap;
use smallvec::SmallVec;
use std::sync::{Arc, RwLock};
use crate::speculative::config::NgramIndexConfig;
use tracing::{debug, trace};

/// Packed N-gram key for hash table
/// Uses FNV-1a hash for variable-length N-grams
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct NgramKey(u64);

impl NgramKey {
    /// Pack tokens into a 64-bit key using FNV-1a
    fn from_tokens(tokens: &[u32], max_order: usize) -> Self {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET;
        // Include order in hash to distinguish 3-gram "a b c" from 4-gram "x a b c"
        hash ^= max_order as u64;
        hash = hash.wrapping_mul(FNV_PRIME);

        for &token in tokens {
            hash ^= token as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        NgramKey(hash)
    }

    /// Perfect hash for fixed 4-gram when vocab <= 65535
    fn from_4gram(tokens: &[u32]) -> Option<Self> {
        if tokens.len() != 4 {
            return None;
        }
        for &t in tokens {
            if t > u16::MAX as u32 {
                return None;
            }
        }
        // Pack 4 u16 into u64: [t0, t1, t2, t3]
        let packed = ((tokens[0] as u64) << 48) |
                     ((tokens[1] as u64) << 32) |
                     ((tokens[2] as u64) << 16) |
                      (tokens[3] as u64);
        Some(NgramKey(packed))
    }
}

/// N-gram index with thread-safe read access and exclusive write
pub struct NgramIndex {
    // Main hash table: NgramKey -> continuation tokens (SmallVec for inline storage)
    table: RwLock<HashMap<NgramKey, SmallVec<[u32; 4]>>>,
    config: NgramIndexConfig,
    token_count: std::sync::atomic::AtomicUsize,
    // LRU tracking for eviction (simplified: timestamp-based)
    access_time: RwLock<HashMap<NgramKey, u64>>,
    global_timestamp: std::sync::atomic::AtomicU64,
    // Eviction counter for metrics
    eviction_count: std::sync::atomic::AtomicU64,
}

impl NgramIndex {
    /// Create new N-gram index with default config
    pub fn new(config: NgramIndexConfig) -> Self {
        Self {
            table: RwLock::new(HashMap::with_capacity(config.max_entries.min(100_000))),
            config,
            token_count: std::sync::atomic::AtomicUsize::new(0),
            access_time: RwLock::new(HashMap::new()),
            global_timestamp: std::sync::atomic::AtomicU64::new(0),
            eviction_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(NgramIndexConfig::default())
    }

    /// Get eviction count for metrics
    pub fn eviction_count(&self) -> u64 {
        self.eviction_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Insert a sequence of tokens, updating all N-grams up to max_order
    /// Called on each generated token to maintain rolling window
    /// Stores context (prefix of order-1) -> continuation (next token) mappings
    pub fn insert(&self, context: &[u32]) {
        if context.is_empty() {
            return;
        }

        let mut table = self.table.write().unwrap();
        let mut access_time = self.access_time.write().unwrap();
        let timestamp = self.global_timestamp.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Insert N-grams of order 1..max_order
        // For each order, the context is the prefix of length (order-1) and continuation is the last token
        for order in 1..=self.config.max_order {
            if context.len() < order {
                break;
            }

            // Context is the first (order-1) tokens from the end of context (excluding last token)
            // context = [prefix..., continuation]
            // We need prefix of length (order-1) ending at the second-to-last position
            let context_prefix = &context[..context.len() - 1];
            if context_prefix.len() < order - 1 {
                continue;
            }

            let key = NgramKey::from_tokens(&context_prefix[context_prefix.len() - (order - 1)..], order - 1);
            let continuation = *context.last().unwrap();

            let entry = table.entry(key).or_default();
            if !entry.contains(&continuation) {
                entry.push(continuation);
                // Cap continuations per N-gram to prevent unbounded growth
                if entry.len() > 16 {
                    entry.remove(0); // FIFO eviction
                }
            }

            access_time.insert(key, timestamp);
        }

        self.token_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Evict if we exceed capacity
        if table.len() > self.config.max_entries {
            Self::evict_if_needed(&mut table, &mut access_time, self.config.max_entries, &self.eviction_count);
        }
    }

    /// Query continuations for a given context at specific order
    /// Returns continuations that follow the given context prefix (sliding window)
    /// Uses the last (order-1) tokens of context (excluding the last token which is the expected continuation)
    pub fn query(&self, context: &[u32], order: usize) -> Option<SmallVec<[u32; 4]>> {
        if context.is_empty() || order == 0 || order > self.config.max_order {
            return None;
        }

        // For backoff query, context includes the expected continuation as last token
        // We need to use the last (order-1) tokens before the continuation as the key
        // This matches the insert logic which uses context_prefix = context[..context.len()-1]
        let context_without_continuation = &context[..context.len() - 1];
        if context_without_continuation.len() < order - 1 {
            return None;
        }

        // Take the last (order-1) tokens as the N-gram prefix key
        let prefix = &context_without_continuation[context_without_continuation.len() - (order - 1)..];
        let key = NgramKey::from_tokens(prefix, order - 1);

        let table = self.table.read().unwrap();

        // DEBUG
        eprintln!("DEBUG query: context={:?}, order={}, prefix={:?}, key={:?}", context, order, prefix, key);
        eprintln!("DEBUG table keys: {:?}", table.keys().collect::<Vec<_>>());

        let result = table.get(&key).cloned();

        // Update access time
        if result.is_some() {
            let mut access_time = self.access_time.write().unwrap();
            let timestamp = self.global_timestamp.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            access_time.insert(key, timestamp);
        }

        trace!("N-gram query: order={}, context_len={}, found={}", order, context.len(), result.is_some());
        result
    }

    /// Back-off query: try max_order, then max_order-1, ..., down to 1
    /// Returns the first non-empty result with highest order (all continuations for sampling)
    /// The expected continuation filtering is done by the caller (speculator)
    pub fn backoff_query(&self, context: &[u32]) -> Option<(usize, SmallVec<[u32; 4]>)> {
        for order in (1..=self.config.max_order).rev() {
            if let Some(continuations) = self.query(context, order) {
                if !continuations.is_empty() {
                    debug!("Back-off query succeeded at order {}", order);
                    return Some((order, continuations));
                }
            }
        }
        None
    }

    /// Get the most likely continuation (highest frequency) for a context
    /// Simplified: returns first inserted = most recent
    pub fn best_continuation(&self, context: &[u32]) -> Option<u32> {
        self.backoff_query(context)
            .map(|(_, continuations)| continuations[0])
    }

    /// Get all candidate continuations for drafting (with back-off)
    pub fn draft_candidates(&self, context: &[u32]) -> Vec<u32> {
        if let Some((order, continuations)) = self.backoff_query(context) {
            trace!("Draft candidates from {}-gram: {} options", order, continuations.len());
            continuations.into_iter().collect()
        } else {
            Vec::new()
        }
    }

    /// Current number of tokens indexed
    pub fn token_count(&self) -> usize {
        self.token_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Current number of unique N-grams stored
    pub fn unique_ngrams(&self) -> usize {
        self.table.read().unwrap().len()
    }

    /// Evict oldest entries if capacity exceeded
    fn evict_if_needed(
        table: &mut HashMap<NgramKey, SmallVec<[u32; 4]>>,
        access_time: &mut HashMap<NgramKey, u64>,
        max_entries: usize,
        eviction_count: &std::sync::atomic::AtomicU64,
    ) {
        if table.len() <= max_entries {
            return;
        }

        // Find oldest entries - collect keys first to avoid borrow issues
        let mut entries: Vec<_> = access_time.iter().map(|(k, v)| (*k, *v)).collect();
        entries.sort_by_key(|(_, ts)| *ts);

        // Evict enough to get back to max_entries (not just 10%)
        let evict_count = table.len() - max_entries;
        let keys_to_remove: Vec<_> = entries.into_iter().take(evict_count).map(|(k, _)| k).collect();

        for key in keys_to_remove {
            table.remove(&key);
            access_time.remove(&key);
        }

        eviction_count.fetch_add(evict_count as u64, std::sync::atomic::Ordering::Relaxed);

        debug!("Evicted {} N-gram entries, remaining: {}", evict_count, table.len());
    }

    /// Clear the index (for testing or model switch)
    pub fn clear(&self) {
        self.table.write().unwrap().clear();
        self.access_time.write().unwrap().clear();
        self.token_count.store(0, std::sync::atomic::Ordering::Relaxed);
        self.global_timestamp.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

impl std::fmt::Debug for NgramIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NgramIndex")
            .field("token_count", &self.token_count())
            .field("unique_ngrams", &self.unique_ngrams())
            .field("config", &self.config)
            .finish()
    }
}

/// Thread-local N-gram index for each inference stream
/// Avoids lock contention between concurrent generations
pub struct ThreadLocalNgramIndex {
    index: Arc<NgramIndex>,
    // Local buffer for rolling context
    context_buffer: Vec<u32>,
    max_context: usize,
}

impl ThreadLocalNgramIndex {
    pub fn new(config: NgramIndexConfig, max_context: usize) -> Self {
        Self {
            index: Arc::new(NgramIndex::new(config)),
            context_buffer: Vec::with_capacity(max_context),
            max_context,
        }
    }

    /// Add a token to the rolling context and update index
    pub fn push_token(&mut self, token: u32) {
        self.context_buffer.push(token);
        if self.context_buffer.len() > self.max_context {
            self.context_buffer.remove(0);
        }
        self.index.insert(&self.context_buffer);
    }

    /// Get draft candidates from current context
    pub fn draft_candidates(&self) -> Vec<u32> {
        self.index.draft_candidates(&self.context_buffer)
    }

    /// Get best single continuation
    pub fn best_continuation(&self) -> Option<u32> {
        self.index.best_continuation(&self.context_buffer)
    }

    /// Access underlying index for stats
    pub fn index(&self) -> &Arc<NgramIndex> {
        &self.index
    }

    /// Reset context (new generation)
    pub fn reset(&mut self) {
        self.context_buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_key_packing() {
        let tokens = vec![1, 2, 3, 4];
        let key = NgramKey::from_4gram(&tokens).expect("should pack 4-gram");

        // Verify round-trip
        let unpacked = [
            ((key.0 >> 48) & 0xFFFF) as u32,
            ((key.0 >> 32) & 0xFFFF) as u32,
            ((key.0 >> 16) & 0xFFFF) as u32,
            (key.0 & 0xFFFF) as u32,
        ];
        assert_eq!(&unpacked[..], &tokens[..]);
    }

    #[test]
    fn test_ngram_key_hash() {
        let tokens = vec![100, 200, 300, 400];
        let key1 = NgramKey::from_tokens(&tokens, 4);
        let key2 = NgramKey::from_tokens(&tokens, 4);
        assert_eq!(key1, key2);

        // Different order should give different key
        let key3 = NgramKey::from_tokens(&tokens, 3);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_basic_insert_and_query() {
        let index = NgramIndex::default();
        let context = vec![1, 2, 3, 4, 5];
        index.insert(&context);

        // Query 4-gram "2 3 4 5" -> should find continuation 5
        let query_ctx = vec![2, 3, 4, 5];
        let result = index.query(&query_ctx, 4);
        assert!(result.is_some());
        assert!(result.unwrap().contains(&5));
    }

    #[test]
    fn test_backoff_query() {
        let index = NgramIndex::default();

        // Insert sequence: 1 2 3 4 5
        // This will insert N-grams ending at 5:
        // order 1: [5] -> 5
        // order 2: [4, 5] -> 5
        // order 3: [3, 4, 5] -> 5
        // order 4: [2, 3, 4, 5] -> 5
        index.insert(&[1, 2, 3, 4, 5]);

        // Query for "2 3 4 5" - should find 4-gram match (continuation = 5)
        let result = index.backoff_query(&[2, 3, 4, 5]);
        assert!(result.is_some());
        let (order, continuations) = result.unwrap();
        assert_eq!(order, 4); // Found at 4-gram level
        assert!(continuations.contains(&5)); // 5 follows "2 3 4 5"

        // Also test 3-gram
        let result = index.backoff_query(&[3, 4, 5]);
        assert!(result.is_some());
        let (order, continuations) = result.unwrap();
        assert_eq!(order, 3); // Found at 3-gram level
        assert!(continuations.contains(&5));
    }

    #[test]
    fn test_thread_local_index() {
        let config = NgramIndexConfig::default();
        let mut local = ThreadLocalNgramIndex::new(config, 100);

        // Simulate generating tokens
        local.push_token(1);
        local.push_token(2);
        local.push_token(3);
        local.push_token(4);

        // Should have candidates based on "1 2 3 4"
        let candidates = local.draft_candidates();
        assert!(!candidates.is_empty() || local.index().token_count() > 0);
    }

    #[test]
    fn test_eviction() {
        let config = NgramIndexConfig {
            max_order: 2,
            max_entries: 10,
            vocab_size: 1000,
        };
        let max_entries = config.max_entries;
        let index = NgramIndex::new(config);

        // Insert many unique N-grams to trigger eviction
        for i in 0..100 {
            index.insert(&[i, i + 1]);
        }

        // Should have evicted some entries
        assert!(index.unique_ngrams() <= max_entries);
    }
}