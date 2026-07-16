use anyhow::{anyhow, Context, Result};
use bincode::{Decode, Encode};
use clap::{Parser, Subcommand};
use half::f16;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use tracing::{info, warn};
use uuid::Uuid;

// New modules for HF downloader and safetensors parser
pub mod downloader;
pub mod safetensors;

// ============================================================================
// Constants & Magic
// ============================================================================

pub const BRAIN_MAGIC: [u8; 8] = *b"BRAIN.1\0";
pub const BRAIN_VERSION: u32 = 1;
pub const WEIGHTS_ALIGNMENT: u64 = 64;
pub const INDEX_ENTRY_SIZE: usize = 64;

// ============================================================================
// Sharded Disk-Streaming Spec
// ============================================================================
//
// A sharded model is laid out as a single static-resident base fragment
// (`00.brain`) plus exactly `NUM_SHARDS` sequential streaming fragments
// (`01.brain` .. `15.brain`). The maximum streaming footprint of any one
// shard is `SHARD_MAX_BYTES` (2 GiB by directive), capping the active
// resident tensor memory during evaluation.
//
// Layout on disk (one directory per model):
//   <model>/00.brain                 -- always RAM-resident (embeddings, attn)
//   <model>/01.brain .. <model>/15.brain   -- 15 streaming fragments
//   <model>/<model>.shard.idx        -- tiny index: shard table + tensor routes
//
// `ShardIndex` is the only piece that stays permanently resident; it is
// the contract the memory overlays (mem-posix / mem-windows) and the LRU
// cache driver in `stream-cache` read to decide which shard to map.

/// 2 GiB hard cap per streaming shard, enforced by the packer.
pub const SHARD_MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// Number of streaming fragments mandated by the sharding directive.
pub const NUM_SHARDS: u16 = 15;
/// Magic for the shard index file.
pub const SHARD_INDEX_MAGIC: [u8; 8] = *b"SHARD.1\0";
/// Index version.
pub const SHARD_INDEX_VERSION: u16 = 1;

/// One entry per shard (00=base, 01..15=streaming). Fixed 64 bytes.
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct ShardDesc {
    /// 0 for the static base shard, 1..=15 for streaming fragments.
    pub shard_id: u16,
    pub version: u16,
    pub byte_length: u64,
    pub tensor_count: u64,
    /// First tensor index (into the manifest's tensor list) owned by this shard.
    pub first_tensor: u64,
    pub reserved: [u64; 4],
}

impl ShardDesc {
    pub const SIZE: usize = 64;

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        let mut o = 0;
        macro_rules! w_u16 {
            ($v:expr) => {{
                buf[o..o + 2].copy_from_slice(&($v).to_le_bytes());
                o += 2;
            }};
        }
        macro_rules! w_u64 {
            ($v:expr) => {{
                buf[o..o + 8].copy_from_slice(&($v).to_le_bytes());
                o += 8;
            }};
        }
        w_u16!(self.shard_id);
        w_u16!(self.version);
        // 4 bytes of padding to 8-align u64s
        o += 4;
        w_u64!(self.byte_length);
        w_u64!(self.tensor_count);
        w_u64!(self.first_tensor);
        let reserved = self.reserved;
        for d in reserved.iter() {
            w_u64!(*d);
        }
        debug_assert_eq!(o, Self::SIZE);
        buf
    }

    pub fn from_bytes(buf: &[u8; Self::SIZE]) -> Self {
        let mut o = 0;
        let shard_id = u16::from_le_bytes(buf[o..o + 2].try_into().unwrap());
        o += 2;
        let version = u16::from_le_bytes(buf[o..o + 2].try_into().unwrap());
        o += 2;
        o += 4; // padding
        let byte_length = u64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
        o += 8;
        let tensor_count = u64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
        o += 8;
        let first_tensor = u64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
        o += 8;
        let mut reserved = [0u64; 4];
        for d in reserved.iter_mut() {
            *d = u64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
            o += 8;
        }
        debug_assert_eq!(o, Self::SIZE);
        Self { shard_id, version, byte_length, tensor_count, first_tensor, reserved }
    }
}

/// Per-shard index file. Tiny; stays permanently in RAM.
#[derive(Debug, Clone)]
pub struct ShardIndex {
    pub version: u16,
    pub num_shards: u16,
    /// Size of the always-resident base fragment (00.brain).
    pub base_header_size: u64,
    /// Max bytes per streaming shard (enforced).
    pub per_shard_max: u64,
    /// One entry per shard (length == num_shards + 1, index 0 == base).
    pub shards: Vec<ShardDesc>,
    /// Routing table: tensor name -> shard id. Lets the LRU driver and the
    /// memory overlays know which shard owns a tensor without re-reading the
    /// manifest on every lookup.
    pub routes: HashMap<String, u16>,
}

impl ShardIndex {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + 2 + 2 + 8 + 8 + self.shards.len() * ShardDesc::SIZE);
        buf.extend_from_slice(&SHARD_INDEX_MAGIC);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&self.num_shards.to_le_bytes());
        buf.extend_from_slice(&self.base_header_size.to_le_bytes());
        buf.extend_from_slice(&self.per_shard_max.to_le_bytes());
        for shard in &self.shards {
            buf.extend_from_slice(&shard.to_bytes());
        }
        // Routes: u64 count, then each route as: u16 shard_id, u64 name_len, name bytes.
        buf.extend_from_slice(&(self.routes.len() as u64).to_le_bytes());
        for (name, &shard_id) in &self.routes {
            buf.extend_from_slice(&shard_id.to_le_bytes());
            let name_bytes = name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(name_bytes);
        }
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cur = std::io::Cursor::new(bytes);
        let mut magic = [0u8; 8];
        cur.read_exact(&mut magic)?;
        if magic != SHARD_INDEX_MAGIC {
            return Err(anyhow!("Invalid shard index magic: {:?}", magic));
        }
        let mut b2 = [0u8; 2];
        cur.read_exact(&mut b2)?;
        let version = u16::from_le_bytes(b2);
        cur.read_exact(&mut b2)?;
        let num_shards = u16::from_le_bytes(b2);
        let mut b8 = [0u8; 8];
        cur.read_exact(&mut b8)?;
        let base_header_size = u64::from_le_bytes(b8);
        cur.read_exact(&mut b8)?;
        let per_shard_max = u64::from_le_bytes(b8);

        let total = num_shards as usize + 1; // include base
        let mut shards = Vec::with_capacity(total);
        for _ in 0..total {
            let mut entry = [0u8; ShardDesc::SIZE];
            cur.read_exact(&mut entry)?;
            shards.push(ShardDesc::from_bytes(&entry));
        }

        let mut route_count_b = [0u8; 8];
        cur.read_exact(&mut route_count_b)?;
        let route_count = u64::from_le_bytes(route_count_b) as usize;
        let mut routes = HashMap::with_capacity(route_count);
        for _ in 0..route_count {
            cur.read_exact(&mut b2)?;
            let shard_id = u16::from_le_bytes(b2);
            cur.read_exact(&mut b8)?;
            let name_len = u64::from_le_bytes(b8) as usize;
            let mut name = vec![0u8; name_len];
            cur.read_exact(&mut name)?;
            routes.insert(String::from_utf8(name)?, shard_id);
        }

        Ok(Self { version, num_shards, base_header_size, per_shard_max, shards, routes })
    }

    /// Write the index file to `<dir>/<name>.shard.idx`.
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let bytes = self.to_bytes();
        std::fs::write(&path, &bytes)?;
        info!("Wrote shard index ({} routes) to {}", self.routes.len(), path.as_ref().display());
        Ok(())
    }

    pub fn read<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(&path)?;
        Self::from_bytes(&bytes)
    }
}

/// Build a sharded layout from an in-memory `BrainPack`, writing:
///   - `<out_dir>/<base_name>.00.brain` (static base)
///   - `<out_dir>/<base_name>.01.brain` .. `15.brain`
///   - `<out_dir>/<base_name>.shard.idx`
///
/// Routing policy (matches the directive):
///   * Tensors whose names start with `embed` or `token_emb` or contain
///     `attention` / `attn` / `norm` / `rope` go to shard 0 (resident).
///   * Everything else falls into streaming shards, evenly, with each shard
///     capped at [`SHARD_MAX_BYTES`]. When a tensor would overflow the active
///     shard, the next shard is opened.
pub fn write_sharded(pack: &BrainPack, out_dir: &Path, base_name: &str) -> Result<ShardIndex> {
    std::fs::create_dir_all(out_dir)?;

    // Classify tensors.
    let is_resident = |name: &str| {
        let n = name.to_ascii_lowercase();
        n.contains("embed") || n.contains("attn") || n.contains("attention")
            || n.contains("norm") || n.contains("rope") || n.contains("positional")
    };

    // Pass 1: assign resident tensors to shard 0 and the rest to shard >=1.
    let mut routes: HashMap<String, u16> = HashMap::new();
    let mut shard_tensors: Vec<Vec<usize>> = vec![Vec::new(); (NUM_SHARDS as usize) + 1];

    for (i, t) in pack.manifest.tensors.iter().enumerate() {
        if is_resident(&t.name) {
            routes.insert(t.name.clone(), 0);
            shard_tensors[0].push(i);
        }
    }

    // Pass 2: distribute the remaining tensors across NUM_SHARDS streaming
    // shards, filling each up to SHARD_MAX_BYTES before moving on.
    let mut current_shard: u16 = 1;
    let mut current_len: u64 = 0;
    for (i, t) in pack.manifest.tensors.iter().enumerate() {
        if routes.contains_key(&t.name) {
            continue;
        }
        if current_len + t.size_bytes > SHARD_MAX_BYTES && current_shard < NUM_SHARDS {
            current_shard += 1;
            current_len = 0;
        }
        if current_shard > NUM_SHARDS {
            return Err(anyhow!(
                "Model exceeds {} streaming shards (cap {}B each); refusing to emit invalid layout",
                NUM_SHARDS, SHARD_MAX_BYTES
            ));
        }
        routes.insert(t.name.clone(), current_shard);
        shard_tensors[current_shard as usize].push(i);
        current_len += t.size_bytes;
    }

    // Write the base shard (00.brain): full BrainPack container, but only
    // embedding/attn/norm tensors. Reuse the standard writer by building a
    // trimmed BrainPack so downstream reader paths stay identical.
    let mut base_builder = BrainPackBuilder::new()
        .model(pack.manifest.model.clone())
        .metadata(pack.manifest.metadata.clone());
    for &i in &shard_tensors[0] {
        let t = &pack.manifest.tensors[i];
        let view = pack.get_tensor(&t.name)?;
        if t.quantization.is_some() {
            let qp = t.quantization.as_ref().unwrap();
            let qdata = if qp.scales_size > 0 {
                pack.weights[qp.scales_offset as usize..][..qp.scales_size as usize].to_vec()
            } else {
                Vec::new()
            };
            base_builder = base_builder.add_tensor_with_quant(t.clone(), view.data, &qdata)?;
        } else {
            base_builder = base_builder.add_tensor(t.clone(), view.data)?;
        }
    }
    let base_pack = base_builder.build()?;
    let base_path = out_dir.join(format!("{}.00.brain", base_name));
    base_pack.write(&base_path)?;
    let base_header_size = std::fs::metadata(&base_path)?.len();

    // Write streaming shards: each is a flat append of tensor payloads,
    // aligned to WEIGHTS_ALIGNMENT, prefixed by a per-shard TensorIndexEntry
    // block describing the tensors it owns. The reader reconstructs the
    // global manifest from the shard index routes.
    let mut shards: Vec<ShardDesc> = Vec::with_capacity((NUM_SHARDS as usize) + 1);
    shards.push(ShardDesc {
        shard_id: 0,
        version: SHARD_INDEX_VERSION,
        byte_length: base_header_size,
        tensor_count: shard_tensors[0].len() as u64,
        first_tensor: 0,
        reserved: [0; 4],
    });

    for s in 1..=NUM_SHARDS as u16 {
        let path = out_dir.join(format!("{}.{:02}.brain", base_name, s));
        let owned = &shard_tensors[s as usize];
        let mut writer = File::create(&path)?;
        // Index header: u64 tensor_count, then tensor_count × TensorIndexEntry.
        writer.write_all(&(owned.len() as u64).to_le_bytes())?;
        let _ = writer.stream_position()?; // align later
        // Write entries first (we'll fix offsets after writing payloads).
        // For simplicity, write a placeholder for each entry then rewrite.
        let entry_pos = writer.stream_position()?;
        for _ in owned {
            writer.write_all(&[0u8; TensorIndexEntry::SIZE])?;
        }
        // Align payloads to 64 bytes.
        let pos = writer.stream_position()?;
        let pad = (WEIGHTS_ALIGNMENT - (pos % WEIGHTS_ALIGNMENT)) % WEIGHTS_ALIGNMENT;
        writer.write_all(&vec![0u8; pad as usize])?;
        let payload_start = writer.stream_position()?;

        let mut offsets: Vec<u64> = Vec::with_capacity(owned.len());
        for &i in owned {
            let t = &pack.manifest.tensors[i];
            let view = pack.get_tensor(&t.name)?;
            // Align within the shard.
            let here = writer.stream_position()?;
            let p = (WEIGHTS_ALIGNMENT - (here % WEIGHTS_ALIGNMENT)) % WEIGHTS_ALIGNMENT;
            writer.write_all(&vec![0u8; p as usize])?;
            // Absolute file offset of this tensor's payload (from file start).
            let abs_off = writer.stream_position()?;
            writer.write_all(view.data)?;
            offsets.push(abs_off);
        }
        let byte_length = writer.stream_position()?;

        // Rewrite the index entries with correct file offsets. Stash the
        // global manifest tensor index in `reserved` (lower 32 bits of the
        // first u64) so `ShardedModel::locate` can match by global index
        // without trusting contiguous ordering.
        writer.seek(SeekFrom::Start(entry_pos))?;
        for (j, &i) in owned.iter().enumerate() {
            let t = &pack.manifest.tensors[i];
            let abs_off = offsets[j];
            let mut entry = TensorIndexEntry::from_tensor(t, 0);
            entry.offset = abs_off;
            entry.size_bytes = t.size_bytes;
            entry.reserved = i as u32; // global tensor index (low 32 bits)
            writer.write_all(&entry.to_bytes())?;
        }
        writer.flush()?;

        shards.push(ShardDesc {
            shard_id: s,
            version: SHARD_INDEX_VERSION,
            byte_length,
            tensor_count: owned.len() as u64,
            first_tensor: shard_tensors[s as usize].first().map(|x| *x as u64).unwrap_or(0),
            reserved: [0; 4],
        });
    }

    let index = ShardIndex {
        version: SHARD_INDEX_VERSION,
        num_shards: NUM_SHARDS,
        base_header_size,
        per_shard_max: SHARD_MAX_BYTES,
        shards,
        routes,
    };
    index.write(out_dir.join(format!("{}.shard.idx", base_name)))?;
    Ok(index)
}

// ============================================================================
// Data Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
#[repr(u32)]
pub enum DataType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    Q4_0 = 3,
    Q4KM = 4,
    Q4KS = 5,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8K = 9,
    I8 = 10,
    I16 = 11,
    I32 = 12,
    I64 = 13,
    U8 = 14,
    Bool = 15,
    F8E4M3 = 16,
    F8E5M2 = 17,
}

impl DataType {
    pub fn size(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 | DataType::BF16 | DataType::I16 => 2,
            DataType::I8 | DataType::U8 | DataType::Bool => 1,
            DataType::I32 | DataType::I64 => 4,
            DataType::F8E4M3 | DataType::F8E5M2 => 1,
            DataType::Q4_0 | DataType::Q4KM | DataType::Q4KS => 1, // packed
            DataType::Q5_0 | DataType::Q5_1 => 1,
            DataType::Q8_0 | DataType::Q8K => 1,
        }
    }

    pub fn alignment(&self) -> u64 {
        match self {
            DataType::F32 => 4,
            DataType::F16 | DataType::BF16 | DataType::I16 => 2,
            DataType::I32 | DataType::I64 => 4,
            DataType::I8 | DataType::U8 | DataType::Bool => 1,
            DataType::F8E4M3 | DataType::F8E5M2 => 1,
            _ => 64, // quantized types need 64-byte alignment
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
#[repr(u32)]
pub enum QuantizationScheme {
    None = 0,
    Q4_0 = 1,
    Q4KM = 2,
    Q4KS = 3,
    Q5_0 = 4,
    Q5_1 = 5,
    Q8_0 = 6,
    Q8K = 7,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub architecture: String,
    pub parameter_count: u64,
    pub quantization: String,
    pub context_length: u32,
    pub vocab_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    pub scales_offset: u64,
    pub scales_size: u64,
    pub zero_points_offset: u64,
    pub zero_points_size: u64,
    pub block_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: DataType,
    pub offset: u64,
    pub size_bytes: u64,
    pub quantization: Option<QuantizationParams>,
    pub quantization_type: QuantizationScheme,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub created_epoch: u64,
    pub created_by: String,
    pub checksum: String,
    pub license: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub format: String,
    pub model: ModelInfo,
    pub tensors: Vec<TensorInfo>,
    pub metadata: Metadata,
}

// ============================================================================
// Tensor Index Entry (64 bytes exactly)
// ============================================================================

#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct TensorIndexEntry {
    pub name_offset: u64,
    pub name_length: u64,
    pub offset: u64,
    pub size_bytes: u64,
    pub ndim: u32,
    pub dtype: u32,
    pub quantization_type: u32,
    pub reserved: u32,
    pub shape: [u64; 8],
    pub quant_params_offset: u64,
    pub quant_params_size: u64,
    pub reserved2: [u64; 2],
}

impl TensorIndexEntry {
    pub const SIZE: usize = 144;

    /// Serialize entry to a fixed 64-byte little-endian buffer.
    /// Use this instead of bincode because the struct is `#[repr(C, packed)]`,
    /// for which bincode's derive macro cannot generate sound code.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        let mut o = 0;
        macro_rules! w_u32 {
            ($v:expr) => {{
                buf[o..o + 4].copy_from_slice(&($v).to_le_bytes());
                o += 4;
            }};
        }
        macro_rules! w_u64 {
            ($v:expr) => {{
                buf[o..o + 8].copy_from_slice(&($v).to_le_bytes());
                o += 8;
            }};
        }
        w_u64!(self.name_offset);
        w_u64!(self.name_length);
        w_u64!(self.offset);
        w_u64!(self.size_bytes);
        w_u32!(self.ndim);
        w_u32!(self.dtype);
        w_u32!(self.quantization_type);
        w_u32!(self.reserved);
        // Copy `shape` out of the packed struct before iterating, because
        // borrowing a field of a `#[repr(C, packed)]` struct is UB (E0793).
        let shape = self.shape;
        for d in shape.iter() {
            w_u64!(*d);
        }
        w_u64!(self.quant_params_offset);
        w_u64!(self.quant_params_size);
        w_u64!(self.reserved2[0]);
        w_u64!(self.reserved2[1]);
        debug_assert_eq!(o, Self::SIZE);
        buf
    }

    /// Deserialize entry from a 64-byte little-endian buffer.
    pub fn from_bytes(buf: &[u8; Self::SIZE]) -> Self {
        let mut o = 0;
        macro_rules! r_u32 {
            () => {{
                let v = u32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
                o += 4;
                v
            }};
        }
        macro_rules! r_u64 {
            () => {{
                let v = u64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
                o += 8;
                v
            }};
        }
        let name_offset = r_u64!();
        let name_length = r_u64!();
        let offset = r_u64!();
        let size_bytes = r_u64!();
        let ndim = r_u32!();
        let dtype = r_u32!();
        let quantization_type = r_u32!();
        let reserved = r_u32!();
        let mut shape = [0u64; 8];
        for d in shape.iter_mut() {
            *d = r_u64!();
        }
        let quant_params_offset = r_u64!();
        let quant_params_size = r_u64!();
        let reserved2_0 = r_u64!();
        let reserved2_1 = r_u64!();
        debug_assert_eq!(o, Self::SIZE);
        Self {
            name_offset,
            name_length,
            offset,
            size_bytes,
            ndim,
            dtype,
            quantization_type,
            reserved,
            shape,
            quant_params_offset,
            quant_params_size,
            reserved2: [reserved2_0, reserved2_1],
        }
    }

    pub fn from_tensor(tensor: &TensorInfo, name_offset: u64) -> Self {
        let mut shape = [0u64; 8];
        for (i, &dim) in tensor.shape.iter().enumerate() {
            if i < 8 {
                shape[i] = dim;
            }
        }

        Self {
            name_offset,
            name_length: tensor.name.len() as u64,
            offset: tensor.offset,
            size_bytes: tensor.size_bytes,
            ndim: tensor.shape.len() as u32,
            dtype: tensor.dtype as u32,
            quantization_type: tensor.quantization_type as u32,
            reserved: 0,
            shape,
            quant_params_offset: tensor.quantization.as_ref().map(|q| q.scales_offset).unwrap_or(0),
            quant_params_size: tensor.quantization.as_ref().map(|q| q.scales_size).unwrap_or(0),
            reserved2: [0, 0],
        }
    }
}

// ============================================================================
// Quantization Parameter Structures
// ============================================================================

#[repr(C, packed)]
pub struct Q4KBlock {
    pub scales: [u8; 32],
    pub mins: [u8; 32],
    pub maxs: [u8; 32],
    pub weights: [u8; 128],
    pub high_bits: [u8; 32], // Only for Q4_K_M
}

impl Q4KBlock {
    pub const SIZE: usize = 256; // bytes per block
    pub const ELEMENTS: usize = 256;
}

#[repr(C, packed)]
pub struct Q8_0Block {
    pub scale: f32,
    pub weights: [i8; 32],
}

impl Q8_0Block {
    pub const SIZE: usize = 36; // 4 + 32 bytes
    pub const ELEMENTS: usize = 32;
}

// ============================================================================
// Brain Pack Builder
// ============================================================================

pub struct BrainPackBuilder {
    tensors: Vec<TensorInfo>,
    model: Option<ModelInfo>,
    metadata: Option<Metadata>,
    weights_data: Vec<u8>,
    name_strings: Vec<String>,
}

impl BrainPackBuilder {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            model: None,
            metadata: None,
            weights_data: Vec::new(),
            name_strings: Vec::new(),
        }
    }

    pub fn model(mut self, model: ModelInfo) -> Self {
        self.model = Some(model);
        self
    }

    pub fn metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn add_tensor(mut self, mut tensor: TensorInfo, data: &[u8]) -> Result<Self> {
        // Align weights data
        let align = tensor.dtype.alignment();
        let padding = (align - (self.weights_data.len() as u64 % align)) % align;
        self.weights_data.extend(vec![0u8; padding as usize]);

        tensor.offset = self.weights_data.len() as u64;
        tensor.size_bytes = data.len() as u64;

        // Add quantization params if present
        if let Some(ref mut qparams) = tensor.quantization {
            let q_align = 64;
            let q_padding = (q_align - (self.weights_data.len() as u64 % q_align)) % q_align;
            self.weights_data.extend(vec![0u8; q_padding as usize]);

            qparams.scales_offset = self.weights_data.len() as u64;
            // Scales will be written by quantization-specific logic
        }

        self.weights_data.extend_from_slice(data);
        self.tensors.push(tensor);
        Ok(self)
    }

    pub fn add_tensor_with_quant(mut self, mut tensor: TensorInfo, data: &[u8], qparams_data: &[u8]) -> Result<Self> {
        let align = tensor.dtype.alignment();
        let padding = (align - (self.weights_data.len() as u64 % align)) % align;
        self.weights_data.extend(vec![0u8; padding as usize]);

        tensor.offset = self.weights_data.len() as u64;
        tensor.size_bytes = data.len() as u64;

        if let Some(ref mut qparams) = tensor.quantization {
            let q_align = 64;
            let q_padding = (q_align - (self.weights_data.len() as u64 % q_align)) % q_align;
            self.weights_data.extend(vec![0u8; q_padding as usize]);

            qparams.scales_offset = self.weights_data.len() as u64;
            qparams.scales_size = qparams_data.len() as u64;
            self.weights_data.extend_from_slice(qparams_data);
        }

        self.weights_data.extend_from_slice(data);
        self.tensors.push(tensor);
        Ok(self)
    }

    pub fn build(self) -> Result<BrainPack> {
        let model = self.model.ok_or_else(|| anyhow!("Model info required"))?;
        let metadata = self.metadata.unwrap_or_else(|| Metadata {
            created_epoch: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            created_by: "brain-pack v0.1".to_string(),
            checksum: String::new(), // Will be computed
            license: "unknown".to_string(),
            description: String::new(),
        });

        // Compute checksum of weights
        let mut hasher = Sha256::new();
        hasher.update(&self.weights_data);
        let checksum = format!("sha256:{:x}", hasher.finalize());

        let mut metadata = metadata;
        metadata.checksum = checksum;

        Ok(BrainPack {
            manifest: Manifest {
                version: BRAIN_VERSION,
                format: "brain".to_string(),
                model,
                tensors: self.tensors,
                metadata,
            },
            weights: self.weights_data,
        })
    }
}

// ============================================================================
// Brain Pack Container
// ============================================================================

#[derive(Debug)]
pub struct BrainPack {
    pub manifest: Manifest,
    pub weights: Vec<u8>,
}

impl BrainPack {
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut file = File::create(&path).context("Failed to create output file")?;
        self.write_to(&mut file)?;
        info!("Written .brain file to {}", path.as_ref().display());
        Ok(())
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buffer);
        self.write_to(&mut cursor)?;
        Ok(buffer)
    }

    pub fn write_to<W: Write + Seek>(&self, writer: &mut W) -> Result<()> {
        // 1. Write magic bytes
        writer.write_all(&BRAIN_MAGIC)?;

        // 2. Serialize manifest to JSON
        let manifest_json = serde_json::to_vec(&self.manifest)?;
        let manifest_len = manifest_json.len() as u64;

        // 3. Write manifest length (u64 little-endian)
        writer.write_all(&manifest_len.to_le_bytes())?;

        // 4. Write manifest JSON + null terminator
        writer.write_all(&manifest_json)?;
        writer.write_all(&[0u8])?; // null terminator

        // 5. Align to 8-byte boundary for index map
        let current_pos = writer.stream_position()?;
        let index_align_padding = (8 - (current_pos % 8)) % 8;
        writer.write_all(&vec![0u8; index_align_padding as usize])?;

        // 6. Write tensor index map
        let name_table_start = writer.stream_position()?;
        let mut name_table = Vec::new();

        for tensor in &self.manifest.tensors {
            let name_offset = name_table.len() as u64;
            name_table.extend(tensor.name.as_bytes());
            name_table.push(0); // null terminator

            let entry = TensorIndexEntry::from_tensor(tensor, name_offset);
            let entry_bytes = entry.to_bytes();
            writer.write_all(&entry_bytes)?;
        }

        // 7. Align weights to 64-byte boundary
        let current_pos = writer.stream_position()?;
        let weights_align_padding = (WEIGHTS_ALIGNMENT - (current_pos % WEIGHTS_ALIGNMENT)) % WEIGHTS_ALIGNMENT;
        writer.write_all(&vec![0u8; weights_align_padding as usize])?;

        // 8. Write weights data
        writer.write_all(&self.weights)?;

        writer.flush()?;
        Ok(())
    }

    pub fn read<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(&path).context("Failed to open .brain file")?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::read_from_bytes(&mmap)
    }

    pub fn read_from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(bytes);

        // 1. Verify magic
        let mut magic = [0u8; 8];
        cursor.read_exact(&mut magic)?;
        if magic != BRAIN_MAGIC {
            return Err(anyhow!("Invalid magic bytes: expected BRAIN.1\\0, got {:?}", magic));
        }

        // 2. Read manifest length
        let mut len_bytes = [0u8; 8];
        cursor.read_exact(&mut len_bytes)?;
        let manifest_len = u64::from_le_bytes(len_bytes) as usize;

        // 3. Read manifest JSON
        let mut manifest_json = vec![0u8; manifest_len];
        cursor.read_exact(&mut manifest_json)?;

        // Read null terminator
        let mut null_byte = [0u8; 1];
        cursor.read_exact(&mut null_byte)?;
        if null_byte[0] != 0 {
            warn!("Manifest not null-terminated");
        }

        let manifest: Manifest = serde_json::from_slice(&manifest_json)?;

        // 4. Align to 8-byte boundary
        let current_pos = cursor.position();
        let index_align_padding = (8 - (current_pos % 8)) % 8;
        cursor.seek(SeekFrom::Current(index_align_padding as i64))?;

        // 5. Read tensor index entries
        let num_tensors = manifest.tensors.len();
        let mut tensors = Vec::with_capacity(num_tensors);

        // Read name table first (we need to parse entries to know where it ends)
        // Actually, names are stored inline in the JSON, so we just read entries
        for _ in 0..num_tensors {
            let mut entry_bytes = [0u8; TensorIndexEntry::SIZE];
            cursor.read_exact(&mut entry_bytes)?;
            let entry = TensorIndexEntry::from_bytes(&entry_bytes);

            // Verify entry matches manifest
            if entry.ndim as usize != manifest.tensors[tensors.len()].shape.len() {
                warn!("Tensor index entry shape dimension mismatch");
            }
            tensors.push(entry);
        }

        // 6. Align to weights
        let current_pos = cursor.position();
        let weights_align_padding = (WEIGHTS_ALIGNMENT - (current_pos % WEIGHTS_ALIGNMENT)) % WEIGHTS_ALIGNMENT;
        cursor.seek(SeekFrom::Current(weights_align_padding as i64))?;

        // 7. Read weights (rest of file)
        let weights_start = cursor.position() as usize;
        let weights = bytes[weights_start..].to_vec();

        // 8. Verify checksum
        let mut hasher = Sha256::new();
        hasher.update(&weights);
        let computed_checksum = format!("sha256:{:x}", hasher.finalize());
        if computed_checksum != manifest.metadata.checksum {
            return Err(anyhow!(
                "Checksum mismatch: expected {}, got {}",
                manifest.metadata.checksum,
                computed_checksum
            ));
        }

        Ok(Self { manifest, weights })
    }

    /// Borrow the in-memory weights buffer. Used by memory mappers that
    /// loaded the pack into RAM (e.g. the sharded base fragment).
    pub fn weights_ptr(&self, _mmap: &[u8]) -> *const u8 {
        self.weights.as_ptr()
    }

    /// Length of the in-memory weights buffer, in bytes.
    pub fn weights_len(&self) -> usize {
        self.weights.len()
    }

    /// Zero-copy tensor view naming the tensor by name with data borrowed
    /// from `weights`. The `_mmap` parameter is accepted for API symmetry
    /// with the memory crates but is unused; the view borrows `self.weights`.
    pub fn tensor_view<'a>(&'a self, name: &str, _mmap: &'a [u8]) -> Result<TensorView<'a>> {
        let tensor_info = self.manifest.tensors.iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found", name))?;
        let offset = tensor_info.offset as usize;
        let size = tensor_info.size_bytes as usize;
        if offset + size > self.weights.len() {
            return Err(anyhow!("Tensor data out of bounds"));
        }
        Ok(TensorView {
            data: &self.weights[offset..offset + size],
            shape: tensor_info.shape.clone(),
            dtype: tensor_info.dtype,
            quantization: tensor_info.quantization.clone(),
            quantization_type: tensor_info.quantization_type,
        })
    }

    /// Parse only the manifest (no weights copy). Used by memory overlays
    /// that want to know tensor offsets without materializing the payload.
    pub fn manifest_from_bytes(bytes: &[u8]) -> Result<Manifest> {
        let pack = Self::read_from_bytes(bytes)?;
        Ok(pack.manifest)
    }

    pub fn get_tensor(&self, name: &str) -> Result<TensorView> {
        let tensor_info = self.manifest.tensors.iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found", name))?;

        let entry = self.manifest.tensors.iter()
            .position(|t| t.name == name)
            .ok_or_else(|| anyhow!("Tensor index not found"))?;

        let offset = tensor_info.offset as usize;
        let size = tensor_info.size_bytes as usize;

        if offset + size > self.weights.len() {
            return Err(anyhow!("Tensor data out of bounds"));
        }

        Ok(TensorView {
            data: &self.weights[offset..offset + size],
            shape: tensor_info.shape.clone(),
            dtype: tensor_info.dtype,
            quantization: tensor_info.quantization.clone(),
            quantization_type: tensor_info.quantization_type,
        })
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.manifest.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    pub fn model_info(&self) -> &ModelInfo {
        &self.manifest.model
    }
}

// ============================================================================
// Sharded Model Reader (streaming fragment accessor)
// ============================================================================
//
// `ShardedModel` owns the always-resident `ShardIndex` plus the loaded base
// fragment (`00.brain`, a standard `BrainPack`). Streaming shards are NOT
// loaded here; the memory overlays / LRU driver in `stream-cache` map and
// evict them as needed, asking this struct for a tensor's shard + offset
// via [`ShardedModel::locate`].

pub struct ShardedModel {
    pub index: ShardIndex,
    pub base: BrainPack,
    /// Absolute file paths for `00.brain` .. `15.brain`, in shard-id order.
    pub shard_paths: Vec<std::path::PathBuf>,
}

/// Result of locating a tensor: which shard owns it and where inside.
#[derive(Debug, Clone, Copy)]
pub struct TensorLocation {
    pub shard_id: u16,
    /// Byte offset inside the shard file (absolute within the file).
    pub file_offset: u64,
    pub size_bytes: u64,
}

impl ShardedModel {
    /// Open a sharded model directory. Looks for `<dir>/<name>.shard.idx`,
    /// `<dir>/<name>.00.brain`, and `<dir>/<name>.NN.brain`.
    pub fn open(dir: &Path, name: &str) -> Result<Self> {
        let idx_path = dir.join(format!("{}.shard.idx", name));
        let index = ShardIndex::read(&idx_path)?;

        let mut shard_paths = Vec::with_capacity(index.shards.len());
        for shard in &index.shards {
            let shard_id = shard.shard_id;
            let p = dir.join(format!("{}.{:02}.brain", name, shard_id));
            if !p.exists() {
                return Err(anyhow!("Missing shard file for shard {}: {}", shard_id, p.display()));
            }
            shard_paths.push(p);
        }

        // The base shard is a standard BrainPack container.
        let base = BrainPack::read(&shard_paths[0])?;

        Ok(Self { index, base, shard_paths })
    }

    /// Where does `tensor_name` live? Returns None if unknown.
    pub fn locate(&self, tensor_name: &str) -> Option<TensorLocation> {
        let shard_id = *self.index.routes.get(tensor_name)?;
        if shard_id == 0 {
            // Resident in the base BrainPack; offsets are in-memory.
            let t = self.base.manifest.tensors.iter().find(|t| t.name == tensor_name)?;
            return Some(TensorLocation { shard_id, file_offset: t.offset, size_bytes: t.size_bytes });
        }
        // Streaming shard. The shard file header is:
        //   u64 tensor_count, then tensor_count × TensorIndexEntry.
        // Each entry's `reserved` (low 32 bits) holds the global manifest
        // index, used here to map the entry to its tensor name.
        let path = self.shard_paths.get(shard_id as usize)?;
        let mut file = File::open(path).ok()?;
        let mut count_buf = [0u8; 8];
        if file.read_exact(&mut count_buf).is_err() {
            return None;
        }
        let count = u64::from_le_bytes(count_buf) as usize;
        for _ in 0..count {
            let mut buf = [0u8; TensorIndexEntry::SIZE];
            if file.read_exact(&mut buf).is_err() {
                return None;
            }
            let entry = TensorIndexEntry::from_bytes(&buf);
            let global = entry.reserved as usize;
            if let Some(t) = self.base.manifest.tensors.get(global) {
                if t.name == tensor_name {
                    return Some(TensorLocation {
                        shard_id,
                        file_offset: entry.offset,
                        size_bytes: entry.size_bytes,
                    });
                }
            }
        }
        None
    }

    /// Convenience: names of all resident (shard 0) tensors.
    pub fn resident_tensor_names(&self) -> Vec<&str> {
        self.index.routes.iter()
            .filter_map(|(n, &s)| if s == 0 { Some(n.as_str()) } else { None })
            .collect()
    }
}

// ============================================================================
// Tensor View (Zero-Copy)
// ============================================================================

#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    pub data: &'a [u8],
    pub shape: Vec<u64>,
    pub dtype: DataType,
    pub quantization: Option<QuantizationParams>,
    pub quantization_type: QuantizationScheme,
}

impl<'a> TensorView<'a> {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }

    pub fn as_f32(&self) -> Result<Vec<f32>> {
        match self.dtype {
            DataType::F32 => {
                let data = self.data;
                if data.len() % 4 != 0 {
                    return Err(anyhow!("F32 data length not multiple of 4"));
                }
                let mut result = Vec::with_capacity(data.len() / 4);
                for chunk in data.chunks_exact(4) {
                    result.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Ok(result)
            }
            DataType::F16 => {
                let data = self.data;
                if data.len() % 2 != 0 {
                    return Err(anyhow!("F16 data length not multiple of 2"));
                }
                let mut result = Vec::with_capacity(data.len() / 2);
                for chunk in data.chunks_exact(2) {
                    let val = f16::from_le_bytes([chunk[0], chunk[1]]);
                    result.push(val.to_f32());
                }
                Ok(result)
            }
            _ => Err(anyhow!("Unsupported dtype for f32 conversion: {:?}", self.dtype)),
        }
    }

    pub fn as_f16(&self) -> Result<Vec<f16>> {
        match self.dtype {
            DataType::F16 => {
                let data = self.data;
                if data.len() % 2 != 0 {
                    return Err(anyhow!("F16 data length not multiple of 2"));
                }
                let mut result = Vec::with_capacity(data.len() / 2);
                for chunk in data.chunks_exact(2) {
                    result.push(f16::from_le_bytes([chunk[0], chunk[1]]));
                }
                Ok(result)
            }
            _ => Err(anyhow!("Unsupported dtype for f16 conversion: {:?}", self.dtype)),
        }
    }

    pub fn dequantize_q4_k(&self) -> Result<Vec<f32>> {
        // Q4_K_M / Q4_K_S dequantization
        // This is a simplified version - real implementation matches llama.cpp
        let num_elements = self.num_elements();
        let mut result = vec![0.0f32; num_elements];

        // Implementation depends on exact quantization format
        // For now, return zeros - real implementation in compute kernels
        warn!("Q4_K dequantization not fully implemented in host code");
        Ok(result)
    }
}

// ============================================================================
// CLI Interface
// ============================================================================

#[derive(Parser)]
#[command(name = "brain-pack", version, about = "Compile models to .brain format")]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Pack tensors into .brain file
    Pack {
        #[arg(short, long, value_name = "FILE")]
        output: String,

        #[arg(short, long, value_name = "FILE")]
        manifest: String,

        #[arg(short, long, value_name = "DIR")]
        weights_dir: String,

        /// Emit a sharded layout: one static `00.brain` plus exactly 15
        /// streaming `01.brain` .. `15.brain` fragments, each capped at
        /// 2 GiB, plus a `<base>.shard.idx` routing table.
        #[arg(long, default_value_t = false)]
        shard: bool,
    },

    /// Unpack .brain file to tensors
    Unpack {
        #[arg(value_name = "FILE")]
        input: String,

        #[arg(short, long, value_name = "DIR")]
        output_dir: String,
    },

    /// Verify .brain file integrity
    Verify {
        #[arg(value_name = "FILE")]
        input: String,
    },

    /// List tensors in .brain file
    List {
        #[arg(value_name = "FILE")]
        input: String,
    },

    /// Download model from Hugging Face Hub
    Download {
        /// Hugging Face repo ID (e.g., "meta-llama/Llama-2-7b-hf")
        #[arg(short, long, value_name = "REPO_ID")]
        repo: String,

        /// Specific files to download (default: all .safetensors files)
        #[arg(short, long, value_name = "FILE", num_args = 1..)]
        files: Option<Vec<String>>,

        /// Output directory
        #[arg(short, long, value_name = "DIR", default_value = ".")]
        output: String,

        /// Hugging Face token for private/gated repos
        #[arg(long, value_name = "TOKEN")]
        token: Option<String>,

        /// Git revision (branch, tag, commit)
        #[arg(long, value_name = "REV", default_value = "main")]
        revision: String,

        /// Maximum concurrent chunk downloads
        #[arg(long, value_name = "N", default_value_t = 4)]
        max_chunks: usize,

        /// Chunk size in bytes
        #[arg(long, value_name = "BYTES", default_value_t = 1048576)]
        chunk_size: u64,
    },

    /// Convert .safetensors to .brain format (optionally sharded)
    Convert {
        /// Input .safetensors file
        #[arg(short, long, value_name = "FILE")]
        input: String,

        /// Output directory
        #[arg(short, long, value_name = "DIR")]
        output: String,

        /// Model name for output files
        #[arg(long, value_name = "NAME")]
        name: Option<String>,

        /// Emit sharded layout (00.brain + 15 streaming shards + .shard.idx)
        #[arg(long, default_value_t = false)]
        shard: bool,

        /// Max bytes per streaming shard
        #[arg(long, value_name = "BYTES", default_value_t = 2147483648)]
        shard_max_bytes: u64,

        /// Number of streaming shards
        #[arg(long, value_name = "N", default_value_t = 15)]
        num_shards: u16,
    },
}

pub fn run_cli() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Pack { output, manifest, weights_dir, shard } => {
            pack_command(output, manifest, weights_dir, shard)?;
        }
        Commands::Unpack { input, output_dir } => {
            unpack_command(input, output_dir)?;
        }
        Commands::Verify { input } => {
            verify_command(input)?;
        }
        Commands::List { input } => {
            list_command(input)?;
        }
        Commands::Download { repo, files, output, token, revision, max_chunks, chunk_size } => {
            download_command(repo, files, output, token, revision, max_chunks, chunk_size)?;
        }
        Commands::Convert { input, output, name, shard, shard_max_bytes, num_shards } => {
            convert_command(input, output, name, shard, shard_max_bytes, num_shards)?;
        }
    }
    Ok(())
}

fn pack_command(output: String, manifest_path: String, weights_dir: String, shard: bool) -> Result<()> {
    let manifest_content = std::fs::read_to_string(&manifest_path)?;
    let manifest: Manifest = serde_json::from_str(&manifest_content)?;

    let mut builder = BrainPackBuilder::new()
        .model(manifest.model.clone())
        .metadata(manifest.metadata.clone());

    for tensor in &manifest.tensors {
        let weight_path = std::path::Path::new(&weights_dir).join(&tensor.name);
        let data = std::fs::read(&weight_path)
            .with_context(|| format!("Failed to read weight file: {}", weight_path.display()))?;

        if tensor.quantization.is_some() {
            // For quantized tensors, quantization params might be in separate file
            let qparams_path = weight_path.with_extension("qparams");
            let qparams_data = if qparams_path.exists() {
                std::fs::read(&qparams_path)?
            } else {
                Vec::new()
            };
            builder = builder.add_tensor_with_quant(tensor.clone(), &data, &qparams_data)?;
        } else {
            builder = builder.add_tensor(tensor.clone(), &data)?;
        }
    }

    let pack = builder.build()?;

    if shard {
        let out = std::path::Path::new(&output);
        let out_dir = out.parent().unwrap_or_else(|| std::path::Path::new(".")).to_path_buf();
        let base_name = out.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "model".to_string());
        let idx = write_sharded(&pack, &out_dir, &base_name)?;
        info!(
            "Sharded layout: base {}B resident, {} streaming shards (cap {}B each), {} routes",
            idx.base_header_size, idx.num_shards, idx.per_shard_max, idx.routes.len()
        );
    } else {
        pack.write(&output)?;
    }

    info!("Successfully packed {} tensors to {}", manifest.tensors.len(), output);
    Ok(())
}

fn unpack_command(input: String, output_dir: String) -> Result<()> {
    std::fs::create_dir_all(&output_dir)?;

    let pack = BrainPack::read(&input)?;

    // Write manifest
    let manifest_json = serde_json::to_vec_pretty(&pack.manifest)?;
    std::fs::write(std::path::Path::new(&output_dir).join("manifest.json"), manifest_json)?;

    // Write each tensor
    for tensor in &pack.manifest.tensors {
        let view = pack.get_tensor(&tensor.name)?;
        let weight_path = std::path::Path::new(&output_dir).join(&tensor.name);
        std::fs::write(&weight_path, view.data)?;

        if let Some(qparams) = &tensor.quantization {
            if qparams.scales_size > 0 {
                let qparams_path = weight_path.with_extension("qparams");
                let qparams_data = &pack.weights[qparams.scales_offset as usize..][..qparams.scales_size as usize];
                std::fs::write(qparams_path, qparams_data)?;
            }
        }
    }

    info!("Unpacked {} tensors to {}", pack.manifest.tensors.len(), output_dir);
    Ok(())
}

fn verify_command(input: String) -> Result<()> {
    let pack = BrainPack::read(&input)?;
    info!("Verification PASSED for {}", input);
    info!("Model: {} ({})", pack.manifest.model.name, pack.manifest.model.architecture);
    info!("Tensors: {}", pack.manifest.tensors.len());
    info!("Weights size: {} bytes", pack.weights.len());
    info!("Checksum: {}", pack.manifest.metadata.checksum);
    Ok(())
}

fn list_command(input: String) -> Result<()> {
    let path = std::path::Path::new(&input);

    // Check if it's a shard index file
    if path.extension().and_then(|s| s.to_str()) == Some("idx") {
        let index = ShardIndex::read(&input)?;
        println!("Shard Index: {}", input);
        println!("  Version: {}", index.version);
        println!("  Number of shards: {}", index.num_shards);
        println!("  Base header size: {} bytes", index.base_header_size);
        println!("  Per-shard max: {} bytes", index.per_shard_max);
        println!("  Routes: {} tensors", index.routes.len());
        println!();
        println!("{:<40} {:<10} {:<12} {:<12}", "NAME", "SHARD", "SHARD_ID", "OFFSET");
        println!("{}", "-".repeat(80));

        for (name, &shard_id) in &index.routes {
            if let Some(shard) = index.shards.get(shard_id as usize) {
                let byte_length = shard.byte_length;
                println!("{:<40} {:<10} {:<12} {:<12}",
                    name,
                    format!("shard_{}", shard_id),
                    shard_id,
                    byte_length
                );
            } else {
                println!("{:<40} {:<10} {:<12} {:<12}", name, format!("shard_{}", shard_id), shard_id, "N/A");
            }
        }
    } else {
        let pack = BrainPack::read(&input)?;

        println!("{:<40} {:<15} {:<12} {:>12} {:>10}", "NAME", "DTYPE", "QUANT", "SHAPE", "SIZE");
        println!("{}", "-".repeat(100));

        for tensor in &pack.manifest.tensors {
            let shape_str = format!("[{}]", tensor.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","));
            let size_mb = tensor.size_bytes as f64 / (1024.0 * 1024.0);
            println!("{:<40} {:<15} {:<12} {:>12} {:>9.2} MB",
                tensor.name,
                format!("{:?}", tensor.dtype),
                format!("{:?}", tensor.quantization_type),
                shape_str,
                size_mb
            );
        }
    }
    Ok(())
}

fn download_command(
    repo: String,
    files: Option<Vec<String>>,
    output: String,
    token: Option<String>,
    revision: String,
    max_chunks: usize,
    chunk_size: u64,
) -> Result<()> {
    use crate::downloader::{download_model, DownloadConfig};

    let rt = tokio::runtime::Runtime::new()?;

    let file_refs: Vec<&str> = files.as_ref().map(|f| f.iter().map(|s| s.as_str()).collect()).unwrap_or_default();

    let config = DownloadConfig {
        repo_id: repo.clone(),
        revision: revision.clone(),
        files: file_refs.iter().map(|s| s.to_string()).collect(),
        output_dir: std::path::PathBuf::from(&output),
        token: token.clone(),
        max_concurrent_chunks: max_chunks,
        chunk_size,
    };

    let progress_rx = rt.block_on(async {
        download_model(&repo, &file_refs, &config.output_dir, config.token.as_deref(), Some(&config.revision)).await
    })?;

    // Print progress
    let pb = indicatif::ProgressBar::new_spinner();
    pb.set_style(
        indicatif::ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap()
    );
    pb.set_message(format!("Downloading {}...", repo));
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    while let Ok(progress) = rt.block_on(progress_rx.recv_async()) {
        if progress.finished {
            if let Some(err) = progress.error {
                pb.finish_with_message(format!("✗ {}: {}", progress.file_name, err));
            } else {
                pb.finish_with_message(format!("✓ {} ({} bytes)", progress.file_name, progress.total_bytes));
            }
        } else {
            pb.set_message(format!(
                "{} {}/{} bytes ({:.1} MB/s)",
                progress.file_name,
                progress.bytes_downloaded,
                progress.total_bytes,
                progress.speed_bps / (1024.0 * 1024.0)
            ));
        }
    }

    info!("Download complete for {}", repo);
    Ok(())
}

fn convert_command(
    input: String,
    output: String,
    name: Option<String>,
    shard: bool,
    shard_max_bytes: u64,
    num_shards: u16,
) -> Result<()> {
    use crate::safetensors::SafetensorsReader;
    use crate::{BrainPackBuilder, DataType};
    use std::collections::HashMap;

    let reader = SafetensorsReader::open(&input)?;
    let header = reader.header();

    let model_name = name.unwrap_or_else(|| {
        header.metadata.as_ref()
            .and_then(|m| m.get("model_name"))
            .cloned()
            .unwrap_or_else(|| "converted-model".to_string())
    });

    let model_info = crate::safetensors::convert::build_model_info(header);

    let mut builder = BrainPackBuilder::new()
        .model(model_info)
        .metadata(Metadata {
            created_epoch: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            created_by: "brain-pack convert".to_string(),
            checksum: String::new(),
            license: "unknown".to_string(),
            description: format!("Converted from {}", input),
        });

    let tensor_names = reader.tensor_names();
    let tensor_names_owned: Vec<String> = tensor_names.iter().map(|s| s.to_string()).collect();
    let mut tensor_offsets: HashMap<String, u64> = HashMap::new();
    let mut offset: u64 = 0;

    for name in &tensor_names_owned {
        let info = reader.tensor_info(name).unwrap();
        let dtype = crate::safetensors::SafetensorsDtype::from_str(&info.dtype).unwrap();
        let size = info.shape.iter().product::<u64>() * dtype.size_bytes() as u64;
        tensor_offsets.insert(name.clone(), offset);
        offset += size;
    }

    for name in &tensor_names_owned {
        let info = reader.tensor_info(&name).unwrap();
        let dtype = crate::safetensors::SafetensorsDtype::from_str(&info.dtype).unwrap();

        let brain_dtype = match dtype {
            crate::safetensors::SafetensorsDtype::F32 => DataType::F32,
            crate::safetensors::SafetensorsDtype::F16 => DataType::F16,
            crate::safetensors::SafetensorsDtype::BF16 => DataType::BF16,
            crate::safetensors::SafetensorsDtype::I32 => DataType::I32,
            crate::safetensors::SafetensorsDtype::I64 => DataType::I64,
            crate::safetensors::SafetensorsDtype::I8 => DataType::I8,
            crate::safetensors::SafetensorsDtype::U8 => DataType::U8,
            crate::safetensors::SafetensorsDtype::Bool => DataType::Bool,
            crate::safetensors::SafetensorsDtype::F8E4M3 => DataType::F8E4M3,
            crate::safetensors::SafetensorsDtype::F8E5M2 => DataType::F8E5M2,
            _ => return Err(anyhow::anyhow!("Unsupported dtype for brain format: {:?}", dtype)),
        };

        let data = reader.tensor_data(&name)?;
        let tensor_info = crate::safetensors::convert::tensor_info_to_brain(
            &name,
            info,
            tensor_offsets[name.as_str()],
            None,
        )?;

        builder = builder.add_tensor(tensor_info, data)?;
    }

    let pack = builder.build()?;

    if shard {
        let out_dir = std::path::Path::new(&output);
        std::fs::create_dir_all(out_dir)?;
        let base_name = model_name.clone();
        let idx = crate::write_sharded(&pack, out_dir, &base_name)?;
        info!(
            "Sharded layout: base {}B resident, {} streaming shards (cap {}B each), {} routes",
            idx.base_header_size, idx.num_shards, idx.per_shard_max, idx.routes.len()
        );
    } else {
        pack.write(&output)?;
    }

    info!("Successfully converted {} to {} ({} tensors)", input, output, tensor_names_owned.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("test.brain");

        let model = ModelInfo {
            name: "test-model".to_string(),
            architecture: "test".to_string(),
            parameter_count: 1000,
            quantization: "f16".to_string(),
            context_length: 2048,
            vocab_size: 32000,
        };

        let metadata = Metadata {
            created_epoch: 1234567890,
            created_by: "test".to_string(),
            checksum: String::new(),
            license: "test".to_string(),
            description: "test model".to_string(),
        };

        // Create some fake tensor data
        let tensor_data: Vec<u8> = (0..100).map(|i| (i % 256) as u8).collect();
        let tensor_data2: Vec<u8> = (100..200).map(|i| (i % 256) as u8).collect();

        let builder = BrainPackBuilder::new()
            .model(model.clone())
            .metadata(metadata.clone())
            .add_tensor(TensorInfo {
                name: "layer1.weight".to_string(),
                shape: vec![10, 10],
                dtype: DataType::F16,
                offset: 0,
                size_bytes: 0,
                quantization: None,
                quantization_type: QuantizationScheme::None,
            }, &tensor_data).unwrap()
            .add_tensor(TensorInfo {
                name: "layer2.weight".to_string(),
                shape: vec![10, 10],
                dtype: DataType::F16,
                offset: 0,
                size_bytes: 0,
                quantization: None,
                quantization_type: QuantizationScheme::None,
            }, &tensor_data2).unwrap();

        let pack = builder.build().unwrap();
        pack.write(&output_path).unwrap();

        // Read back
        let pack2 = BrainPack::read(&output_path).unwrap();

        assert_eq!(pack2.manifest.model.name, "test-model");
        assert_eq!(pack2.manifest.tensors.len(), 2);

        let t1 = pack2.get_tensor("layer1.weight").unwrap();
        assert_eq!(t1.data.len(), 100);

        let t2 = pack2.get_tensor("layer2.weight").unwrap();
        assert_eq!(t2.data.len(), 100);
    }

    #[test]
    fn test_magic_bytes() {
        assert_eq!(BRAIN_MAGIC, *b"BRAIN.1\0");
    }

    #[test]
    fn test_tensor_index_entry_size() {
        assert_eq!(std::mem::size_of::<TensorIndexEntry>(), 144);
    }

    #[test]
    fn test_dtype_alignment() {
        assert_eq!(DataType::F32.alignment(), 4);
        assert_eq!(DataType::F16.alignment(), 2);
        assert_eq!(DataType::Q4KM.alignment(), 64);
    }
}