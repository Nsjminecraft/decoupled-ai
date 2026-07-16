//! Safetensors Parser Module
//!
//! Native Rust parser for Hugging Face .safetensors format.
//! Reads header metadata and memory-maps weight data for zero-copy access.

use anyhow::{anyhow, Context, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Safetensors file header (first 8 bytes = little-endian u64 header size)
const SAFETENSORS_MAGIC: &[u8] = b"safetensors"; // Not used in format, but common extension

/// Tensor metadata from safetensors header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsTensorInfo {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offsets: [u64; 2], // [begin, end) relative to start of data section
}

/// Parsed safetensors header
/// Note: In the actual safetensors format, tensors are at the root level of the JSON,
/// not nested under a "tensors" key. The "__metadata__" key is optional.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsHeader {
    #[serde(flatten)]
    pub tensors: HashMap<String, SafetensorsTensorInfo>,

    #[serde(rename = "__metadata__")]
    pub metadata: Option<HashMap<String, String>>,
}

/// Data type enumeration matching safetensors spec
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetensorsDtype {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
    F8E4M3,
    F8E5M2,
}

impl SafetensorsDtype {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "F64" | "f64" => Some(Self::F64),
            "F32" | "f32" => Some(Self::F32),
            "F16" | "f16" => Some(Self::F16),
            "BF16" | "bf16" => Some(Self::BF16),
            "I64" | "i64" => Some(Self::I64),
            "I32" | "i32" => Some(Self::I32),
            "I16" | "i16" => Some(Self::I16),
            "I8" | "i8" => Some(Self::I8),
            "U8" | "u8" => Some(Self::U8),
            "BOOL" | "bool" => Some(Self::Bool),
            "F8_E4M3" | "f8_e4m3" => Some(Self::F8E4M3),
            "F8_E5M2" | "f8_e5m2" => Some(Self::F8E5M2),
            _ => None,
        }
    }

    pub fn size_bytes(&self) -> usize {
        match self {
            Self::F64 | Self::I64 => 8,
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::F8E4M3 | Self::F8E5M2 | Self::I16 => 2,
            Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    pub fn alignment(&self) -> usize {
        self.size_bytes()
    }
}

/// Safetensors file reader with memory-mapped data access
pub struct SafetensorsReader {
    file: File,
    mmap: Mmap,
    header: SafetensorsHeader,
    data_start: usize,
    data_end: usize,
}

impl SafetensorsReader {
    /// Open and parse a .safetensors file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(&path).context("Failed to open safetensors file")?;
        let mmap = unsafe { Mmap::map(&file) }.context("Failed to memory-map file")?;

        // Parse header
        let (header, data_start) = Self::parse_header(&mmap)?;
        let data_end = mmap.len();

        Ok(Self {
            file,
            mmap,
            header,
            data_start,
            data_end,
        })
    }

    /// Parse the safetensors header
    /// Format: [u64 header_size][JSON header][data...]
    fn parse_header(mmap: &[u8]) -> Result<(SafetensorsHeader, usize)> {
        if mmap.len() < 8 {
            return Err(anyhow!("File too small for safetensors header"));
        }

        // Read header size (little-endian u64)
        let header_size = u64::from_le_bytes(
            mmap[0..8].try_into().context("Invalid header size bytes")?
        ) as usize;

        if mmap.len() < 8 + header_size {
            return Err(anyhow!("File too small for declared header size"));
        }

        // Parse JSON header
        let header_json = &mmap[8..8 + header_size];
        let header_str = std::str::from_utf8(header_json)
            .context("Header is not valid UTF-8")?;

        let header: SafetensorsHeader = serde_json::from_str(header_str)
            .context("Failed to parse safetensors header JSON")?;

        let data_start = 8 + header_size;

        // Align data start to 8-byte boundary (safetensors spec)
        let data_start = (data_start + 7) & !7;

        Ok((header, data_start))
    }

    /// Get the parsed header
    pub fn header(&self) -> &SafetensorsHeader {
        &self.header
    }

    /// Get tensor info by name
    pub fn tensor_info(&self, name: &str) -> Option<&SafetensorsTensorInfo> {
        self.header.tensors.get(name)
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> Vec<&String> {
        self.header.tensors.keys().collect()
    }

    /// Get raw tensor data as bytes (zero-copy from mmap)
    pub fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        let info = self.tensor_info(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found", name))?;

        let begin = self.data_start + info.data_offsets[0] as usize;
        let end = self.data_start + info.data_offsets[1] as usize;

        if end > self.mmap.len() {
            return Err(anyhow!("Tensor data extends beyond file end"));
        }

        Ok(&self.mmap[begin..end])
    }

    /// Get tensor data as f32 slice (converting from various dtypes)
    pub fn tensor_data_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.tensor_info(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found", name))?;

        let dtype = SafetensorsDtype::from_str(&info.dtype)
            .ok_or_else(|| anyhow!("Unknown dtype: {}", info.dtype))?;

        let data = self.tensor_data(name)?;
        let elem_size = dtype.size_bytes();
        let count = data.len() / elem_size;

        let mut result = Vec::with_capacity(count);
        for chunk in data.chunks_exact(elem_size) {
            let val = match dtype {
                SafetensorsDtype::F32 => {
                    let bytes: [u8; 4] = chunk.try_into().unwrap();
                    f32::from_le_bytes(bytes)
                }
                SafetensorsDtype::F16 => {
                    let bytes: [u8; 2] = chunk.try_into().unwrap();
                    f16::from_le_bytes(bytes).to_f32()
                }
                SafetensorsDtype::BF16 => {
                    let bytes: [u8; 2] = chunk.try_into().unwrap();
                    bf16::from_le_bytes(bytes).to_f32()
                }
                SafetensorsDtype::I32 => {
                    let bytes: [u8; 4] = chunk.try_into().unwrap();
                    i32::from_le_bytes(bytes) as f32
                }
                SafetensorsDtype::I64 => {
                    let bytes: [u8; 8] = chunk.try_into().unwrap();
                    i64::from_le_bytes(bytes) as f32
                }
                _ => return Err(anyhow!("Unsupported dtype for f32 conversion: {:?}", dtype)),
            };
            result.push(val);
        }

        Ok(result)
    }

    /// Get tensor data as i32 slice (with dtype verification)
    pub fn tensor_data_i32(&self, name: &str) -> Result<Vec<i32>> {
        let info = self.tensor_info(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found", name))?;

        let dtype = SafetensorsDtype::from_str(&info.dtype)
            .ok_or_else(|| anyhow!("Unknown dtype: {}", info.dtype))?;

        let data = self.tensor_data(name)?;
        let elem_size = dtype.size_bytes();
        let count = data.len() / elem_size;

        let mut result = Vec::with_capacity(count);
        for chunk in data.chunks_exact(elem_size) {
            let val = match dtype {
                SafetensorsDtype::I32 => {
                    let bytes: [u8; 4] = chunk.try_into().unwrap();
                    i32::from_le_bytes(bytes)
                }
                SafetensorsDtype::I64 => {
                    let bytes: [u8; 8] = chunk.try_into().unwrap();
                    i64::from_le_bytes(bytes) as i32
                }
                _ => return Err(anyhow!("Unsupported dtype for i32 conversion: {:?}", dtype)),
            };
            result.push(val);
        }

        Ok(result)
    }

    /// Get tensor data as i64 slice (with dtype verification)
    pub fn tensor_data_i64(&self, name: &str) -> Result<Vec<i64>> {
        let info = self.tensor_info(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found", name))?;

        let dtype = SafetensorsDtype::from_str(&info.dtype)
            .ok_or_else(|| anyhow!("Unknown dtype: {}", info.dtype))?;

        let data = self.tensor_data(name)?;
        let elem_size = dtype.size_bytes();
        let count = data.len() / elem_size;

        let mut result = Vec::with_capacity(count);
        for chunk in data.chunks_exact(elem_size) {
            let val = match dtype {
                SafetensorsDtype::I64 => {
                    let bytes: [u8; 8] = chunk.try_into().unwrap();
                    i64::from_le_bytes(bytes)
                }
                SafetensorsDtype::I32 => {
                    let bytes: [u8; 4] = chunk.try_into().unwrap();
                    i32::from_le_bytes(bytes) as i64
                }
                _ => return Err(anyhow!("Unsupported dtype for i64 conversion: {:?}", dtype)),
            };
            result.push(val);
        }

        Ok(result)
    }

    /// Get tensor shape
    pub fn tensor_shape(&self, name: &str) -> Option<Vec<u64>> {
        self.tensor_info(name).map(|t| t.shape.clone())
    }

    /// Get tensor dtype
    pub fn tensor_dtype(&self, name: &str) -> Option<SafetensorsDtype> {
        self.tensor_info(name)
            .and_then(|t| SafetensorsDtype::from_str(&t.dtype))
    }

    /// Total file size
    pub fn file_size(&self) -> u64 {
        self.mmap.len() as u64
    }

    /// Data section size
    pub fn data_size(&self) -> u64 {
        (self.data_end - self.data_start) as u64
    }
}

/// Convert safetensors to brain-pack tensor format
pub mod convert {
    use super::*;
    use crate::{TensorInfo, DataType, QuantizationScheme, QuantizationParams, ModelInfo};

    /// Convert safetensors tensor info to brain-pack TensorInfo
    pub fn tensor_info_to_brain(
        name: &str,
        st_info: &SafetensorsTensorInfo,
        offset: u64,
        quantization: Option<QuantizationParams>,
    ) -> Result<TensorInfo> {
        let dtype = SafetensorsDtype::from_str(&st_info.dtype)
            .ok_or_else(|| anyhow!("Unknown safetensors dtype: {}", st_info.dtype))?;

        let brain_dtype = match dtype {
            SafetensorsDtype::F32 => DataType::F32,
            SafetensorsDtype::F16 => DataType::F16,
            SafetensorsDtype::BF16 => DataType::BF16,
            SafetensorsDtype::I32 => DataType::I32,
            SafetensorsDtype::I64 => DataType::I64,
            SafetensorsDtype::I8 => DataType::I8,
            SafetensorsDtype::U8 => DataType::U8,
            SafetensorsDtype::Bool => DataType::Bool,
            SafetensorsDtype::F8E4M3 => DataType::F8E4M3,
            SafetensorsDtype::F8E5M2 => DataType::F8E5M2,
            _ => return Err(anyhow!("Unsupported dtype for brain format: {:?}", dtype)),
        };

        let size_bytes = st_info.shape.iter().product::<u64>() * dtype.size_bytes() as u64;

        Ok(TensorInfo {
            name: name.to_string(),
            shape: st_info.shape.clone(),
            dtype: brain_dtype,
            offset,
            size_bytes,
            quantization,
            quantization_type: QuantizationScheme::None,
        })
    }

    /// Build a basic ModelInfo from safetensors metadata
    pub fn build_model_info(header: &SafetensorsHeader) -> ModelInfo {
        // Try to infer architecture from tensor names
        let arch = infer_architecture(&header.tensors);

        // Calculate total parameter count
        let param_count: u64 = header.tensors.values()
            .map(|t| t.shape.iter().product::<u64>())
            .sum();

        ModelInfo {
            name: header.metadata.as_ref()
                .and_then(|m| m.get("model_name"))
                .cloned()
                .unwrap_or_else(|| "safetensors-model".to_string()),
            architecture: arch,
            parameter_count: param_count,
            quantization: "none".to_string(),
            context_length: infer_context_length(&header.tensors),
            vocab_size: infer_vocab_size(&header.tensors),
        }
    }

    fn infer_architecture(tensors: &HashMap<String, SafetensorsTensorInfo>) -> String {
        // Simple heuristics based on common naming patterns
        let has_attention = tensors.keys().any(|k| k.contains("attn") || k.contains("attention"));
        let has_mlp = tensors.keys().any(|k| k.contains("mlp") || k.contains("feed_forward"));
        let has_embedding = tensors.keys().any(|k| k.contains("embed"));

        if has_attention && has_mlp && has_embedding {
            "llama".to_string() // Default to llama-like
        } else if has_attention {
            "transformer".to_string()
        } else {
            "unknown".to_string()
        }
    }

    fn count_layers(tensors: &HashMap<String, SafetensorsTensorInfo>) -> u32 {
        // Count unique layer indices from tensor names like "layers.0.attention..."
        let mut layers = std::collections::HashSet::new();
        for name in tensors.keys() {
            if let Some(idx) = extract_layer_index(name) {
                layers.insert(idx);
            }
        }
        layers.len() as u32
    }

    fn extract_layer_index(name: &str) -> Option<u32> {
        // Match patterns like "layers.0." or "model.layers.0."
        for part in name.split('.') {
            if let Ok(idx) = part.parse::<u32>() {
                // Check if previous part was "layers" or "layer"
                return Some(idx);
            }
        }
        None
    }

    fn infer_hidden_size(tensors: &HashMap<String, SafetensorsTensorInfo>) -> u32 {
        // Look for attention projection weights
        for (name, info) in tensors {
            if (name.contains("q_proj") || name.contains("k_proj") || name.contains("v_proj"))
                && info.shape.len() == 2
            {
                return info.shape[0] as u32;
            }
        }
        4096 // default
    }

    fn infer_num_heads(tensors: &HashMap<String, SafetensorsTensorInfo>) -> u32 {
        // Typically hidden_size / head_dim
        let hidden = infer_hidden_size(tensors);
        // Common head dims: 64, 128
        if hidden % 128 == 0 { hidden / 128 } else { hidden / 64 }
    }

    fn infer_vocab_size(tensors: &HashMap<String, SafetensorsTensorInfo>) -> u32 {
        for (name, info) in tensors {
            if (name.contains("embed") || name.contains("wte")) && info.shape.len() == 2 {
                return info.shape[0] as u32;
            }
            if name.contains("lm_head") && info.shape.len() == 2 {
                return info.shape[0] as u32;
            }
        }
        32000 // default
    }

    fn infer_context_length(tensors: &HashMap<String, SafetensorsTensorInfo>) -> u32 {
        // Look for positional embeddings or rope config
        for (name, info) in tensors {
            if (name.contains("pos_embed") || name.contains("position_emb")) && info.shape.len() >= 1 {
                return info.shape[0] as u32;
            }
        }
        4096 // default context length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_safetensors() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();

        // Create a minimal valid safetensors file
        // Header format: tensors at root level (not under "tensors" key), optional __metadata__
        let header_json = r#"{"test":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let header_bytes = header_json.as_bytes();
        let header_size = header_bytes.len() as u64;

        // Write header size (u64 LE)
        file.write_all(&header_size.to_le_bytes()).unwrap();
        // Write header JSON
        file.write_all(header_bytes).unwrap();
        // Pad to 8-byte alignment
        let padding = (8 - (header_bytes.len() + 8) % 8) % 8;
        file.write_all(&vec![0u8; padding]).unwrap();
        // Write data (6 f32 values = 24 bytes)
        let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&data_bytes).unwrap();

        file
    }

    #[test]
    fn test_parse_header() {
        let file = create_test_safetensors();
        let reader = SafetensorsReader::open(file.path()).unwrap();

        let header = reader.header();
        assert!(header.tensors.contains_key("test"));

        let info = header.tensors.get("test").unwrap();
        assert_eq!(info.dtype, "F32");
        assert_eq!(info.shape, vec![2, 3]);
        assert_eq!(info.data_offsets, [0, 24]);
    }

    #[test]
    fn test_tensor_data() {
        let file = create_test_safetensors();
        let reader = SafetensorsReader::open(file.path()).unwrap();

        let data = reader.tensor_data("test").unwrap();
        assert_eq!(data.len(), 24);

        let typed = reader.tensor_data_f32("test").unwrap();
        assert_eq!(typed, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_dtype_parsing() {
        assert_eq!(SafetensorsDtype::from_str("F32"), Some(SafetensorsDtype::F32));
        assert_eq!(SafetensorsDtype::from_str("f16"), Some(SafetensorsDtype::F16));
        assert_eq!(SafetensorsDtype::from_str("BF16"), Some(SafetensorsDtype::BF16));
        assert_eq!(SafetensorsDtype::from_str("I32"), Some(SafetensorsDtype::I32));
        assert_eq!(SafetensorsDtype::from_str("UNKNOWN"), None);
    }
}