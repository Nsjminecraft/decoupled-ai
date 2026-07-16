# DeCoupled-AI Binary Container Specification
## `.brain` / `.param` Binary File Format v1.0

---

## 1. File Layout Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  MAGIC BYTES (8 bytes)                                          │
│  0x42 0x52 0x41 0x49 0x4E 0x2E 0x31 0x00  = "BRAIN.1\0"         │
├─────────────────────────────────────────────────────────────────┤
│  JSON METADATA LENGTH (8 bytes, little-endian u64)              │
│  Length of the JSON manifest string in bytes                    │
├─────────────────────────────────────────────────────────────────┤
│  JSON MANIFEST STRING (variable, UTF-8)                         │
│  Null-terminated JSON object (see §2)                           │
├─────────────────────────────────────────────────────────────────┤
│  TENSOR LAYOUT INDEX MAP (variable)                             │
│  Array of TensorIndexEntry (see §3)                             │
├─────────────────────────────────────────────────────────────────┤
│  RAW QUANTIZED WEIGHTS ARRAY (variable, aligned to 64 bytes)    │
│  Concatenated raw tensor data, 64-byte aligned                  │
└─────────────────────────────────────────────────────────────────┘
```

**Alignment Requirements:**
- File header (magic + length) = 16 bytes, no padding
- JSON Manifest: null-terminated, no alignment requirement
- Tensor Index Map: starts at next 8-byte boundary after JSON
- Weights Array: starts at next 64-byte boundary after index map

---

## 2. JSON Manifest Schema

```json
{
  "version": 1,
  "format": "brain",
  "model": {
    "name": "string",
    "architecture": "llama|gpt|mistral|mixtral|custom",
    "parameter_count": 7000000000,
    "quantization": "q4_k_m|q4_0|q8_0|f16|f32|bf16",
    "context_length": 8192,
    "vocab_size": 128256
  },
  "tensors": [
    {
      "name": "string",
      "shape": [int, ...],
      "dtype": "q4_k_m|q4_0|q8_0|f16|f32|bf16",
      "offset": 0,
      "size_bytes": 0,
      "quantization_params": {
        "scales_offset": 0,
        "scales_size": 0,
        "zero_points_offset": 0,
        "zero_points_size": 0,
        "block_size": 32
      }
    }
  ],
  "metadata": {
    "created_epoch": 1704067200,
    "created_by": "brain-pack v1.0",
    "checksum": "sha256:hex_string",
    "license": "string",
    "description": "string"
  }
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | u32 | Yes | Format version (currently 1) |
| `format` | string | Yes | Must be `"brain"` |
| `model.name` | string | Yes | Human-readable model name |
| `model.architecture` | enum | Yes | Architecture family identifier |
| `model.parameter_count` | u64 | Yes | Total parameter count |
| `model.quantization` | enum | Yes | Primary quantization scheme |
| `model.context_length` | u32 | Yes | Max context window |
| `model.vocab_size` | u32 | Yes | Vocabulary size |
| `tensors[].name` | string | Yes | Unique tensor identifier |
| `tensors[].shape` | u32[] | Yes | Tensor dimensions (row-major) |
| `tensors[].dtype` | enum | Yes | Per-tensor dtype (may differ from model default) |
| `tensors[].offset` | u64 | Yes | Byte offset from start of weights array |
| `tensors[].size_bytes` | u64 | Yes | Size in bytes of this tensor's data |
| `tensors[].quantization_params` | object | Conditional | Required for quantized dtypes |
| `metadata.checksum` | string | Yes | SHA256 of weights array only |

---

## 3. Tensor Layout Index Map

Each entry is a fixed-size struct (64 bytes), packed sequentially:

```c
#pragma pack(push, 1)
typedef struct {
    uint64_t name_offset;        // Offset into JSON string table (or 0 if inline)
    uint64_t name_length;        // Length of name (if not in JSON)
    uint64_t offset;             // Byte offset from start of weights array
    uint64_t size_bytes;         // Size in bytes
    uint32_t ndim;               // Number of dimensions (max 8)
    uint32_t dtype;              // DataType enum (see §4)
    uint32_t quantization_type;  // QuantizationScheme enum (see §5)
    uint32_t reserved;           // Reserved for future use (must be 0)
    uint64_t shape[8];           // Dimensions (unused dims = 0)
    uint64_t quant_params_offset; // Offset to quantization params in weights array
    uint64_t quant_params_size;   // Size of quantization params
    uint64_t reserved2[2];       // Future expansion
} TensorIndexEntry;  // Exactly 64 bytes
#pragma pack(pop)
```

### DataType Enum (dtype)

```c
typedef enum {
    DT_F32     = 0,
    DT_F16     = 1,
    DT_BF16    = 2,
    DT_Q4_0    = 3,   // 4-bit block quantized (llama.cpp style)
    DT_Q4_K_M  = 4,   // 4-bit k-quant medium
    DT_Q4_K_S  = 5,   // 4-bit k-quant small
    DT_Q5_0    = 6,
    DT_Q5_1    = 7,
    DT_Q8_0    = 8,
    DT_Q8_K    = 9,
    DT_I8      = 10,
    DT_I16     = 11,
    DT_I32     = 12,
} DataType;
```

### QuantizationScheme Enum

```c
typedef enum {
    QS_NONE      = 0,
    QS_Q4_0      = 1,
    QS_Q4_K_M    = 2,
    QS_Q4_K_S    = 3,
    QS_Q5_0      = 4,
    QS_Q5_1      = 5,
    QS_Q8_0      = 6,
    QS_Q8_K      = 7,
} QuantizationScheme;
```

---

## 4. Quantization Parameter Layout

For quantized tensors (`dtype >= DT_Q4_0`), quantization parameters are stored **inline in the weights array** at `quant_params_offset`:

### Q4_K_M / Q4_K_S Block Layout (per block of 256 elements)
```c
// Block size: 256 elements = 32 scales + 32 min/max + 128 packed weights + 32 scales (high bits)
typedef struct {
    uint8_t scales[32];        // 6-bit scales (per 8 elements)
    uint8_t mins[32];          // Min values (per 8 elements)
    uint8_t maxs[32];          // Max values (per 8 elements)
    uint8_t weights[128];      // 4-bit packed weights (256 elements)
    uint8_t high_bits[32];     // High 2 bits for Q4_K_M (32 groups of 8)
} Q4KBlock;  // Total: 256 bytes per 256 elements
```

### Q8_0 Block Layout (per block of 32 elements)
```c
typedef struct {
    float scale;           // FP32 scale
    int8_t weights[32];    // INT8 weights
} Q8_0Block;  // 36 bytes per 32 elements
```

---

## 5. File Integrity & Validation

### Magic Bytes Verification
```c
static const uint8_t BRAIN_MAGIC[8] = {0x42, 0x52, 0x41, 0x49, 0x4E, 0x2E, 0x31, 0x00};
// "BRAIN.1\0"
```

### Checksum Verification
- `metadata.checksum` = SHA256(weights_array_bytes)
- Verified on every `mmap` load before tensor access

### Alignment Checks
```c
// Verify weights array starts at 64-byte boundary
assert((weights_ptr - file_base) % 64 == 0);

// Verify each tensor offset is aligned to its dtype requirement
assert(tensor.offset % dtype_alignment(tensor.dtype) == 0);
```

---

## 6. Memory Mapping Strategy

### POSIX (Linux/macOS)
```c
int fd = open(path, O_RDONLY);
size_t fsize = lseek(fd, 0, SEEK_END);
void* base = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
// Parse header at base, then tensor_index at calculated offset
```

### Windows
```c
HANDLE hFile = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
HANDLE hMap = CreateFileMappingW(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
void* base = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
// Parse identically to POSIX
```

### Zero-Copy Tensor Access
```c
// Direct pointer into mmap'd region - NO COPY
TensorView tensor = {
    .data = (uint8_t*)base + tensor_index.offset,
    .shape = tensor_index.shape,
    .dtype = tensor_index.dtype,
    .strides = compute_strides(tensor_index.shape),
};
```

---

## 7. Versioning & Forward Compatibility

| Version | Magic Bytes | Changes |
|---------|-------------|---------|
| 1 | `BRAIN.1\0` | Initial specification |

**Forward Compatibility Rules:**
- Readers MUST ignore unknown `dtype` / `quantization_type` values (treat as opaque)
- Readers MUST ignore unknown fields in JSON manifest
- Writers MUST zero all `reserved` fields
- New tensor index fields appended to struct (size increases) - readers use `sizeof(TensorIndexEntry)` from their version

---

## 8. Example: Minimal Valid `.brain` File (Hex Dump)

```
Offset  Hex Dump                                              ASCII
0000:   42 52 41 49 4E 2E 31 00  3C 00 00 00 00 00 00 00   BRAIN.1.<.......
0010:   7B 22 76 65 72 73 69 6F  6E 22 3A 31 2C 22 66 6F   {"version":1,"fo
0020:   72 6D 61 74 22 3A 22 62  72 61 69 6E 22 2C 22 6D   rmat":"brain","m
0030:   6F 64 65 6C 22 3A 7B 22  6E 61 6D 65 22 3A 22 74   odel":{"name":"t
0040:   69 6E 79 22 2C 22 61 72  63 68 69 74 65 63 74 75   iny","architectu
...     (JSON manifest continues) ...
003C:   00                                                    (null terminator)
0040:   [TensorIndexEntry x N]                                (64 bytes each)
...     (aligned to 64-byte boundary)
XXXX:   [Raw Weights Data]                                    (aligned to 64-byte boundary)
```

---

## 9. Reference Implementation Checklist

- [ ] `brain-pack` CLI: Serialize tensors + manifest → `.brain`
- [ ] `brain-unpack` CLI: Deserialize `.brain` → tensors + manifest
- [ ] `mmap` loader: Zero-copy tensor access on POSIX
- [ ] `MapViewOfFile` loader: Zero-copy tensor access on Windows
- [ ] Validation: Magic bytes, checksum, alignment, version
- [ ] Quantization param parsing for all supported schemes
- [ ] Forward-compatible tensor index parsing

---

*Specification Version: 1.0 | Generated for DeCoupled-AI Project*