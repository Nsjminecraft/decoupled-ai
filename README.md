# DeCoupled-AI

**High-performance LLM inference server with speculative decoding**

DeCoupled-AI is a production-ready inference server for Large Language Models featuring:
- **Speculative Decoding** with N-gram draft generation and batched target verification
- **Multi-backend support**: CUDA, ROCm, Metal, CPU (WebGPU coming soon)
- **OpenAI-compatible API** for drop-in replacement
- **Embedded web dashboard** for model management and monitoring
- **Native installers** for Linux (.deb), Windows (.msi), and macOS (.tar.gz)
- **Universal bootstrap installer** via `curl | sh`

## Quick Start

### Universal Installer (Linux/macOS)
```bash
curl -sSfL https://github.com/nsjminecraft/DeCoupled-AI/releases/latest/download/install.sh | sh
```

### Linux (Debian/Ubuntu)
```bash
# Download latest .deb from releases
sudo dpkg -i decoupled-ai_1.0.0_amd64.deb
# Service starts automatically
systemctl status decoupled-ai
```

### Windows
```powershell
# Download latest .msi from releases
msiexec /i decoupled-ai-1.0.0-x86_64.msi /quiet
# Or run installer GUI
```

### macOS
```bash
# Download latest .tar.gz from releases
tar -xzf decoupled-ai-1.0.0-x86_64-apple-darwin.tar.gz
./install.sh
```

### Docker
```bash
docker run -d -p 8080:8080 \
  -v $HOME/.cache/decoupled-ai:/var/lib/decoupled-ai \
  decoupled-ai/server:latest
```

## Usage

### Start Server
```bash
# Foreground
decoupled-ai-server

# Daemon mode (systemd on Linux)
decoupled-ai-server --daemon

# With custom config
decoupled-ai-server --config /path/to/config.toml
```

### API Endpoints
```bash
# Health check
curl http://localhost:8080/health

# OpenAI-compatible completions
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3-8b", "prompt": "Hello", "max_tokens": 100}'

# Chat completions
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3-8b", "messages": [{"role": "user", "content": "Hi!"}]}'

# List models
curl http://localhost:8080/v1/models

# Metrics (Prometheus)
curl http://localhost:8080/metrics
```

### Web Dashboard
Open http://localhost:8080 in your browser for:
- Model management (download, delete, configure)
- Real-time inference metrics
- Speculative decoding statistics
- System resource monitoring

## Configuration

Main config: `/etc/decoupled-ai/config.toml` (system) or `~/.config/decoupled-ai/config.toml` (user)

Key sections:
```toml
[server]
port = 8080
workers = 4

[model]
model_path = "/var/lib/decoupled-ai/models"
gpu_layers = -1  # -1 = all, 0 = CPU only

[speculative]
enabled = true
max_draft_tokens = 5
ngram_order = 4

[cuda]
enabled = true
device_id = -1
memory_fraction = 0.9
```

## Model Support

Place models in `$MODEL_PATH` (default: `/var/lib/decoupled-ai/models` or `~/.local/share/decoupled-ai/models`):

- **GGUF** (llama.cpp format): `model-q4_k_m.gguf`
- **Safetensors**: `model.safetensors` + `config.json`
- **PyTorch**: `pytorch_model.bin` + `config.json`

Supported architectures: LLaMA, Mistral, Qwen, Phi, Gemma, CodeLlama, and more.

## Speculative Decoding

DeCoupled-AI implements **draftless N-gram speculative decoding**:

1. **N-gram Indexer**: Sliding window FNV-1a hash index (4-gram → 3-gram → 2-gram → 1-gram backoff)
2. **Draft Generator**: Produces 1-5 candidate tokens from local N-gram statistics
3. **Batched Verifier**: Single forward pass verifies all draft tokens against target model
4. **KV-Cache Mask**: Efficiently adjusts attention mask on partial acceptance/rejection

Typical speedup: **1.5-2.5x** with minimal quality loss.

## Building from Source

```bash
# Requirements: Rust 1.75+, Zig (for cross-compilation), cargo-packager
cargo install cargo-zigbuild cargo-packager

# Build release
cargo build --release --workspace

# Package installers
cargo packager --target x86_64-unknown-linux-gnu --package deb
cargo packager --target x86_64-pc-windows-msvc --package msi
cargo packager --target x86_64-apple-darwin --package tar.gz
```

## Architecture

```
decoupled-ai/
├── brain-pack/         # Model loading & quantization
├── compute-cpu/        # CPU inference backend
├── compute-cuda/       # CUDA backend (optional)
├── compute-metal/      # Metal backend (macOS)
├── compute-rocm/       # ROCm backend (Linux AMD)
├── engine-ipc/         # Inter-process communication
├── server-backend/     # HTTP/gRPC server
├── api-openai/         # OpenAI-compatible API
├── frontend-ui/        # Embedded dashboard (HTML/JS/CSS)
├── stream-cache/       # KV-cache & streaming
├── weight-handle/      # Memory-mapped weights
└── mem-windows/        # Windows memory management
```

## License

MIT OR Apache-2.0

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

- Issues: [GitHub Issues](https://github.com/nsjminecraft/DeCoupled-AI/issues)
- Discussions: [GitHub Discussions](https://github.com/nsjminecraft/DeCoupled-AI/discussions)