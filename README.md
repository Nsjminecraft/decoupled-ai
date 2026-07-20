# DeCoupled-AI

![DeCoupled-AI Banner](https://raw.githubusercontent.com/nsjminecraft/DeCoupled-AI/main/assets/banner.svg)

[![Build Status](https://github.com/nsjminecraft/DeCoupled-AI/workflows/CI/badge.svg)](https://github.com/nsjminecraft/DeCoupled-AI/actions)
[![Latest Release](https://img.shields.io/github/v/release/nsjminecraft/DeCoupled-AI?include_prereleases)](https://github.com/nsjminecraft/DeCoupled-AI/releases)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Rust Version](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg)](https://github.com/nsjminecraft/DeCoupled-AI/releases)

> **High-performance LLM inference server with speculative decoding, multi-backend GPU support, and OpenAI-compatible API**

---

## 🚀 Overview

DeCoupled-AI is a production-ready LLM inference server designed for high-throughput, low-latency serving of large language models. It features **speculative decoding** with N-gram draft generation and batched target verification, delivering 2-3x speedups over standard autoregressive decoding.

### Key Features

| Feature | Description |
|---------|-------------|
| ⚡ **Speculative Decoding** | N-gram draft generation + batched target verification for 2-3x throughput gains |
| 🎮 **Multi-Backend GPU** | CUDA (NVIDIA), ROCm (AMD), Metal (Apple Silicon), CPU fallback with AVX2/AVX-512 |
| 🔌 **OpenAI-Compatible API** | Drop-in replacement for OpenAI `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` |
| 🌐 **Embedded Web Dashboard** | Model management, real-time monitoring, benchmarking UI |
| 📦 **Native Installers** | `.deb` (Debian/Ubuntu), `.rpm` (Fedora/RHEL), `.pkg.tar.zst` (Arch), `.msi` (Windows), `.tar.gz` (macOS), AppImage |
| 🔄 **OTA Updates** | GitHub Releases-based auto-update with background checker |
| 🎯 **GPU Auto-Detection** | Detects all GPUs, scores by capability, prompts for selection when multiple present |

---

## 📋 Quick Start

### Universal Installer (Linux/macOS) — Recommended

```bash
# Latest stable release
curl -fsSL https://raw.githubusercontent.com/nsjminecraft/DeCoupled-AI/master/packaging/universal/install.sh | bash

# Specific version
curl -fsSL https://github.com/nsjminecraft/DeCoupled-AI/releases/download/v1.0.8/install.sh | bash
```

The universal installer (`packaging/universal/install.sh`):
- Detects your OS (Linux/macOS) and architecture (x86_64/ARM64)
- Downloads the appropriate pre-built binary from GitHub Releases
- Installs to `~/.local/bin` (user) or system location with `sudo`
- Creates default configuration at `~/.config/decoupled-ai/config.toml`
- Sets up systemd user service (Linux) or launchd agent (macOS)
- Adds install directory to PATH in your shell config
- Creates an uninstaller at `~/.local/bin/decoupled-ai-uninstall`

### Manual Binary Download

Download the appropriate archive for your platform from [Releases](https://github.com/nsjminecraft/DeCoupled-AI/releases):

| Platform | Architecture | Asset |
|----------|-------------|-------|
| Linux | x86_64 | `decoupled-ai-x86_64-unknown-linux-musl.tar.gz` |
| Linux | ARM64 | `decoupled-ai-aarch64-unknown-linux-musl.tar.gz` |
| macOS | x86_64 | `decoupled-ai-x86_64-apple-darwin.tar.gz` |
| macOS | ARM64 (Apple Silicon) | `decoupled-ai-aarch64-apple-darwin.tar.gz` |

Extract and run:
```bash
tar -xzf decoupled-ai-*.tar.gz
./decoupled-ai-server
```

### Windows

Download `decoupled-ai-x86_64-windows.zip` from [Releases](https://github.com/nsjminecraft/DeCoupled-AI/releases), extract, and run `decoupled-ai-server.exe`.

### Start the Server

```bash
# Start with default config (auto-detects GPU)
decoupled-ai-server

# Or use the short CLI command
decoupled-ai

# Or run as systemd service (Linux)
systemctl --user start decoupled-ai
systemctl --user enable decoupled-ai  # Start on login
```

### Embedded Web Dashboard (React 18 + Tailwind)

Open http://localhost:8080 in your browser for the web UI (served from static files).

The dashboard provides:
- **Chat Interface** — Streaming chat with loaded models
- **Model Management** — Load/unload GGUF models, view model info
- **Speculative Decoding** — Configure N-gram draft settings
- **Model Download** — Download models from Hugging Face
- **Settings** — Configure server, GPU, updates, API key

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DeCoupled-AI Server                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   OpenAI     │    │   Embedded   │    │   Model      │                   │
│  │   Compatible │    │   Web UI     │    │   Manager    │                   │
│  │   REST API   │    │   (Axum)     │    │   (Hot Swap) │                   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                   │
│         │                   │                   │                           │
│         └───────────────────┼───────────────────┘                           │
│                             ▼                                               │
│                    ┌──────────────────┐                                    │
│                    │   Engine IPC     │                                    │
│                    │   (Message Bus)  │                                    │
│                    └────────┬─────────┘                                    │
│                             │                                              │
│        ┌────────────────────┼────────────────────┐                        │
│        ▼                    ▼                    ▼                        │
│ ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│ │  CUDA       │     │  ROCm       │     │  Metal      │                  │
│ │  Backend    │     │  Backend    │     │  Backend    │                  │
│ │  (NVIDIA)   │     │  (AMD)      │     │  (Apple)    │                  │
│ └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                  │
│        │                   │                   │                         │
│        └───────────────────┼───────────────────┘                         │
│                            ▼                                              │
│                   ┌─────────────────┐                                     │
│                   │  CPU Fallback   │                                     │
│                   │  (AVX2/AVX-512) │                                     │
│                   └─────────────────┘                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

| Crate | Purpose |
|-------|---------|
| `server-backend` | Axum web server, OpenAI API, web dashboard, OTA updater, GPU detection |
| `engine-ipc` | High-performance message passing between server and inference engine |
| `compute-cpu` | CPU inference with SIMD (AVX2/AVX-512) acceleration |
| `compute-cuda` | NVIDIA CUDA backend (optional feature) |
| `compute-rocm` | AMD ROCm backend (optional feature) |
| `compute-metal` | Apple Metal backend (optional feature) |
| `brain-pack` | Model format (GGUF-compatible), quantization, weight loading |
| `stream-cache` | KV cache management with paging and eviction |
| `weight-handle` | Memory-mapped weight access with reference counting |
| `mem-windows` / `mem-posix` | Cross-platform memory mapping abstractions |

---

## ⚙️ Configuration

### Default Config (`/etc/decoupled-ai/config.toml`)

```toml
host = "auto"
port = 8080
model_dir = "/var/lib/decoupled-ai/models"
backend = "auto"
api_key = "sk-decoupled-ai-dev"
enable_cors = true
max_request_size = 104857600  # 100MB

# GPU auto-detection
gpu_index = null
gpu_interactive = false

# OTA updates
auto_update = true
auto_install_updates = false
update_check_interval = 86400  # 24 hours in seconds

# Auto-load model on startup
auto_load = true

# Speculative decoding
[speculative]
enabled = true
draft_tokens = 5
verification_batch = 8
ngram_order = 3
```

### CUDA Config (`/etc/decoupled-ai/cuda.toml`)

```toml
[cuda]
# Minimum compute capability (e.g., 7.0 = Volta, 8.0 = Ampere, 9.0 = Hopper)
min_compute_capability = 70

# Enable CUDA Graphs for reduced launch overhead
cuda_graphs = true

# Memory pool settings
[allocator]
initial_size_mb = 512
max_size_mb = 0  # 0 = unlimited
growth_factor = 1.5
```

---

## 🔌 API Reference

### Chat Completions (OpenAI Compatible)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-decoupled-ai-dev" \
  -d '{
    "model": "llama-3-8b-q4",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain speculative decoding in 2 sentences."}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": true
  }'
```

### Completions (Legacy)

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b-q4",
    "prompt": "The future of AI is",
    "max_tokens": 100,
    "temperature": 0.8
  }'
```

### Embeddings

```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-small-en-v1.5",
    "input": ["Hello world", "DeCoupled-AI is fast"]
  }'
```

### Model Management

```bash
# List available models
curl http://localhost:8080/api/v1/models

# Load a model
curl -X POST http://localhost:8080/api/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/var/lib/decoupled-ai/models/llama-3-8b-q4.gguf"}'

# Unload a model
curl -X POST http://localhost:8080/api/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model_id": "llama-3-8b-q4"}'
```

### GPU Management

```bash
# Detect available GPUs
curl http://localhost:8080/api/v1/gpu/detect

# Select GPU by index
curl -X POST http://localhost:8080/api/v1/gpu/select \
  -H "Content-Type: application/json" \
  -d '{"gpu_index": 1}'
```

### OTA Updates

```bash
# Check for updates
curl http://localhost:8080/api/v1/update/check

# Install update
curl -X POST http://localhost:8080/api/v1/update/install
```

---

## 🎮 GPU Support Matrix

| Vendor | Backend | Min Version | Features |
|--------|---------|-------------|----------|
| **NVIDIA** | CUDA | 11.8+ | FP16, BF16, INT8, CUDA Graphs, Tensor Cores |
| **AMD** | ROCm | 5.6+ | FP16, BF16, Matrix Cores |
| **Apple** | Metal | macOS 12+ | FP16, ANE (Neural Engine) |
| **Intel** | OpenCL/SYCL | OpenCL 3.0 | FP16, XMX |
| **CPU** | Native | Any x86_64/ARM64 | AVX2, AVX-512, NEON, SVE |

### GPU Auto-Detection

On startup, DeCoupled-AI automatically detects all available GPUs and scores them:

```
[INFO] Detecting GPUs...
[INFO] Found 3 GPU(s):
  [0] NVIDIA RTX 4090 (CUDA 12.4, 24GB, CC 8.9) - Score: 95
  [1] AMD Radeon RX 7900 XTX (ROCm 6.0, 24GB) - Score: 88
  [2] Apple M2 Max (Metal 3, 32GB Unified) - Score: 82
[INFO] Auto-selected: [0] NVIDIA RTX 4090
```

**Interactive selection** (when multiple GPUs):
```bash
decoupled-ai-server --gpu-interactive
```
```
Multiple GPUs detected. Select one:
  [0] NVIDIA RTX 4090 (CUDA) - 24GB - Recommended
  [1] AMD RX 7900 XTX (ROCm) - 24GB
  [2] CPU Fallback (AVX2) - System RAM
Enter choice [0-2]: 
```

---

## 🔄 Over-the-Air (OTA) Updates

DeCoupled-AI includes a complete OTA update system:

### CLI Usage

```bash
# Check for updates
decoupled-ai-server --check-updates

# Auto-update (download + install)
decoupled-ai-server --auto-update --auto-install-updates

# Enable background update checker (runs every 24h)
decoupled-ai-server --auto-update
```

### Configuration

```toml
[updates]
check_interval_hours = 24
auto_install = false
include_prerelease = false
```

### Platform Installers

| Platform | Format | Installation |
|----------|--------|--------------|
| Windows | `.msi` | Silent: `msiexec /i update.msi /quiet /norestart` |
| Linux (Debian) | `.deb` | `dpkg -i update.deb` |
| Linux (RPM) | `.rpm` | `rpm -Uvh update.rpm` |
| Linux (Arch) | `.pkg.tar.zst` | `pacman -U update.pkg.tar.zst` |
| macOS | `.tar.gz` | Extract + run `install.sh` |

---

## 🛠️ Building from Source

### Prerequisites

- **Rust 1.75+** (install via [rustup](https://rustup.rs))
- **CMake 3.20+** (for some dependencies)
- **Platform-specific**:
  - Linux: `build-essential`, `pkg-config`, `libssl-dev`, `clang`
  - Windows: Visual Studio 2022 + Windows SDK
  - macOS: Xcode Command Line Tools

### Build Commands

```bash
# Clone repository
git clone https://github.com/nsjminecraft/DeCoupled-AI.git
cd DeCoupled-AI

# Build all (CPU backend only)
cargo build --release --workspace

# Build with CUDA support (Linux only)
cargo build --release --workspace --features cuda

# Build with ROCm support (Linux only)
cargo build --release --workspace --features rocm

# Build with Metal support (macOS only)
cargo build --release --workspace --features metal

# Build all GPU backends
cargo build --release --workspace --features cuda,rocm,metal

# Run tests
cargo test --release --workspace
```

### Binary Location

```bash
# Server binary
./target/release/decoupled-ai-server

# Run directly
./target/release/decoupled-ai-server --config config/default.toml
```

---

## 📦 Creating Distributable Packages

DeCoupled-AI uses a **universal installer script** for Linux and macOS that works across all distributions. No per-distro packages needed.

### Universal Installer (Linux/macOS)

The installer at `packaging/universal/install.sh` is uploaded to GitHub Releases and can be run directly:

```bash
# Upload to releases (run during release process)
gh release upload v1.0.0 packaging/universal/install.sh
```

Users install with:
```bash
curl -sSfL https://github.com/nsjminecraft/DeCoupled-AI/releases/latest/download/install.sh | sh
```

### Windows

For Windows, distribute the binary in a zip archive:

```bash
# Create Windows release archive
cargo build --release --workspace
zip -j decoupled-ai-windows-x86_64.zip target/release/decoupled-ai-server.exe README.md LICENSE-MIT LICENSE-APACHE
```

Upload `decoupled-ai-windows-x86_64.zip` to GitHub Releases.

### Building Release Binaries

```bash
# Linux (musl static binary - works on all distros)
rustup target add x86_64-unknown-linux-musl aarch64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl --workspace
cargo build --release --target aarch64-unknown-linux-musl --workspace

# macOS
cargo build --release --target x86_64-apple-darwin --workspace --features metal
cargo build --release --target aarch64-apple-darwin --workspace --features metal

# Windows
cargo build --release --target x86_64-pc-windows-msvc --workspace

# Package for release
tar -czf decoupled-ai-x86_64-unknown-linux-musl.tar.gz -C target/x86_64-unknown-linux-musl/release decoupled-ai-server
tar -czf decoupled-ai-aarch64-unknown-linux-musl.tar.gz -C target/aarch64-unknown-linux-musl/release decoupled-ai-server
tar -czf decoupled-ai-x86_64-apple-darwin.tar.gz -C target/x86_64-apple-darwin/release decoupled-ai-server
tar -czf decoupled-ai-aarch64-apple-darwin.tar.gz -C target/aarch64-apple-darwin/release decoupled-ai-server
```

### Release Assets Checklist

When creating a GitHub Release, attach these assets:

| Asset | Platform | Description |
|-------|----------|-------------|
| `decoupled-ai-x86_64-unknown-linux-musl.tar.gz` | Linux x86_64 | Static musl binary |
| `decoupled-ai-aarch64-unknown-linux-musl.tar.gz` | Linux ARM64 | Static musl binary |
| `decoupled-ai-x86_64-apple-darwin.tar.gz` | macOS Intel | Native binary |
| `decoupled-ai-aarch64-apple-darwin.tar.gz` | macOS Apple Silicon | Native binary |
| `decoupled-ai-windows-x86_64.zip` | Windows x86_64 | Binary + licenses |
| `install.sh` | Linux/macOS | Universal installer script |

---

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
FROM rust:1.75-slim as builder
WORKDIR /app
COPY . .
RUN cargo build --release --workspace --features cuda

FROM nvidia/cuda:12.4-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/decoupled-ai-server /usr/bin/
COPY config/ /etc/decoupled-ai/
COPY frontend-ui/assets/ /usr/share/decoupled-ai/assets/
EXPOSE 8080
USER 999:999
ENTRYPOINT ["decoupled-ai-server", "--config", "/etc/decoupled-ai/config.toml"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  decoupled-ai:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - DECOUPLED_AI_CONFIG=/etc/decoupled-ai/config.toml
    volumes:
      - ./models:/var/lib/decoupled-ai/models:ro
      - ./cache:/var/lib/decoupled-ai/cache
      - ./logs:/var/log/decoupled-ai
    ports:
      - "8080:8080"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

---

## 📊 Performance Benchmarks

### Speculative Decoding Speedup

| Model | Standard | Speculative | Speedup |
|-------|----------|-------------|---------|
| Llama-3-8B-Q4 | 45 tok/s | 128 tok/s | **2.8x** |
| Llama-3-70B-Q4 | 12 tok/s | 31 tok/s | **2.6x** |
| Mistral-7B-Q4 | 52 tok/s | 145 tok/s | **2.8x** |
| Phi-3-mini-Q4 | 85 tok/s | 220 tok/s | **2.6x** |

*Measured on RTX 4090, batch size 1, 4096 context*

### Memory Usage

| Model | VRAM (Q4) | System RAM | KV Cache (8K ctx) |
|-------|-----------|------------|-------------------|
| 7B | 4.5 GB | 1 GB | 1.2 GB |
| 13B | 8 GB | 2 GB | 2.4 GB |
| 34B | 20 GB | 4 GB | 6 GB |
| 70B | 40 GB | 8 GB | 12 GB |

---

## 🔧 Advanced Usage

### Custom Model Path

```bash
decoupled-ai-server \
  --config /etc/decoupled-ai/config.toml \
  --model-dir /mnt/models \
  --cache-dir /mnt/cache
```

### Multiple Models (Hot Swapping)

```bash
# Load multiple models at startup
decoupled-ai-server --models llama-3-8b:./models/llama-3-8b.gguf,mistral-7b:./models/mistral-7b.gguf
```

### API Key Authentication

```bash
# Set API key in config or env
export DECOUPLED_AI_API_KEY="sk-your-secret-key"

# Use in requests
curl -H "Authorization: Bearer sk-your-secret-key" ...
```

### TLS/SSL

```toml
[server]
tls_cert = "/etc/decoupled-ai/cert.pem"
tls_key = "/etc/decoupled-ai/key.pem"
```

---

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check GPU detection
decoupled-ai-server --check-updates --gpu-interactive

# Verify drivers
nvidia-smi        # NVIDIA
rocm-smi          # AMD
system_profiler SPDisplaysDataType  # macOS
```

### Out of Memory

```toml
# Reduce batch size and context
[engine]
max_batch_size = 8
max_seq_len = 4096

# Enable CPU offloading
[gpu]
cpu_offload = true
offload_layers = 10
```

### Port Already in Use

```bash
# Change port in config or CLI
decoupled-ai-server --port 8081
```

### Model Loading Fails

```bash
# Verify model format
file model.gguf
# Should show: GGUF format, version 3

# Check compatibility
decoupled-ai-server --verify-model ./model.gguf
```

---

## 🤝 Contributing

### Development Setup

```bash
# Fork and clone
git clone https://github.com/your-username/DeCoupled-AI.git
cd DeCoupled-AI

# Install pre-commit hooks
cargo install cargo-husky
cargo husky install

# Run checks
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and lints (`cargo test --workspace && cargo clippy --workspace`)
5. Commit with conventional commits (`feat: add amazing feature`)
6. Push and open a PR

### Code Style

- **Rust**: `rustfmt` + `clippy` (CI enforced)
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/)
- **Documentation**: Update README and inline docs for new features

---

## 📄 License

Dual-licensed under **MIT OR Apache-2.0** at your option.

- [LICENSE-MIT](LICENSE-MIT)
- [LICENSE-APACHE](LICENSE-APACHE)

---

## 🙏 Acknowledgments

- **llama.cpp** - GGUF format and quantization reference
- **ggml** - Tensor operations and backends
- **Axum** - Web framework
- **Candle** - ML framework inspiration
- **All contributors** - See [CONTRIBUTORS.md](CONTRIBUTORS.md)

---

## 📞 Support & Community

- **Issues**: [GitHub Issues](https://github.com/nsjminecraft/DeCoupled-AI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nsjminecraft/DeCoupled-AI/discussions)
- **Security**: See [SECURITY.md](SECURITY.md) for responsible disclosure

---

<div align="center">
  <strong>DeCoupled-AI</strong> — Making LLM inference fast, accessible, and production-ready.
  <br>
  <sub>Built with ❤️ in Rust</sub>
</div>