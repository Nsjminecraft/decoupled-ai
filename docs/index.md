# DeCoupled-AI Documentation

Welcome to DeCoupled-AI, a high-performance LLM inference server with speculative decoding.

## Table of Contents

- [Getting Started](getting-started.md)
- [Installation](installation.md)
- [Configuration](configuration.md)
- [API Reference](api-reference.md)
- [Speculative Decoding](speculative-decoding.md)
- [Model Management](model-management.md)
- [Performance Tuning](performance-tuning.md)
- [Troubleshooting](troubleshooting.md)

## Quick Start

```bash
# Install via universal installer
curl -sSfL https://github.com/nsjminecraft/DeCoupled-AI/releases/latest/download/install.sh | sh

# Start the server
decoupled-ai-server

# Open dashboard
open http://localhost:8080
```

## Features

- **High-Performance Inference**: Optimized CPU/GPU backends with Metal, CUDA, and ROCm support
- **Speculative Decoding**: N-gram based draft generation with batched verification
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API endpoints
- **Web Dashboard**: Built-in model management and monitoring UI
- **Multi-Platform**: Native installers for Linux (.deb), Windows (.msi), macOS (.tar.gz)

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Web Dashboard │────▶│  REST/gRPC API  │────▶│  Inference Core │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                    ┌─────────────────┐     ┌────────────▼────────┐
                    │  Model Loader   │◀───▶│  Speculative Engine │
                    └─────────────────┘     └────────────────────┘
```

## Links

- [GitHub Repository](https://github.com/nsjminecraft/DeCoupled-AI)
- [Issue Tracker](https://github.com/nsjminecraft/DeCoupled-AI/issues)
- [Discussions](https://github.com/nsjminecraft/DeCoupled-AI/discussions)