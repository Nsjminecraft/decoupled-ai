# Installation Guide

## Universal Installer (Recommended) — Linux & macOS

The easiest way to install DeCoupled-AI on any Linux distribution or macOS:

```bash
# Latest stable release
curl -sSfL https://github.com/nsjminecraft/DeCoupled-AI/releases/latest/download/install.sh | sh

# Specific version
curl -sSfL https://github.com/nsjminecraft/DeCoupled-AI/releases/download/v1.0.0/install.sh | sh
```

Or with a specific install directory:

```bash
INSTALL_DIR=/usr/local/bin curl -sSfL https://github.com/nsjminecraft/DeCoupled-AI/releases/latest/download/install.sh | sh
```

The universal installer (`packaging/universal/install.sh`):
- Detects your OS (Linux/macOS) and architecture (x86_64/ARM64)
- Downloads the appropriate pre-built binary from GitHub Releases
- Installs to `~/.local/bin` (user) or system location with `sudo`
- Creates default configuration at `~/.config/decoupled-ai/config.toml`
- Sets up systemd user service (Linux) or launchd agent (macOS)
- Adds install directory to PATH in your shell config
- Creates an uninstaller at `~/.local/bin/decoupled-ai-uninstall`

### What Gets Installed

| Component | Location |
|-----------|----------|
| Binary | `~/.local/bin/decoupled-ai-server` |
| Config | `~/.config/decoupled-ai/config.toml` |
| Models/Cache | `~/.local/share/decoupled-ai/` |
| Systemd service (Linux) | `~/.config/systemd/user/decoupled-ai.service` |
| Launchd agent (macOS) | `~/Library/LaunchAgents/ai.decoupled.server.plist` |
| Uninstaller | `~/.local/bin/decoupled-ai-uninstall` |

---

## Post-Install: Start the Server

**Linux (systemd):**
```bash
systemctl --user start decoupled-ai
journalctl --user -u decoupled-ai -f  # view logs

# Enable auto-start on login
systemctl --user enable decoupled-ai
loginctl enable-linger $USER  # keep service running after logout
```

**macOS (launchd):**
```bash
launchctl start ai.decoupled.server
tail -f ~/.local/share/decoupled-ai/server.log  # view logs

# Service auto-starts on login
```

**Manual (foreground):**
```bash
decoupled-ai-server --config ~/.config/decoupled-ai/config.toml
```

**Web UI:** http://localhost:8080  
**API:** http://localhost:8080/v1 (OpenAI-compatible)

---

## Uninstall

```bash
~/.local/bin/decoupled-ai-uninstall
```

This stops services, removes the binary, and optionally removes config/data directories.

---

## Manual Binary Download

If you prefer not to use the installer, download the appropriate archive for your platform from [GitHub Releases](https://github.com/nsjminecraft/DeCoupled-AI/releases):

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

---

## Building from Source

Requires Rust 1.75+ and a C toolchain:

```bash
git clone https://github.com/nsjminecraft/DeCoupled-AI.git
cd DeCoupled-AI
cargo build --release --workspace
```

The binary will be at `target/release/decoupled-ai-server`.

---

## GPU Support

The binary auto-detects your GPU backend:

| Platform | Backend | Requirements |
|----------|---------|--------------|
| NVIDIA | CUDA | NVIDIA driver + CUDA toolkit |
| AMD | ROCm | ROCm runtime |
| Apple Silicon | Metal | macOS 12+ |
| Any | CPU (AVX2/AVX-512) | Fallback |

Configure in `~/.config/decoupled-ai/config.toml`:
```toml
[gpu]
backend = "auto"  # cuda, rocm, metal, cpu
device_id = 0
memory_fraction = 0.9
```

---

## Docker

```bash
# Pull image
docker pull decoupled-ai/decoupled-ai:latest

# Run with GPU support (NVIDIA)
docker run --gpus all -p 8080:8080 -v decoupled-ai-data:/var/lib/decoupled-ai decoupled-ai/decoupled-ai:latest

# Run CPU-only
docker run -p 8080:8080 -v decoupled-ai-data:/var/lib/decoupled-ai decoupled-ai/decoupled-ai:latest-cpu
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `command not found` | Add `~/.local/bin` to PATH, restart shell |
| Permission denied | Check file permissions, run with sudo if needed |
| Port 8080 in use | Change port in config or stop conflicting service |
| GPU not detected | Install CUDA/ROCm drivers, check `nvidia-smi` |
| Model fails to load | Verify model path, check disk space, check format |

---

## Next Steps

- [Getting Started](getting-started.md) - Your first inference
- [Configuration](configuration.md) - Customize settings
- [API Reference](api-reference.md) - OpenAI-compatible endpoints