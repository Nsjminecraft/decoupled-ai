# Installation Guide

## Universal Installer (Recommended)

The easiest way to install DeCoupled-AI on any platform:

```bash
# Latest stable release
curl -sSfL https://github.com/nsjminecraft/DeCoupled-AI/releases/latest/download/install.sh | sh

# Specific version
curl -sSfL https://github.com/nsjminecraft/DeCoupled-AI/releases/download/v1.0.0/install.sh | sh
```

The installer will:
1. Detect your OS and architecture
2. Download the appropriate package
3. Install to `~/.local/bin` (user) or system location (with sudo)
4. Configure PATH automatically
5. Set up default configuration

## Platform-Specific Packages

### Linux (Debian/Ubuntu)

```bash
# Download .deb package
wget https://github.com/nsjminecraft/DeCoupled-AI/releases/download/v1.0.0/decoupled-ai_1.0.0_amd64.deb

# Install
sudo dpkg -i decoupled-ai_1.0.0_amd64.deb
sudo apt-get install -f  # Fix any missing dependencies

# Service management
sudo systemctl start decoupled-ai
sudo systemctl enable decoupled-ai
sudo systemctl status decoupled-ai
```

### Linux (Generic/Static)

```bash
# Download static binary
wget https://github.com/nsjminecraft/DeCoupled-AI/releases/download/v1.0.0/decoupled-ai-x86_64-unknown-linux-musl.tar.gz

# Extract
tar -xzf decoupled-ai-x86_64-unknown-linux-musl.tar.gz

# Run directly
./decoupled-ai-server
```

### Windows

```powershell
# Option 1: MSI Installer (recommended)
msiexec /i decoupled-ai-1.0.0-x86_64.msi

# Option 2: Chocolatey (when available)
choco install decoupled-ai

# Option 3: Scoop (when available)
scoop install decoupled-ai

# Option 4: Manual - download ZIP and extract
# Run decoupled-ai-server.exe from extracted folder
```

### macOS

```bash
# Option 1: Homebrew (when available)
brew install decoupled-ai

# Option 2: Download and extract
curl -L -o decoupled-ai.tar.gz https://github.com/nsjminecraft/DeCoupled-AI/releases/download/v1.0.0/decoupled-ai-x86_64-apple-darwin.tar.gz
tar -xzf decoupled-ai.tar.gz
./decoupled-ai-server

# Option 3: Universal installer
curl -sSfL https://github.com/nsjminecraft/DeCoupled-AI/releases/latest/download/install.sh | sh
```

## Docker

```bash
# Pull image
docker pull decoupled-ai/decoupled-ai:latest

# Run with GPU support (NVIDIA)
docker run --gpus all -p 8080:8080 -v decoupled-ai-data:/var/lib/decoupled-ai decoupled-ai/decoupled-ai:latest

# Run CPU-only
docker run -p 8080:8080 -v decoupled-ai-data:/var/lib/decoupled-ai decoupled-ai/decoupled-ai:latest-cpu
```

## Building from Source

```bash
# Prerequisites
# - Rust 1.75+ (rustup.rs)
# - Zig 0.13+ (for cross-compilation)
# - System dependencies: pkg-config, libssl-dev, clang

git clone https://github.com/nsjminecraft/DeCoupled-AI.git
cd DeCoupled-AI

# Build release
cargo build --release --workspace

# Binary at: target/release/decoupled-ai-server
```

## Post-Installation

1. **Start the server**:
   ```bash
   decoupled-ai-server
   ```

2. **Open the dashboard**:
   Navigate to `http://localhost:8080`

3. **Download a model**:
   Use the dashboard or CLI to download models

4. **Configure** (optional):
   Edit config at:
   - Linux: `/etc/decoupled-ai/config.toml`
   - macOS: `~/.config/decoupled-ai/config.toml`
   - Windows: `%PROGRAMFILES%\DeCoupled-AI\config\default.toml`

## Uninstallation

### Linux (.deb)
```bash
sudo dpkg -r decoupled-ai        # Remove package
sudo dpkg -P decoupled-ai        # Purge (remove config/data)
```

### Windows (MSI)
```powershell
msiexec /x decoupled-ai-1.0.0-x86_64.msi
```

### Universal Installer
```bash
~/.local/bin/decoupled-ai-uninstall
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `command not found` | Add `~/.local/bin` to PATH |
| Permission denied | Check file permissions, run with sudo if needed |
| Port 8080 in use | Change port in config or stop conflicting service |
| GPU not detected | Install CUDA/ROCm drivers, check `nvidia-smi` |
| Model fails to load | Verify model path, check disk space, check format |

## Next Steps

- [Getting Started](getting-started.md) - Your first inference
- [Configuration](configuration.md) - Customize settings
- [API Reference](api-reference.md) - OpenAI-compatible endpoints