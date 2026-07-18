#!/bin/sh
#
# DeCoupled-AI Universal Installer
# Works on Linux (all distros) and macOS
# Usage: curl -sSfL https://github.com/nsjminecraft/DeCoupled-AI/releases/latest/download/install.sh | sh
#        curl -sSfL https://github.com/nsjminecraft/DeCoupled-AI/releases/download/v1.0.0/install.sh | sh
#

set -eu

# Configuration
REPO="nsjminecraft/DeCoupled-AI"
BINARY_NAME="decoupled-ai-server"
INSTALL_DIR="${INSTALL_DIR:-${HOME}/.local/bin}"
CONFIG_DIR="${CONFIG_DIR:-${HOME}/.config/decoupled-ai}"
DATA_DIR="${DATA_DIR:-${HOME}/.local/share/decoupled-ai}"
SYSTEMD_DIR="${SYSTEMD_DIR:-${HOME}/.config/systemd/user}"
LAUNCHD_DIR="${LAUNCHD_DIR:-${HOME}/Library/LaunchAgents}"

# Colors (only if stdout is a TTY)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NC='\033[0m' # No Color
else
    RED='' GREEN='' YELLOW='' BLUE='' BOLD='' NC=''
fi

log_info() { printf "${BLUE}%s${NC}\n" "$*"; }
log_success() { printf "${GREEN}%s${NC}\n" "$*"; }
log_warn() { printf "${YELLOW}%s${NC}\n" "$*"; }
log_error() { printf "${RED}%s${NC}\n" "$*" >&2; }
log_step() { printf "${BOLD}${BLUE}==>${NC} ${BOLD}%s${NC}\n" "$*"; }

# Detect OS and architecture
detect_platform() {
    OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
    ARCH="$(uname -m)"

    case "$OS" in
        linux) OS="linux" ;;
        darwin) OS="macos" ;;
        *) log_error "Unsupported OS: $OS"; exit 1 ;;
    esac

    case "$ARCH" in
        x86_64|amd64) ARCH="x86_64" ;;
        aarch64|arm64) ARCH="aarch64" ;;
        *) log_error "Unsupported architecture: $ARCH"; exit 1 ;;
    esac

    # Determine asset name pattern
    case "$OS" in
        linux)
            # Prefer musl static binary for maximum compatibility
            ASSET_PATTERN="decoupled-ai-${ARCH}-unknown-linux-musl.tar.gz"
            ;;
        macos)
            ASSET_PATTERN="decoupled-ai-${ARCH}-apple-darwin.tar.gz"
            ;;
    esac

    log_info "Detected platform: ${OS}/${ARCH}"
    log_info "Asset pattern: ${ASSET_PATTERN}"
}

# Get latest release version from GitHub
get_latest_version() {
    log_step "Fetching latest release version..."

    # Try GitHub API first (rate limited but reliable)
    if command -v curl >/dev/null 2>&1; then
        VERSION=$(curl -sSfL "https://api.github.com/repos/${REPO}/releases/latest" \
            | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": "\(.*\)".*/\1/')
    elif command -v wget >/dev/null 2>&1; then
        VERSION=$(wget -qO- "https://api.github.com/repos/${REPO}/releases/latest" \
            | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": "\(.*\)".*/\1/')
    else
        log_error "curl or wget required"
        exit 1
    fi

    if [ -z "$VERSION" ] || [ "$VERSION" = "null" ]; then
        log_error "Could not fetch latest version"
        exit 1
    fi

    log_info "Latest version: ${VERSION}"
}

# Download and verify asset
download_asset() {
    local version="$1"
    local url="https://github.com/${REPO}/releases/download/${version}/${ASSET_PATTERN}"
    local tmp_dir=$(mktemp -d)
    local asset_path="${tmp_dir}/${ASSET_PATTERN}"

    log_step "Downloading ${ASSET_PATTERN}..."
    log_info "URL: ${url}"

    if command -v curl >/dev/null 2>&1; then
        curl -sSfL -o "$asset_path" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$asset_path" "$url"
    else
        log_error "curl or wget required"
        exit 1
    fi

    if [ ! -f "$asset_path" ] || [ ! -s "$asset_path" ]; then
        log_error "Download failed or file is empty"
        log_info "Check if asset exists at: https://github.com/${REPO}/releases/tag/${version}"
        rm -rf "$tmp_dir"
        exit 1
    fi

    log_success "Downloaded $(du -h "$asset_path" | cut -f1)"

    # Extract
    log_step "Extracting..."
    tar -xzf "$asset_path" -C "$tmp_dir"

    # Find binary
    BINARY_PATH=$(find "$tmp_dir" -name "$BINARY_NAME" -type f | head -1)
    if [ -z "$BINARY_PATH" ] || [ ! -f "$BINARY_PATH" ]; then
        log_error "Binary not found in archive"
        find "$tmp_dir" -type f | head -20
        rm -rf "$tmp_dir"
        exit 1
    fi

    chmod +x "$BINARY_PATH"
    TMP_DIR="$tmp_dir"
}

# Install binary
install_binary() {
    log_step "Installing binary to ${INSTALL_DIR}..."

    mkdir -p "$INSTALL_DIR"

    # Check if we need sudo for system-wide install
    if [ "$INSTALL_DIR" = "/usr/local/bin" ] || [ "$INSTALL_DIR" = "/opt/decoupled-ai/bin" ]; then
        if [ "$(id -u)" -ne 0 ]; then
            log_warn "System-wide install requires sudo. Re-running with sudo..."
            exec sudo -E sh "$0" "$@"
        fi
    fi

    cp "$BINARY_PATH" "${INSTALL_DIR}/${BINARY_NAME}"
    log_success "Installed ${BINARY_NAME} to ${INSTALL_DIR}"
}

# Create default config
create_config() {
    log_step "Creating default configuration..."

    mkdir -p "$CONFIG_DIR"
    mkdir -p "$DATA_DIR/models"
    mkdir -p "$DATA_DIR/cache"

    local config_file="${CONFIG_DIR}/config.toml"

    if [ ! -f "$config_file" ]; then
        cat > "$config_file" <<EOF
# DeCoupled-AI Configuration
# Generated by universal installer on $(date)

[server]
host = "127.0.0.1"
port = 8080
workers = 4

[model]
# Default model path (will be created on first model download)
default_path = "${DATA_DIR}/models"
# Auto-load first available model on startup
auto_load = true

[storage]
# Model cache directory
cache_dir = "${DATA_DIR}/cache"
# Database path for metadata
db_path = "${DATA_DIR}/models.db"

[gpu]
# Auto-detect GPU backend (cuda, rocm, metal, cpu)
backend = "auto"
# Prefer specific GPU if multiple available (0, 1, etc.)
device_id = 0
# Memory fraction to use (0.0-1.0)
memory_fraction = 0.9

[api]
# OpenAI-compatible API settings
api_key = "sk-decoupled-ai-dev"
enable_cors = true
max_request_size = "50MB"

[logging]
level = "info"
format = "json"
file = "${DATA_DIR}/server.log"
max_size_mb = 100
max_files = 5
EOF
        log_success "Created config at ${config_file}"
    else
        log_info "Config already exists at ${config_file}, skipping"
    fi
}

# Setup systemd service (Linux)
setup_systemd() {
    if [ "$OS" != "linux" ]; then
        return 0
    fi

    log_step "Setting up systemd user service..."

    mkdir -p "$SYSTEMD_DIR"

    cat > "${SYSTEMD_DIR}/decoupled-ai.service" <<EOF
[Unit]
Description=DeCoupled-AI Inference Server
Documentation=https://github.com/nsjminecraft/DeCoupled-AI
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=${INSTALL_DIR}/${BINARY_NAME} --config ${CONFIG_DIR}/config.toml
Restart=on-failure
RestartSec=5
WorkingDirectory=${DATA_DIR}
Environment=HOME=${HOME}
Environment=RUST_LOG=info

# Security hardening
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=${DATA_DIR} ${CONFIG_DIR}
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
RestrictNamespaces=yes
LockPersonality=yes
MemoryDenyWriteExecute=yes

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=default.target
EOF

    # Reload, enable, and start immediately
    systemctl --user daemon-reload >/dev/null 2>&1 || true
    systemctl --user enable decoupled-ai.service >/dev/null 2>&1 || true
    systemctl --user start decoupled-ai.service >/dev/null 2>&1 || true

    log_success "Systemd service installed, enabled, and started"
    log_info "Start with: systemctl --user start decoupled-ai"
    log_info "View logs: journalctl --user -u decoupled-ai -f"
    log_info "For headless/boot persistence: loginctl enable-linger \$USER"
    log_info "To run as system service (sudo): sudo cp ${SYSTEMD_DIR}/decoupled-ai.service /etc/systemd/system/decoupled-ai.service && sudo systemctl daemon-reload && sudo systemctl enable --now decoupled-ai"
}

# Setup launchd service (macOS)
setup_launchd() {
    if [ "$OS" != "macos" ]; then
        return 0
    fi

    log_step "Setting up launchd user agent..."

    mkdir -p "$LAUNCHD_DIR"

    cat > "${LAUNCHD_DIR}/ai.decoupled.server.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.decoupled.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>${INSTALL_DIR}/${BINARY_NAME}</string>
        <string>--config</string>
        <string>${CONFIG_DIR}/config.toml</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${DATA_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
        <key>Crashed</key>
        <true/>
    </dict>
    <key>StandardOutPath</key>
    <string>${DATA_DIR}/server.log</string>
    <key>StandardErrorPath</key>
    <string>${DATA_DIR}/server-error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>${HOME}</string>
        <key>RUST_LOG</key>
        <string>info</string>
    </dict>
    <key>ProcessType</key>
    <string>Interactive</string>
    <key>Nice</key>
    <integer>10</integer>
</dict>
</plist>
EOF

    # Load the service
    launchctl load "${LAUNCHD_DIR}/ai.decoupled.server.plist" >/dev/null 2>&1 || true

    log_success "Launchd agent installed and loaded"
    log_info "Start with: launchctl start ai.decoupled.server"
    log_info "View logs: tail -f ${DATA_DIR}/server.log"
}

# Setup PATH in shell config
setup_path() {
    log_step "Configuring PATH..."

    local shell_rc=""
    case "$SHELL" in
        */bash) shell_rc="${HOME}/.bashrc" ;;
        */zsh) shell_rc="${HOME}/.zshrc" ;;
        */fish) shell_rc="${HOME}/.config/fish/config.fish" ;;
        *) shell_rc="${HOME}/.profile" ;;
    esac

    local path_entry="${INSTALL_DIR}"
    local path_check="export PATH=\"${path_entry}:\$PATH\""

    if [ -f "$shell_rc" ] && grep -q "$path_entry" "$shell_rc"; then
        log_info "PATH already configured in ${shell_rc}"
        return 0
    fi

    # Add to shell config
    {
        echo ""
        echo "# DeCoupled-AI"
        echo "$path_check"
    } >> "$shell_rc"

    log_success "Added ${INSTALL_DIR} to PATH in ${shell_rc}"
    log_warn "Restart your shell or run: source ${shell_rc}"
}

# Create uninstaller
create_uninstaller() {
    log_step "Creating uninstaller..."

    cat > "${INSTALL_DIR}/decoupled-ai-uninstall" <<'UNINSTALL_EOF'
#!/bin/sh
# DeCoupled-AI Uninstaller

set -eu

BINARY_NAME="decoupled-ai-server"
INSTALL_DIR="${HOME}/.local/bin"
CONFIG_DIR="${HOME}/.config/decoupled-ai"
DATA_DIR="${HOME}/.local/share/decoupled-ai"
SYSTEMD_DIR="${HOME}/.config/systemd/user"
LAUNCHD_DIR="${HOME}/Library/LaunchAgents"

log_info() { printf "\033[0;34m%s\033[0m\n" "$*"; }
log_success() { printf "\033[0;32m%s\033[0m\n" "$*"; }
log_warn() { printf "\033[1;33m%s\033[0m\n" "$*"; }

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"

echo "Uninstalling DeCoupled-AI..."

# Stop and disable services
if [ "$OS" = "linux" ]; then
    systemctl --user stop decoupled-ai.service >/dev/null 2>&1 || true
    systemctl --user disable decoupled-ai.service >/dev/null 2>&1 || true
    rm -f "${SYSTEMD_DIR}/decoupled-ai.service"
    systemctl --user daemon-reload >/dev/null 2>&1 || true
elif [ "$OS" = "darwin" ]; then
    launchctl unload "${LAUNCHD_DIR}/ai.decoupled.server.plist" >/dev/null 2>&1 || true
    rm -f "${LAUNCHD_DIR}/ai.decoupled.server.plist"
fi

# Remove binary and uninstaller
rm -f "${INSTALL_DIR}/${BINARY_NAME}"
rm -f "${INSTALL_DIR}/decoupled-ai-uninstall"

# Remove config and data (with confirmation)
echo "Remove configuration and data directories?"
echo "  Config: ${CONFIG_DIR}"
echo "  Data:   ${DATA_DIR}"
printf "Type 'yes' to confirm: "
read -r confirm
if [ "$confirm" = "yes" ]; then
    rm -rf "$CONFIG_DIR" "$DATA_DIR"
    log_success "Configuration and data removed"
else
    log_warn "Keeping configuration and data"
fi

# Remove PATH entry from shell configs
for rc in "${HOME}/.bashrc" "${HOME}/.zshrc" "${HOME}/.config/fish/config.fish" "${HOME}/.profile"; do
    if [ -f "$rc" ]; then
        sed -i '/DeCoupled-AI/d; /\.local\/bin/d' "$rc" 2>/dev/null || true
    fi
done

log_success "DeCoupled-AI uninstalled"
log_warn "Restart your shell to update PATH"
UNINSTALL_EOF

    chmod +x "${INSTALL_DIR}/decoupled-ai-uninstall"
    log_success "Uninstaller created at ${INSTALL_DIR}/decoupled-ai-uninstall"
}

# Print post-install instructions
print_summary() {
    echo ""
    log_success "=========================================="
    log_success "  DeCoupled-AI installed successfully!"
    log_success "=========================================="
    echo ""
    log_info "Binary:     ${INSTALL_DIR}/${BINARY_NAME}"
    log_info "Config:     ${CONFIG_DIR}/config.toml"
    log_info "Data dir:   ${DATA_DIR}"
    echo ""

    if [ "$OS" = "linux" ]; then
        log_info "Server is already running (auto-started):"
        echo "  systemctl --user status decoupled-ai"
        echo "  journalctl --user -u decoupled-ai -f  # view logs"
        echo ""
        log_info "To enable auto-start on boot (headless):"
        echo "  loginctl enable-linger \$USER"
        echo ""
        log_info "To run as system service (sudo):"
        echo "  sudo cp ${SYSTEMD_DIR}/decoupled-ai.service /etc/systemd/system/decoupled-ai.service"
        echo "  sudo systemctl daemon-reload && sudo systemctl enable --now decoupled-ai"
    elif [ "$OS" = "macos" ]; then
        log_info "To start the server:"
        echo "  launchctl start ai.decoupled.server"
        echo "  tail -f ${DATA_DIR}/server.log  # view logs"
        echo ""
        log_info "Service is already loaded and will auto-start on login"
    fi

    echo ""
    log_info "Manual start (foreground):"
    echo "  ${INSTALL_DIR}/${BINARY_NAME} --config ${CONFIG_DIR}/config.toml"
    echo ""
    log_info "Web UI: http://localhost:8080"
    log_info "API:    http://localhost:8080/v1"
    echo ""
    log_warn "Restart your shell or run: source ~/.bashrc (or ~/.zshrc)"
    echo ""
    log_info "Uninstall anytime with: ${INSTALL_DIR}/decoupled-ai-uninstall"
}

# Main installation flow
main() {
    echo ""
    log_step "DeCoupled-AI Universal Installer"
    log_info "Repository: ${REPO}"
    echo ""

    detect_platform
    get_latest_version
    download_asset "$VERSION"
    install_binary
    create_config
    setup_systemd
    setup_launchd
    setup_path
    create_uninstaller
    print_summary
}

# Allow passing version as argument
if [ $# -gt 0 ]; then
    VERSION="$1"
    log_info "Using specified version: ${VERSION}"
    detect_platform
    download_asset "$VERSION"
    install_binary
    create_config
    setup_systemd
    setup_launchd
    setup_path
    create_uninstaller
    print_summary
else
    main
fi