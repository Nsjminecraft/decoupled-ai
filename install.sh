#!/bin/sh
# DeCoupled-AI Universal Installer
# Usage: curl -sSfL https://github.com/nsjminecraft/DeCoupled-AI/releases/latest/download/install.sh | sh
#        curl -sSfL https://github.com/nsjminecraft/DeCoupled-AI/releases/download/v1.0.0/install.sh | sh

set -eu

# Configuration
REPO_OWNER="nsjminecraft"
REPO_NAME="DeCoupled-AI"
BINARY_NAME="decoupled-ai-server"
INSTALL_DIR="${HOME}/.local/bin"
CONFIG_DIR="${HOME}/.config/decoupled-ai"
DATA_DIR="${HOME}/.local/share/decoupled-ai"

# Colors for output (if terminal supports it)
if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
    RED=$(tput setaf 1)
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    BLUE=$(tput setaf 4)
    BOLD=$(tput bold)
    RESET=$(tput sgr0)
else
    RED=""
    GREEN=""
    YELLOW=""
    BLUE=""
    BOLD=""
    RESET=""
fi

# Logging functions
log_info() { printf "${BLUE}[INFO]${RESET} %s\n" "$*"; }
log_success() { printf "${GREEN}[SUCCESS]${RESET} %s\n" "$*"; }
log_warn() { printf "${YELLOW}[WARN]${RESET} %s\n" "$*"; }
log_error() { printf "${RED}[ERROR]${RESET} %s\n" "$*" >&2; }
log_step() { printf "${BOLD}${BLUE}==>${RESET} ${BOLD}%s${RESET}\n" "$*"; }

# Error handler
die() {
    log_error "$*"
    exit 1
}

# Check for required commands
check_dependencies() {
    log_step "Checking dependencies..."
    for cmd in curl tar grep mkdir chmod; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            die "Required command '$cmd' not found. Please install it first."
        fi
    done
    log_success "All dependencies satisfied"
}

# Detect OS and architecture
detect_platform() {
    log_step "Detecting platform..."

    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)

    case "$OS" in
        linux)
            PLATFORM_OS="linux"
            ;;
        darwin)
            PLATFORM_OS="apple-darwin"
            ;;
        mingw*|msys*|cygwin*)
            PLATFORM_OS="pc-windows-msvc"
            ;;
        *)
            die "Unsupported operating system: $OS"
            ;;
    esac

    case "$ARCH" in
        x86_64|amd64)
            PLATFORM_ARCH="x86_64"
            ;;
        aarch64|arm64)
            PLATFORM_ARCH="aarch64"
            ;;
        *)
            die "Unsupported architecture: $ARCH"
            ;;
    esac

    # Determine target triple
    case "$PLATFORM_OS" in
        linux)
            # Prefer musl for static linking, fall back to gnu
            if [ -f /etc/alpine-release ] || ldd --version 2>&1 | grep -q musl; then
                TARGET="${PLATFORM_ARCH}-unknown-linux-musl"
                PACKAGE_EXT="tar.gz"
            else
                TARGET="${PLATFORM_ARCH}-unknown-linux-gnu"
                PACKAGE_EXT="deb"
            fi
            ;;
        apple-darwin)
            TARGET="${PLATFORM_ARCH}-apple-darwin"
            PACKAGE_EXT="tar.gz"
            ;;
        pc-windows-msvc)
            TARGET="${PLATFORM_ARCH}-pc-windows-msvc"
            PACKAGE_EXT="msi"
            ;;
    esac

    log_success "Detected: $PLATFORM_OS / $PLATFORM_ARCH -> Target: $TARGET ($PACKAGE_EXT)"
}

# Fetch latest release version from GitHub API
fetch_latest_version() {
    log_step "Fetching latest release information..."

    API_URL="https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/releases/latest"

    # Use curl with follow redirects and silent fail
    RELEASE_JSON=$(curl -sSfL -H "Accept: application/vnd.github.v3+json" "$API_URL") || \
        die "Failed to fetch release information from GitHub API"

    VERSION=$(echo "$RELEASE_JSON" | grep -o '"tag_name": *"[^"]*"' | head -1 | cut -d'"' -f4)

    if [ -z "$VERSION" ]; then
        die "Could not parse version from GitHub API response"
    fi

    log_success "Latest version: $VERSION"
}

# Construct download URL for the platform-specific asset
construct_download_url() {
    # Asset naming pattern: decoupled-ai-<version>-<target>.<ext>
    ASSET_NAME="decoupled-ai-${VERSION}-${TARGET}.${PACKAGE_EXT}"
    DOWNLOAD_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/download/${VERSION}/${ASSET_NAME}"

    log_info "Asset: $ASSET_NAME"
    log_info "URL: $DOWNLOAD_URL"
}

# Download and verify the asset
download_asset() {
    log_step "Downloading $ASSET_NAME..."

    TEMP_DIR=$(mktemp -d)
    trap 'rm -rf "$TEMP_DIR"' EXIT INT TERM

    ASSET_PATH="${TEMP_DIR}/${ASSET_NAME}"

    if ! curl -sSfL -o "$ASSET_PATH" "$DOWNLOAD_URL"; then
        # Try without version prefix in asset name (fallback)
        ASSET_NAME_FALLBACK="decoupled-ai-${TARGET}.${PACKAGE_EXT}"
        DOWNLOAD_URL_FALLBACK="https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/download/${VERSION}/${ASSET_NAME_FALLBACK}"
        log_warn "Primary asset not found, trying fallback: $ASSET_NAME_FALLBACK"
        if ! curl -sSfL -o "$ASSET_PATH" "$DOWNLOAD_URL_FALLBACK"; then
            die "Failed to download asset. Please check if $TARGET is supported in release $VERSION"
        fi
        ASSET_NAME=$ASSET_NAME_FALLBACK
    fi

    # Verify download
    if [ ! -s "$ASSET_PATH" ]; then
        die "Downloaded file is empty"
    fi

    log_success "Downloaded $(du -h "$ASSET_PATH" | cut -f1)"
}

# Extract and install based on package type
install_package() {
    log_step "Installing package..."

    mkdir -p "$INSTALL_DIR" "$CONFIG_DIR" "$DATA_DIR/models" "$DATA_DIR/cache"

    case "$PACKAGE_EXT" in
        tar.gz)
            extract_tarball
            ;;
        deb)
            install_deb
            ;;
        msi)
            install_msi
            ;;
        *)
            die "Unsupported package format: $PACKAGE_EXT"
            ;;
    esac
}

# Extract tar.gz and install binaries
extract_tarball() {
    log_info "Extracting tarball..."

    EXTRACT_DIR="${TEMP_DIR}/extract"
    mkdir -p "$EXTRACT_DIR"

    tar -xzf "$ASSET_PATH" -C "$EXTRACT_DIR"

    # Find the binary (could be in root or subdirectory)
    BINARY_PATH=$(find "$EXTRACT_DIR" -type f -name "$BINARY_NAME*" | head -1)

    if [ -z "$BINARY_PATH" ]; then
        die "Binary not found in archive"
    fi

    log_info "Found binary at: $BINARY_PATH"

    # Copy binary to install directory
    cp "$BINARY_PATH" "$INSTALL_DIR/$BINARY_NAME"
    chmod +x "$INSTALL_DIR/$BINARY_NAME"

    # Copy assets if present
    if [ -d "$EXTRACT_DIR/assets" ]; then
        cp -r "$EXTRACT_DIR/assets" "$DATA_DIR/"
        log_info "Copied frontend assets"
    fi

    # Copy docs if present
    if [ -d "$EXTRACT_DIR/docs" ]; then
        cp -r "$EXTRACT_DIR/docs" "$DATA_DIR/"
        log_info "Copied documentation"
    fi

    # Copy config if present
    if [ -d "$EXTRACT_DIR/config" ]; then
        cp -r "$EXTRACT_DIR/config"/* "$CONFIG_DIR/" 2>/dev/null || true
        log_info "Copied default configuration"
    fi

    log_success "Binary installed to $INSTALL_DIR/$BINARY_NAME"
}

# Install Debian package (requires sudo)
install_deb() {
    log_info "Installing Debian package (requires sudo)..."

    if ! command -v dpkg >/dev/null 2>&1; then
        die "dpkg not found. This package requires a Debian-based system."
    fi

    sudo dpkg -i "$ASSET_PATH" || die "Failed to install .deb package"

    log_success "Debian package installed system-wide"
}

# Install Windows MSI (requires admin)
install_msi() {
    log_info "Installing Windows MSI (requires Administrator)..."

    # On Windows via MSYS2/Git Bash, use msiexec
    if command -v msiexec.exe >/dev/null 2>&1; then
        msiexec.exe /i "$(cygpath -w "$ASSET_PATH")" /quiet /norestart || die "Failed to install .msi package"
    else
        die "msiexec not found. Please run the .msi file manually as Administrator."
    fi

    log_success "MSI package installed"
}

# Verify installation and PATH configuration
verify_installation() {
    log_step "Verifying installation..."

    # Check if binary is in PATH
    if command -v "$BINARY_NAME" >/dev/null 2>&1; then
        VERSION_OUTPUT=$("$BINARY_NAME" --version 2>/dev/null || echo "unknown")
        log_success "$BINARY_NAME is in PATH: $VERSION_OUTPUT"
    else
        log_warn "$BINARY_NAME is NOT in PATH"
        log_info "Install directory: $INSTALL_DIR"
        log_info "Add to your shell profile:"
        case "$SHELL" in
            */zsh)
                echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc"
                ;;
            */fish)
                echo "  fish_add_path \$HOME/.local/bin"
                ;;
            *)
                echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
                ;;
        esac
    fi

    # Verify config directory
    if [ -d "$CONFIG_DIR" ]; then
        log_success "Config directory: $CONFIG_DIR"
    fi

    # Verify data directory
    if [ -d "$DATA_DIR" ]; then
        log_success "Data directory: $DATA_DIR"
    fi
}

# Print post-install instructions
print_next_steps() {
    cat << EOF

${BOLD}${GREEN}Installation complete!${RESET}

${BOLD}Next steps:${RESET}

1. ${BOLD}Add to PATH${RESET} (if not already done):
   ${BLUE}export PATH="\$HOME/.local/bin:\$PATH"${RESET}

   Add to your shell profile (~/.bashrc, ~/.zshrc, or config.fish)

2. ${BOLD}Configure${RESET} (optional):
   Config file: ${BLUE}$CONFIG_DIR/config.toml${RESET}
   Edit to customize model paths, server settings, etc.

3. ${BOLD}Start the server${RESET}:
   ${BLUE}$BINARY_NAME${RESET}

   Or as daemon:
   ${BLUE}$BINARY_NAME --daemon${RESET}

4. ${BOLD}Open Dashboard${RESET}:
   Visit ${BLUE}http://localhost:8080${RESET} in your browser

5. ${BOLD}Manage models${RESET}:
   Models directory: ${BLUE}$DATA_DIR/models${RESET}
   Place your GGUF/Safetensors models there

${BOLD}Documentation:${RESET} $DATA_DIR/docs (if bundled)

${BOLD}Support:${RESET} https://github.com/${REPO_OWNER}/${REPO_NAME}/issues

EOF
}

# Main installation flow
main() {
    log_step "DeCoupled-AI Universal Installer"
    log_info "Repository: $REPO_OWNER/$REPO_NAME"

    check_dependencies
    detect_platform
    fetch_latest_version
    construct_download_url
    download_asset
    install_package
    verify_installation
    print_next_steps

    log_success "Installation completed successfully!"
}

# Run main if not sourced
if [ "${0##*/}" = "install.sh" ] || [ "${0##*/}" = "sh" ]; then
    main "$@"
fi