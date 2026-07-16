#!/bin/bash
# AppImage Build Script for DeCoupled-AI
# Usage: ./build-appimage.sh [version] [arch]
# Example: ./build-appimage.sh 1.0.0 x86_64

set -euo pipefail

VERSION="${1:-1.0.0}"
ARCH="${2:-x86_64}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/target/appimage-build"
APPDIR="${BUILD_DIR}/DeCoupled-AI.AppDir"
APPIMAGE_TOOL="${REPO_ROOT}/tools/appimagetool"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# Check for appimagetool
check_appimagetool() {
    if [ ! -f "$APPIMAGE_TOOL" ]; then
        log_info "Downloading appimagetool..."
        mkdir -p "$(dirname "$APPIMAGE_TOOL")"
        curl -sSfL "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-${ARCH}.AppImage" -o "$APPIMAGE_TOOL"
        chmod +x "$APPIMAGE_TOOL"
    fi
}

# Build the binary first
build_binary() {
    log_info "Building release binary..."
    cd "$REPO_ROOT"
    cargo build --release --bin decoupled-ai-server
}

# Create AppDir structure
create_appdir() {
    log_info "Creating AppDir structure..."
    rm -rf "$APPDIR"
    mkdir -p "$APPDIR"/usr/bin
    mkdir -p "$APPDIR"/usr/share/applications
    mkdir -p "$APPDIR"/usr/share/icons/hicolor/256x256/apps
    mkdir -p "$APPDIR"/usr/share/decoupled-ai/assets
    mkdir -p "$APPDIR"/usr/share/doc/decoupled-ai
    mkdir -p "$APPDIR"/etc/decoupled-ai
    mkdir -p "$APPDIR"/var/lib/decoupled-ai/models
    mkdir -p "$APPDIR"/var/lib/decoupled-ai/cache
    mkdir -p "$APPDIR"/var/log/decoupled-ai
}

# Copy files to AppDir
populate_appdir() {
    log_info "Populating AppDir..."

    # Binary
    cp "$REPO_ROOT/target/release/decoupled-ai-server" "$APPDIR/usr/bin/"
    chmod +x "$APPDIR/usr/bin/decoupled-ai-server"

    # Desktop entry
    cat > "$APPDIR/usr/share/applications/decoupled-ai.desktop" << 'EOF'
[Desktop Entry]
Type=Application
Name=DeCoupled-AI
GenericName=LLM Inference Server
Comment=High-performance LLM inference server with speculative decoding
Exec=decoupled-ai-server
Icon=decoupled-ai
Terminal=false
Categories=Development;Science;ArtificialIntelligence;
StartupNotify=true
Keywords=AI;LLM;inference;server;machine-learning;
EOF

    # Icon (create a simple SVG if not available)
    if [ -f "$REPO_ROOT/assets/icon.svg" ]; then
        cp "$REPO_ROOT/assets/icon.svg" "$APPDIR/usr/share/icons/hicolor/256x256/apps/decoupled-ai.svg"
    else
        cat > "$APPDIR/usr/share/icons/hicolor/256x256/apps/decoupled-ai.svg" << 'EOF'
<svg width="256" height="256" viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="256" height="256" rx="32" fill="url(#grad)"/>
  <text x="128" y="165" font-family="Arial, sans-serif" font-size="100" font-weight="bold" fill="white" text-anchor="middle">AI</text>
  <circle cx="190" cy="66" r="20" fill="#10b981" opacity="0.9"/>
  <text x="190" y="74" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="white" text-anchor="middle">→</text>
</svg>
EOF
    fi

    # AppRun script
    cat > "$APPDIR/AppRun" << 'EOF'
#!/bin/bash
# AppRun entry point for DeCoupled-AI AppImage

HERE="$(dirname "$(readlink -f "${0}")")"

# Set up environment
export DECOUPLED_AI_DATA_DIR="${HOME}/.local/share/decoupled-ai"
export DECOUPLED_AI_CONFIG_DIR="${HOME}/.config/decoupled-ai"
export DECOUPLED_AI_LOG_DIR="${HOME}/.local/share/decoupled-ai/logs"

# Create directories if they don't exist
mkdir -p "${DECOUPLED_AI_DATA_DIR}/models"
mkdir -p "${DECOUPLED_AI_DATA_DIR}/cache"
mkdir -p "${DECOUPLED_AI_CONFIG_DIR}"
mkdir -p "${DECOUPLED_AI_LOG_DIR}"

# Use bundled config if user config doesn't exist
if [ ! -f "${DECOUPLED_AI_CONFIG_DIR}/config.toml" ]; then
    cp "${HERE}/etc/decoupled-ai/config.toml" "${DECOUPLED_AI_CONFIG_DIR}/config.toml" 2>/dev/null || true
fi

# Execute the binary with all arguments
exec "${HERE}/usr/bin/decoupled-ai-server" --config "${DECOUPLED_AI_CONFIG_DIR}/config.toml" "$@"
EOF
    chmod +x "$APPDIR/AppRun"

    # Configuration files
    cp "$REPO_ROOT/config/default.toml" "$APPDIR/etc/decoupled-ai/config.toml"
    cp "$REPO_ROOT/config/cuda.toml" "$APPDIR/etc/decoupled-ai/cuda.toml"

    # Assets
    if [ -d "$REPO_ROOT/frontend-ui/assets" ]; then
        cp -r "$REPO_ROOT/frontend-ui/assets"/* "$APPDIR/usr/share/decoupled-ai/assets/" 2>/dev/null || true
    fi

    # Documentation
    if [ -d "$REPO_ROOT/docs" ]; then
        cp -r "$REPO_ROOT/docs"/* "$APPDIR/usr/share/doc/decoupled-ai/" 2>/dev/null || true
    fi

    # Copy README and LICENSE
    cp "$REPO_ROOT/README.md" "$APPDIR/usr/share/doc/decoupled-ai/" 2>/dev/null || true
    cp "$REPO_ROOT/LICENSE-MIT" "$APPDIR/usr/share/doc/decoupled-ai/" 2>/dev/null || true
    cp "$REPO_ROOT/LICENSE-APACHE" "$APPDIR/usr/share/doc/decoupled-ai/" 2>/dev/null || true
}

# Build AppImage
build_appimage() {
    log_info "Building AppImage..."
    cd "$BUILD_DIR"

    ARCH="$ARCH" "$APPIMAGE_TOOL" "$APPDIR" "DeCoupled-AI-${VERSION}-${ARCH}.AppImage"

    log_success "AppImage created: DeCoupled-AI-${VERSION}-${ARCH}.AppImage"
    ls -lh "DeCoupled-AI-${VERSION}-${ARCH}.AppImage"
}

# Main
main() {
    log_info "Building DeCoupled-AI AppImage v${VERSION} for ${ARCH}"

    check_appimagetool
    build_binary
    create_appdir
    populate_appdir
    build_appimage

    log_success "AppImage build complete!"
    echo "Output: ${BUILD_DIR}/DeCoupled-AI-${VERSION}-${ARCH}.AppImage"
}

main "$@"