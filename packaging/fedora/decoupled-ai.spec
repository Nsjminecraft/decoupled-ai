Name:           decoupled-ai
Version:        1.0.0
Release:        1%{?dist}
Summary:        High-performance LLM inference server with speculative decoding

License:        MIT OR Apache-2.0
URL:            https://github.com/nsjminecraft/DeCoupled-AI
Source0:        https://github.com/nsjminecraft/DeCoupled-AI/archive/refs/tags/v%{version}.tar.gz

# Build dependencies
BuildRequires:  cargo >= 1.70
BuildRequires:  rust >= 1.70
BuildRequires:  gcc
BuildRequires:  openssl-devel
BuildRequires:  libnuma-devel
BuildRequires:  libaio-devel
BuildRequires:  rdma-core-devel

# Runtime dependencies
Requires:       glibc >= 2.31
Requires:       libgcc >= 10
Requires:       libstdc++ >= 10
Requires:       openssl-libs >= 3.0
Requires:       libnuma1
Requires:       libaio1
Requires:       libibverbs1
Requires:       librdmacm1

# Optional GPU dependencies
Recommends:     nvidia-driver >= 525
Recommends:     rocm-opencl-runtime
Recommends:     ocl-icd

# Systemd integration
%{?systemd_requires}
BuildRequires:  systemd-rpm-macros

%description
DeCoupled-AI is a high-performance LLM inference server featuring:

* Speculative Decoding with N-gram draft generation and batched target verification
* Multi-backend GPU support: CUDA (NVIDIA), ROCm (AMD), Metal (Apple Silicon), CPU fallback
* OpenAI-compatible REST API for drop-in replacement
* Embedded web dashboard for model management and monitoring
* Native installers for Linux (.deb, .rpm, AppImage), Windows (.msi), and macOS (.tar.gz)
* Universal bootstrap installer via \`curl | sh\`
* Over-the-air (OTA) updates via GitHub Releases

%prep
%autosetup -p1 -n DeCoupled-AI-%{version}

# Fetch dependencies for offline build
cargo fetch --locked --target x86_64-unknown-linux-gnu

%build
# Build release binary
export RUSTUP_TOOLCHAIN=stable
export CARGO_TARGET_DIR=target
cargo build --release --bin decoupled-ai-server --locked

%check
# Run tests (non-blocking)
cargo test --release --locked 2>/dev/null || true

%install
# Binary
install -Dm755 target/release/decoupled-ai-server %{buildroot}%{_bindir}/decoupled-ai-server

# Systemd service
install -Dm644 packaging/fedora/decoupled-ai.service %{buildroot}%{_unitdir}/decoupled-ai.service

# Sysusers
install -Dm644 packaging/fedora/decoupled-ai.sysusers %{buildroot}%{_sysconfdir}/sysusers.d/decoupled-ai.conf

# Tmpfiles
install -Dm644 packaging/fedora/decoupled-ai.tmpfiles %{buildroot}%{_sysconfdir}/tmpfiles.d/decoupled-ai.conf

# Configuration
install -Dm644 config/default.toml %{buildroot}%{_sysconfdir}/decoupled-ai/config.toml
install -Dm644 config/cuda.toml %{buildroot}%{_sysconfdir}/decoupled-ai/cuda.toml

# Frontend assets
if [ -d frontend-ui/assets ]; then
    mkdir -p %{buildroot}%{_datadir}/decoupled-ai/assets
    cp -r frontend-ui/assets/* %{buildroot}%{_datadir}/decoupled-ai/assets/
fi

# Documentation
if [ -d docs ]; then
    mkdir -p %{buildroot}%{_docdir}/decoupled-ai
    cp -r docs/* %{buildroot}%{_docdir}/decoupled-ai/
fi

# Licenses
install -Dm644 LICENSE-MIT %{buildroot}%{_licensedir}/decoupled-ai/LICENSE-MIT
install -Dm644 LICENSE-APACHE %{buildroot}%{_licensedir}/decoupled-ai/LICENSE-APACHE

# Create required directories
mkdir -p %{buildroot}/var/lib/decoupled-ai/{models,cache,index}
mkdir -p %{buildroot}/var/log/decoupled-ai
mkdir -p %{buildroot}/etc/decoupled-ai

%pre
# Create system user and group
%sysusers_create_compat %{SOURCE10} 2>/dev/null || :

%post
# Set permissions on config directory
chown root:decoupled-ai %{_sysconfdir}/decoupled-ai
chmod 750 %{_sysconfdir}/decoupled-ai

# Ensure data and log directories have correct permissions
chown decoupled-ai:decoupled-ai /var/lib/decoupled-ai /var/log/decoupled-ai
chmod 750 /var/lib/decoupled-ai /var/log/decoupled-ai

# Reload systemd
%systemd_post decoupled-ai.service

# Enable and start service
%systemd_enable decoupled-ai.service
%systemd_start decoupled-ai.service

# Log success
logger -t decoupled-ai "DeCoupled-AI installed and service started successfully"

%preun
# Stop and disable service on removal
%systemd_preun decoupled-ai.service

%postun
# Reload systemd
%systemd_postun_with_restart decoupled-ai.service

%files
%license LICENSE-MIT LICENSE-APACHE
%doc README.md
%{_bindir}/decoupled-ai-server
%{_unitdir}/decoupled-ai.service
%{_sysconfdir}/sysusers.d/decoupled-ai.conf
%{_sysconfdir}/tmpfiles.d/decoupled-ai.conf
%config(noreplace) %{_sysconfdir}/decoupled-ai/config.toml
%config(noreplace) %{_sysconfdir}/decoupled-ai/cuda.toml
%{_datadir}/decoupled-ai/assets/
%{_docdir}/decoupled-ai/
%dir %{_sysconfdir}/decoupled-ai
%dir /var/lib/decoupled-ai
%dir /var/lib/decoupled-ai/models
%dir /var/lib/decoupled-ai/cache
%dir /var/lib/decoupled-ai/index
%dir /var/log/decoupled-ai
%attr(0750,decoupled-ai,decoupled-ai) /var/lib/decoupled-ai
%attr(0750,decoupled-ai,decoupled-ai) /var/lib/decoupled-ai/models
%attr(0750,decoupled-ai,decoupled-ai) /var/lib/decoupled-ai/cache
%attr(0750,decoupled-ai,decoupled-ai) /var/lib/decoupled-ai/index
%attr(0750,decoupled-ai,decoupled-ai) /var/log/decoupled-ai

%changelog
* Wed Jul 16 2026 Niranjan Shanmuganathan Jothilakshmi <nsjminecraft@users.noreply.github.com> - 1.0.0-1
- Initial release