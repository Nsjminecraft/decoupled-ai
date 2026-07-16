Name:           decoupled-ai
Version:        1.0.0
Release:        1%{?dist}
Summary:        High-performance LLM inference server with speculative decoding
License:        MIT OR Apache-2.0
URL:            https://github.com/nsjminecraft/DeCoupled-AI
Source0:        %{name}-%{version}-%{_arch}.tar.gz

BuildRequires:  systemd-rpm-macros
Requires:       systemd
Requires(pre):  shadow-utils
Requires(post): systemd
Requires(preun): systemd
Requires(postun): systemd

%description
DeCoupled-AI is a high-performance LLM inference server featuring:
- Speculative decoding with N-gram draft generation and batched target verification
- Multi-backend support: CUDA, ROCm, Metal, CPU (WebGPU coming soon)
- OpenAI-compatible API for drop-in replacement
- Embedded web dashboard for model management and monitoring
- Native installers for Linux (.deb, .rpm), Windows (.msi), and macOS (.tar.gz)
- Universal bootstrap installer via \`curl | sh\`

%prep
%autosetup -n %{name}-%{version}-%{_arch}

%build
# No build step - binary is pre-built

%install
mkdir -p %{buildroot}/usr/bin
mkdir -p %{buildroot}/usr/lib/systemd/system
mkdir -p %{buildroot}/etc/decoupled-ai
mkdir -p %{buildroot}/var/lib/decoupled-ai/models
mkdir -p %{buildroot}/var/lib/decoupled-ai/cache
mkdir -p %{buildroot}/var/log/decoupled-ai
mkdir -p %{buildroot}/usr/share/decoupled-ai/assets
mkdir -p %{buildroot}/usr/share/doc/decoupled-ai

# Install binary
install -m 755 decoupled-ai-server %{buildroot}/usr/bin/decoupled-ai-server

# Install systemd service
install -m 644 decoupled-ai.service %{buildroot}/usr/lib/systemd/system/decoupled-ai.service

# Install default configuration
install -m 644 config.toml %{buildroot}/etc/decoupled-ai/config.toml

# Install CUDA config (optional)
install -m 644 cuda.toml %{buildroot}/etc/decoupled-ai/cuda.toml

# Install frontend assets
cp -r assets/* %{buildroot}/usr/share/decoupled-ai/assets/ 2>/dev/null || true

# Install documentation
cp -r docs/* %{buildroot}/usr/share/doc/decoupled-ai/ 2>/dev/null || true

%pre
# Create system user and group
getent group decoupled-ai >/dev/null || groupadd -r decoupled-ai
getent passwd decoupled-ai >/dev/null || \
    useradd -r -g decoupled-ai -d /var/lib/decoupled-ai -s /sbin/nologin \
    -c "DeCoupled-AI Inference Server" decoupled-ai

# Create directories with correct ownership
mkdir -p /etc/decoupled-ai /var/lib/decoupled-ai /var/log/decoupled-ai
chown decoupled-ai:decoupled-ai /var/lib/decoupled-ai /var/log/decoupled-ai
chmod 750 /var/lib/decoupled-ai /var/log/decoupled-ai
chmod 755 /etc/decoupled-ai

%post
# Set permissions on config directory
chown root:decoupled-ai /etc/decoupled-ai
chmod 750 /etc/decoupled-ai

# Ensure data and log directories have correct permissions
chown decoupled-ai:decoupled-ai /var/lib/decoupled-ai /var/log/decoupled-ai
chmod 750 /var/lib/decoupled-ai /var/log/decoupled-ai

# Install default configuration if not present
if [ ! -f /etc/decoupled-ai/config.toml ]; then
    cp /usr/share/decoupled-ai/config.toml /etc/decoupled-ai/config.toml 2>/dev/null || true
    chown root:decoupled-ai /etc/decoupled-ai/config.toml
    chmod 640 /etc/decoupled-ai/config.toml
fi

# Reload systemd daemon
systemctl daemon-reload >/dev/null 2>&1 || true

# Enable and start service
systemctl enable decoupled-ai.service >/dev/null 2>&1 || true
systemctl start decoupled-ai.service >/dev/null 2>&1 || true

%preun
if [ $1 -eq 0 ]; then
    # Package removal, not upgrade
    systemctl stop decoupled-ai.service >/dev/null 2>&1 || true
    systemctl disable decoupled-ai.service >/dev/null 2>&1 || true
fi

%postun
systemctl daemon-reload >/dev/null 2>&1 || true

%files
%license LICENSE-MIT LICENSE-APACHE
%doc README.md
%config(noreplace) /etc/decoupled-ai/config.toml
%config(noreplace) /etc/decoupled-ai/cuda.toml
/usr/bin/decoupled-ai-server
/usr/lib/systemd/system/decoupled-ai.service
/usr/share/decoupled-ai/assets/
/usr/share/doc/decoupled-ai/
/var/lib/decoupled-ai/
/var/log/decoupled-ai/

%changelog
* Wed Jul 16 2025 Niranjan Shanmuganathan <nsjminecraft@users.noreply.github.com> - 1.0.0-1
- Initial RPM package for DeCoupled-AI