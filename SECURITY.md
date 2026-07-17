# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in DeCoupled-AI, please report it responsibly.

### Reporting Process

**Do not open a public issue for security vulnerabilities.**

Instead, please report vulnerabilities privately via email to:

**security@decoupled-ai.org**

Please include the following information:

1. **Description** - A clear description of the vulnerability
2. **Steps to Reproduce** - Step-by-step instructions to reproduce the issue
3. **Impact** - What an attacker could achieve
4. **Proof of Concept** - If applicable, a minimal proof of concept
5. **Affected Versions** - Which versions are affected
5. **Suggested Fix** - If you have a suggested fix, please include it

### Response Timeline

- **Acknowledgment**: Within 48 hours of receipt
- **Initial Assessment**: Within 7 days
- **Fix Development**: Within 30 days for critical issues, 90 days for others
- **Public Disclosure**: After a fix is released, typically 30 days after fix release

We will keep you informed of our progress throughout the process.

## Security Considerations for DeCoupled-AI

### Model Loading Security

- **Model Validation**: All models are validated before loading (format validation, size limits, integrity checks)
- **Weight Validation**: Model weights are validated for NaN/Inf values and reasonable ranges
- **Sandboxed Loading**: Model loading happens in isolated processes with restricted permissions
- **Size Limits**: Configurable maximum model size to prevent resource exhaustion

### API Security

- **Rate Limiting**: Built-in rate limiting on all API endpoints
- **Input Validation**: Strict validation on all API inputs (token limits, message sizes, parameter bounds)
- **Authentication**: Optional API key authentication for production deployments
- **CORS**: Configurable CORS policies
- **Request Size Limits**: Configurable request body size limits

### GPU/Compute Security

- **GPU Memory Isolation**: Each model instance gets isolated GPU memory allocation
- **Compute Isolation**: Separate compute streams for concurrent requests
- **Resource Limits**: Configurable GPU memory and compute limits per model
- **Driver Isolation**: Uses vendor-recommended isolation mechanisms (CUDA MPS, ROCm MIG, etc.)

### OTA Update Security

- **Signed Releases**: All releases are signed with GPG keys
- **Checksum Verification**: SHA256 checksums verified before installation
- **Signature Verification**: GPG signature verification before installation
- **HTTPS Only**: All downloads over HTTPS with certificate validation
- **Rollback Support**: Automatic rollback on failed updates

### Network Security

- **TLS Support**: Full TLS 1.3 support for API endpoints
- **Certificate Pinning**: Optional certificate pinning for update checks
- **Private Network Binding**: Default binds to localhost only
- **No Telemetry**: No telemetry, analytics, or phone-home by default

### System Integration Security

- **Systemd Hardening**: Systemd service includes:
  - `NoNewPrivileges=yes`
  - `PrivateTmp=yes`
  - `ProtectSystem=strict`
  - `ProtectHome=yes`
  - `ProtectKernelTunables=yes`
  - `ProtectKernelModules=yes`
  - `ProtectControlGroups=yes`
  - `RestrictRealtime=yes`
  - `RestrictSUIDSGID=yes`
  - `RemoveIPC=yes`
  - `PrivateDevices=yes`
  - `RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX`
  - `LockPersonality=yes`
  - `MemoryDenyWriteExecute=yes`
  - `SystemCallFilter=@system-service`
  - `SystemCallErrorNumber=EPERM`

- **User Isolation**: Runs as dedicated unprivileged system user
- **Capability Dropping**: Drops all capabilities on startup
- **Filesystem Isolation**: Private /tmp, read-only system directories

### Dependency Security

- **Dependency Scanning**: Automated `cargo audit` and `cargo deny` in CI
- **Minimal Dependencies**: Minimal dependency footprint
- **Pinned Dependencies**: All dependencies pinned in Cargo.lock
- **Regular Updates**: Dependencies updated regularly for security patches
- **Supply Chain**: Cargo.lock committed, verified in CI

### Deployment Recommendations

For production deployments:

1. **Enable TLS** - Configure TLS certificates for all API endpoints
2. **Enable Authentication** - Set `API_KEY` environment variable
3. **Configure Firewall** - Restrict API port to trusted networks only
4. **Enable Audit Logging** - Set `RUST_LOG=info` or higher for audit trails
5. **Monitor Resources** - Monitor GPU memory, CPU, and disk usage
6. **Regular Updates** - Enable automatic updates or check for updates regularly
7. **Backup Models** - Regular backups of model files and configurations

### Vulnerability Disclosure

We follow coordinated vulnerability disclosure. We ask that you give us reasonable time to address issues before public disclosure. We will credit researchers who report vulnerabilities responsibly (unless they prefer anonymity).

### Security Hall of Fame

Researchers who have responsibly disclosed vulnerabilities:

- (None yet - be the first!)

## Contact

- **Security Email**: security@decoupled-ai.org
- **PGP Key**: Available at https://decoupled-ai.org/security/pgp-key.asc
- **Response Time**: We aim to respond within 48 hours

---

Thank you for helping keep DeCoupled-AI secure!