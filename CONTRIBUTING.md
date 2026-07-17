# Contributing to DeCoupled-AI

Thank you for your interest in contributing to DeCoupled-AI! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to conduct@decoupled-ai.org.

---

## Getting Started

### Prerequisites

- **Rust 1.75+** (install via [rustup](https://rustup.rs))
- **Git** for version control
- **Platform-specific dependencies**:
  - **Linux**: `build-essential`, `pkg-config`, `libssl-dev`, `clang`, `libnuma-dev`
  - **Windows**: Visual Studio 2022 + Windows SDK, or MSVC Build Tools
  - **macOS**: Xcode Command Line Tools

### Quick Start

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR-USERNAME/DeCoupled-AI.git
cd DeCoupled-AI

# Add upstream remote
git remote add upstream https://github.com/nsjminecraft/DeCoupled-AI.git

# Install development tools
cargo install cargo-husky cargo-nextest cargo-audit cargo-deny

# Install pre-commit hooks
cargo husky install

# Build and test
cargo build --workspace
cargo test --workspace
```

---

## Development Setup

### IDE Configuration

**VS Code** (recommended):
- Install `rust-analyzer` extension
- Install `Even Better TOML` for Cargo.toml editing
- Install `crates` for dependency management
- Settings (`.vscode/settings.json`):
```json
{
  "rust-analyzer.checkOnSave.command": "clippy",
  "rust-analyzer.cargo.allFeatures": true,
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "rust-lang.rust-analyzer"
}
```

**CLI Development**:
```bash
# Watch for changes and rebuild
cargo watch -x "build --workspace" -x "test --workspace"

# Quick check (no build)
cargo check --workspace
```

### Environment Variables

```bash
# Optional: Set for development
export RUST_LOG=debug
export RUST_BACKTRACE=1
export DECOUPLED_AI_CONFIG=./config/default.toml
```

---

## Contribution Workflow

### 1. Find or Create an Issue

- Browse [existing issues](https://github.com/nsjminecraft/DeCoupled-AI/issues)
- For new features/bugs, create an issue first for discussion
- Use issue templates when available

### 2. Create a Feature Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# or for bug fixes
git checkout -b fix/issue-number-description
```

### 3. Make Changes

- Write clean, well-documented code
- Add tests for new functionality
- Update documentation as needed
- Follow the code style guidelines below

### 4. Run Quality Checks

```bash
# Format code
cargo fmt --all --check

# Lint
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Run tests
cargo test --workspace

# Security audit
cargo audit

# Dependency check
cargo deny check
```

### 5. Commit Changes

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: <type>(<scope>): <description>
# Types: feat, fix, docs, style, refactor, perf, test, chore, build, ci

git add .
git commit -m "feat(gpu): add AMD ROCm backend support"
git commit -m "fix(updater): handle network timeout gracefully"
git commit -m "docs(readme): update installation instructions"
```

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
# Create PR via GitHub UI
```

---

## Code Style

### Rust Style Guide

- **Formatting**: `rustfmt` (enforced via CI)
- **Linting**: `clippy` with `-D warnings` (enforced via CI)
- **Edition**: 2021
- **Naming**: Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

### Key Conventions

```rust
// Use descriptive names
let gpu_detection_result = detect_gpus()?;  // Good
let g = detect_gpus()?;                     // Bad

// Error handling with context
use anyhow::{Context, Result};
fn load_model(path: &Path) -> Result<Model> {
    let data = fs::read(path).context("Failed to read model file")?;
    Model::from_bytes(&data).context("Failed to parse model")
}

// Async functions
async fn start_server(config: Config) -> Result<()> {
    // ...
}

// Documentation
/// Starts the inference server with the given configuration.
///
/// # Arguments
/// * `config` - Server configuration
///
/// # Returns
/// Result indicating success or failure
pub async fn run(config: Config) -> Result<()> { ... }
```

### Module Organization

```
src/
├── lib.rs          // Public API, re-exports
├── main.rs         // Binary entry point
├── config.rs       // Configuration types
├── server.rs       // HTTP server
├── engine.rs       // Inference engine
├── gpu_detect.rs   // GPU detection
├── updater.rs      // OTA updates
└── ...
```

---

## Testing

### Test Categories

1. **Unit Tests** - In `#[cfg(test)]` modules alongside code
2. **Integration Tests** - In `tests/` directory
3. **Benchmarks** - In `benches/` directory

### Running Tests

```bash
# All tests
cargo test --workspace

# Specific crate
cargo test -p server-backend

# With output
cargo test -- --nocapture

# Nextest (faster)
cargo nextest run --workspace

# Single test
cargo test test_gpu_detection -- --nocapture
```

### Test Guidelines

- Test public API, not internals
- Use descriptive test names: `test_gpu_selection_prefers_cuda_over_rocm`
- Mock external dependencies (network, filesystem)
- Test error paths, not just happy paths
- Property-based testing with `proptest` for complex logic

---

## Documentation

### Code Documentation

- All public APIs must have `///` doc comments
- Include examples for non-trivial functions
- Document error conditions and panics

### User Documentation

- Update `README.md` for user-facing changes
- Update `docs/` for detailed guides
- Add CLI help text for new flags

### Architecture Documentation

- Document major design decisions in `docs/architecture/`
- Update API docs when changing endpoints

---

## Pull Request Process

### PR Requirements

Before submitting a PR, ensure:

- [ ] All CI checks pass (fmt, clippy, test, audit, deny)
- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] No breaking changes without discussion
- [ ] Commits follow conventional format

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code formatted with `cargo fmt`
- [ ] Linted with `cargo clippy`
- [ ] Security audit passes
- [ ] Dependencies reviewed
- [ ] Documentation updated
```

### Review Process

1. **Automated checks** run on every push
2. **Maintainer review** for code quality and architecture
3. **Approval** required from at least one maintainer
4. **Merge** via squash-and-merge to maintain clean history

---

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. Update version in `Cargo.toml` files
2. Update `CHANGELOG.md`
3. Create release branch: `release/v1.2.0`
4. CI builds and tests all platforms
5. Tag release: `git tag v1.2.0`
6. GitHub Actions creates release artifacts
7. Publish to crates.io (for library crates)
8. Update package repositories (AUR, Homebrew, etc.)

### Pre-release Checklist

- [ ] All tests pass on all supported platforms
- [ ] Benchmarks show no regressions
- [ ] Security audit clean
- [ ] Documentation complete
- [ ] Migration guide for breaking changes

---

## Getting Help

- **Discussions**: [GitHub Discussions](https://github.com/nsjminecraft/DeCoupled-AI/discussions)
- **Issues**: [GitHub Issues](https://github.com/nsjminecraft/DeCoupled-AI/issues)
- **Discord**: [DeCoupled-AI Community](https://discord.gg/decoupled-ai) (if available)
- **Email**: dev@decoupled-ai.org

---

## Recognition

Contributors are recognized in:
- [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Release notes
- GitHub contributor graphs

Thank you for contributing to DeCoupled-AI! 🚀