//! Documentation Generation Engine
//!
//! Generates comprehensive documentation from source code including:
//! - SPEC.md: Binary format specification
//! - API_REFERENCE.md: OpenAI-compatible API docs
//! - BUILD.md: Multi-platform build guide
//! - Code documentation audit

use anyhow::Result;
use clap::{Parser, Subcommand};
use handlebars::Handlebars;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

// ============================================================================
// Documentation Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectDoc {
    pub name: String,
    pub version: String,
    pub description: String,
    pub crates: Vec<CrateDoc>,
    pub modules: Vec<ModuleDoc>,
    pub api_endpoints: Vec<ApiEndpoint>,
    pub build_targets: Vec<BuildTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateDoc {
    pub name: String,
    pub path: String,
    pub description: String,
    pub version: String,
    pub dependencies: Vec<String>,
    pub features: Vec<String>,
    pub public_items: Vec<PublicItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicItem {
    pub name: String,
    pub kind: ItemKind,
    pub signature: String,
    pub doc: String,
    pub module: String,
    pub safety_notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ItemKind {
    Struct,
    Enum,
    Function,
    Method,
    Const,
    Static,
    Trait,
    TypeAlias,
    Macro,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDoc {
    pub path: String,
    pub name: String,
    pub doc: String,
    pub items: Vec<PublicItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiEndpoint {
    pub method: String,
    pub path: String,
    pub summary: String,
    pub description: String,
    pub request_schema: Option<serde_json::Value>,
    pub response_schema: Option<serde_json::Value>,
    pub examples: Vec<ApiExample>,
    pub auth_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiExample {
    pub name: String,
    pub description: String,
    pub request: serde_json::Value,
    pub response: serde_json::Value,
    pub curl: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildTarget {
    pub name: String,
    pub platform: String,
    pub arch: String,
    pub features: Vec<String>,
    pub dependencies: Vec<String>,
    pub build_steps: Vec<String>,
    pub artifacts: Vec<String>,
}

// ============================================================================
// Documentation Generator
// ============================================================================

pub struct DocsGenerator {
    project_root: PathBuf,
    output_dir: PathBuf,
    templates: Handlebars<'static>,
}

impl DocsGenerator {
    pub fn new(project_root: impl AsRef<Path>, output_dir: impl AsRef<Path>) -> Result<Self> {
        let project_root = project_root.as_ref().to_path_buf();
        let output_dir = output_dir.as_ref().to_path_buf();

        let mut templates = Handlebars::new();
        templates.set_strict_mode(true);

        // Register built-in templates
        templates.register_template_string("spec", include_str!("../templates/spec.hbs"))?;
        templates.register_template_string("api_reference", include_str!("../templates/api_reference.hbs"))?;
        templates.register_template_string("build", include_str!("../templates/build.hbs"))?;
        templates.register_template_string("readme", include_str!("../templates/readme.hbs"))?;
        templates.register_template_string("changelog", include_str!("../templates/changelog.hbs"))?;

        Ok(Self {
            project_root,
            output_dir,
            templates,
        })
    }

    pub fn generate_all(&self) -> Result<()> {
        fs::create_dir_all(&self.output_dir)?;

        let project = self.analyze_project()?;

        // Generate SPEC.md
        self.generate_spec(&project)?;

        // Generate API_REFERENCE.md
        self.generate_api_reference(&project)?;

        // Generate BUILD.md
        self.generate_build_guide(&project)?;

        // Generate README.md
        self.generate_readme(&project)?;

        // Generate CHANGELOG.md template
        self.generate_changelog(&project)?;

        // Audit code comments
        self.audit_code_comments(&project)?;

        println!("Documentation generated in {}", self.output_dir.display());
        Ok(())
    }

    fn analyze_project(&self) -> Result<ProjectDoc> {
        let mut crates = Vec::new();
        let mut modules = Vec::new();
        let mut api_endpoints = Vec::new();
        let mut build_targets = Vec::new();

        // Scan for Cargo.toml files
        for entry in WalkDir::new(&self.project_root) {
            let entry = entry?;
            if entry.file_name() == "Cargo.toml" {
                if let Ok(crate_doc) = self.parse_crate(entry.path()) {
                    crates.push(crate_doc);
                }
            }
        }

        // Scan for Rust source files
        for entry in WalkDir::new(&self.project_root) {
            let entry = entry?;
            if entry.path().extension().map(|e| e == "rs").unwrap_or(false) {
                if let Ok(module_doc) = self.parse_module(entry.path()) {
                    modules.push(module_doc);
                }
            }
        }

        // Parse API endpoints from api-openai crate
        if let Ok(endpoints) = self.parse_api_endpoints() {
            api_endpoints = endpoints;
        }

        // Define build targets
        build_targets = self.define_build_targets();

        Ok(ProjectDoc {
            name: "DeCoupled-AI".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "Native systems AI engine with OpenAI-compatible API".to_string(),
            crates,
            modules,
            api_endpoints,
            build_targets,
        })
    }

    fn parse_crate(&self, cargo_toml: &Path) -> Result<CrateDoc> {
        let content = fs::read_to_string(cargo_toml)?;
        let manifest: toml::Value = content.parse()?;

        let name = manifest.get("package")
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("unknown")
            .to_string();

        let version = manifest.get("package")
            .and_then(|p| p.get("version"))
            .and_then(|v| v.as_str())
            .unwrap_or("0.0.0")
            .to_string();

        let description = manifest.get("package")
            .and_then(|p| p.get("description"))
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .to_string();

        let mut dependencies = Vec::new();
        if let Some(deps) = manifest.get("dependencies").and_then(|d| d.as_table()) {
            for (name, _) in deps {
                dependencies.push(name.clone());
            }
        }

        let mut features = Vec::new();
        if let Some(feats) = manifest.get("features").and_then(|f| f.as_table()) {
            for (name, _) in feats {
                features.push(name.clone());
            }
        }

        // Parse public items from source
        let crate_root = cargo_toml.parent().unwrap();
        let public_items = self.extract_public_items(crate_root)?;

        Ok(CrateDoc {
            name,
            path: crate_root.strip_prefix(&self.project_root)?.to_string_lossy().to_string(),
            description,
            version,
            dependencies,
            features,
            public_items,
        })
    }

    fn parse_module(&self, file_path: &Path) -> Result<ModuleDoc> {
        let content = fs::read_to_string(file_path)?;
        let relative_path = file_path.strip_prefix(&self.project_root)?.to_string_lossy().to_string();

        // Simple regex-based parsing (in production, use syn)
        let mut items = Vec::new();
        let mut current_doc = String::new();

        for line in content.lines() {
            let trimmed = line.trim();

            // Collect doc comments
            if trimmed.starts_with("///") || trimmed.starts_with("//!") {
                current_doc.push_str(&trimmed[3..].trim_start());
                current_doc.push('\n');
                continue;
            }

            // Parse public items
            if trimmed.starts_with("pub ") {
                let item = self.parse_public_item(trimmed, &current_doc, &relative_path)?;
                if let Some(item) = item {
                    items.push(item);
                }
                current_doc.clear();
            } else if !trimmed.is_empty() && !trimmed.starts_with("//") {
                current_doc.clear();
            }
        }

        Ok(ModuleDoc {
            path: relative_path.clone(),
            name: file_path.file_stem().unwrap().to_string_lossy().to_string(),
            doc: current_doc,
            items,
        })
    }

    fn parse_public_item(&self, line: &str, doc: &str, module: &str) -> Result<Option<PublicItem>> {
        let kind = if line.contains("pub struct") {
            ItemKind::Struct
        } else if line.contains("pub enum") {
            ItemKind::Enum
        } else if line.contains("pub fn") || line.contains("pub async fn") {
            ItemKind::Function
        } else if line.contains("pub const") {
            ItemKind::Const
        } else if line.contains("pub static") {
            ItemKind::Static
        } else if line.contains("pub trait") {
            ItemKind::Trait
        } else if line.contains("pub type") {
            ItemKind::TypeAlias
        } else if line.contains("pub mod") {
            return Ok(None); // Skip module declarations
        } else {
            return Ok(None);
        };

        // Extract signature (simplified)
        let signature = line.trim_start_matches("pub ").to_string();

        // Check for safety notes
        let mut safety_notes = Vec::new();
        if doc.contains("unsafe") || doc.contains("Safety") || doc.contains("safety") {
            safety_notes.push("Contains unsafe code - review safety invariants".to_string());
        }

        Ok(Some(PublicItem {
            name: self.extract_name(&signature),
            kind,
            signature,
            doc: doc.to_string(),
            module: module.to_string(),
            safety_notes,
        }))
    }

    fn extract_name(&self, signature: &str) -> String {
        // Simple name extraction
        let parts: Vec<&str> = signature.split_whitespace().collect();
        if parts.len() >= 2 {
            parts[1].split('(').next().unwrap_or("unknown").split('<').next().unwrap_or("unknown").trim_end_matches(':').to_string()
        } else {
            "unknown".to_string()
        }
    }

    fn extract_public_items(&self, crate_root: &Path) -> Result<Vec<PublicItem>> {
        let mut items = Vec::new();
        for entry in WalkDir::new(crate_root.join("src")) {
            let entry = entry?;
            if entry.path().extension().map(|e| e == "rs").unwrap_or(false) {
                if let Ok(module) = self.parse_module(entry.path()) {
                    items.extend(module.items);
                }
            }
        }
        Ok(items)
    }

    fn parse_api_endpoints(&self) -> Result<Vec<ApiEndpoint>> {
        // In real implementation, parse from api-openai source
        Ok(vec![
            ApiEndpoint {
                method: "GET".to_string(),
                path: "/v1/models".to_string(),
                summary: "List available models".to_string(),
                description: "Returns a list of loaded .brain models".to_string(),
                request_schema: None,
                response_schema: Some(serde_json::json!({"type": "object", "properties": {"object": {"type": "string"}, "data": {"type": "array"}}})),
                examples: vec![],
                auth_required: true,
            },
            ApiEndpoint {
                method: "POST".to_string(),
                path: "/v1/chat/completions".to_string(),
                summary: "Create chat completion".to_string(),
                description: "Generate a chat completion with optional streaming".to_string(),
                request_schema: Some(serde_json::json!({"type": "object", "properties": {"model": {"type": "string"}, "messages": {"type": "array"}, "stream": {"type": "boolean"}}})),
                response_schema: Some(serde_json::json!({"type": "object", "properties": {"id": {"type": "string"}, "choices": {"type": "array"}}})),
                examples: vec![],
                auth_required: true,
            },
        ])
    }

    fn define_build_targets(&self) -> Vec<BuildTarget> {
        vec![
            BuildTarget {
                name: "linux-x86_64-cuda".to_string(),
                platform: "linux".to_string(),
                arch: "x86_64".to_string(),
                features: vec!["cuda".to_string()],
                dependencies: vec!["nvcc".to_string(), "cuda-toolkit".to_string()],
                build_steps: vec![
                    "cargo build --release --features cuda".to_string(),
                ],
                artifacts: vec!["target/release/decoupled-ai-server".to_string()],
            },
            BuildTarget {
                name: "linux-x86_64-cpu".to_string(),
                platform: "linux".to_string(),
                arch: "x86_64".to_string(),
                features: vec!["cpu".to_string()],
                dependencies: vec!["gcc".to_string(), "clang".to_string()],
                build_steps: vec![
                    "cargo build --release --features cpu".to_string(),
                ],
                artifacts: vec!["target/release/decoupled-ai-server".to_string()],
            },
            BuildTarget {
                name: "macos-arm64-metal".to_string(),
                platform: "macos".to_string(),
                arch: "aarch64".to_string(),
                features: vec!["metal".to_string()],
                dependencies: vec!["xcode".to_string(), "rustup target add aarch64-apple-darwin".to_string()],
                build_steps: vec![
                    "cargo build --release --features metal --target aarch64-apple-darwin".to_string(),
                ],
                artifacts: vec!["target/aarch64-apple-darwin/release/decoupled-ai-server".to_string()],
            },
            BuildTarget {
                name: "windows-x86_64-cpu".to_string(),
                platform: "windows".to_string(),
                arch: "x86_64".to_string(),
                features: vec!["cpu".to_string()],
                dependencies: vec!["msvc".to_string(), "windows-sdk".to_string()],
                build_steps: vec![
                    "cargo build --release --features cpu".to_string(),
                ],
                artifacts: vec!["target/release/decoupled-ai-server.exe".to_string()],
            },
        ]
    }

    // ========================================================================
    // Output Generation
    // ========================================================================

    fn generate_spec(&self, project: &ProjectDoc) -> Result<()> {
        let content = self.templates.render("spec", project)?;
        fs::write(self.output_dir.join("SPEC.md"), content)?;
        Ok(())
    }

    fn generate_api_reference(&self, project: &ProjectDoc) -> Result<()> {
        let content = self.templates.render("api_reference", project)?;
        fs::write(self.output_dir.join("API_REFERENCE.md"), content)?;
        Ok(())
    }

    fn generate_build_guide(&self, project: &ProjectDoc) -> Result<()> {
        let content = self.templates.render("build", project)?;
        fs::write(self.output_dir.join("BUILD.md"), content)?;
        Ok(())
    }

    fn generate_readme(&self, project: &ProjectDoc) -> Result<()> {
        let content = self.templates.render("readme", project)?;
        fs::write(self.output_dir.join("README.md"), content)?;
        Ok(())
    }

    fn generate_changelog(&self, project: &ProjectDoc) -> Result<()> {
        let content = self.templates.render("changelog", project)?;
        fs::write(self.output_dir.join("CHANGELOG.md"), content)?;
        Ok(())
    }

    fn audit_code_comments(&self, project: &ProjectDoc) -> Result<()> {
        let mut audit_report = String::new();
        audit_report.push_str("# Code Documentation Audit\n\n");
        audit_report.push_str(&format!("Generated: {}\n\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));

        let mut unsafe_blocks = 0;
        let mut undocumented_unsafe = 0;
        let mut missing_docs = 0;

        for krate in &project.crates {
            for item in &krate.public_items {
                if item.kind == ItemKind::Function || item.kind == ItemKind::Method {
                    if item.doc.trim().is_empty() {
                        missing_docs += 1;
                        audit_report.push_str(&format!("- **Missing docs**: `{}::{}` ({})\n", krate.name, item.name, item.signature));
                    }
                }

                for safety in &item.safety_notes {
                    unsafe_blocks += 1;
                    if !item.doc.contains("Safety") && !item.doc.contains("safety") && !item.doc.contains("unsafe") {
                        undocumented_unsafe += 1;
                        audit_report.push_str(&format!("- **Undocumented unsafe**: `{}::{}` - {}\n", krate.name, item.name, safety));
                    }
                }
            }
        }

        audit_report.push_str(&format!("\n## Summary\n"));
        audit_report.push_str(&format!("- Total public items: {}\n", project.crates.iter().map(|c| c.public_items.len()).sum::<usize>()));
        audit_report.push_str(&format!("- Missing documentation: {}\n", missing_docs));
        audit_report.push_str(&format!("- Unsafe blocks found: {}\n", unsafe_blocks));
        audit_report.push_str(&format!("- Undocumented unsafe: {}\n", undocumented_unsafe));

        fs::write(self.output_dir.join("AUDIT.md"), audit_report)?;
        Ok(())
    }
}

// ============================================================================
// CLI
// ============================================================================

#[derive(clap::Parser)]
#[command(name = "docs-gen", about = "Generate DeCoupled-AI documentation")]
struct Cli {
    #[arg(short, long, default_value = ".")]
    project: PathBuf,

    #[arg(short, long, default_value = "./docs")]
    output: PathBuf,
}

pub fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    let generator = DocsGenerator::new(&cli.project, &cli.output)?;
    generator.generate_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_name() {
        let gen = DocsGenerator::new(".", "./docs").unwrap();
        assert_eq!(gen.extract_name("struct Foo"), "Foo");
        assert_eq!(gen.extract_name("fn bar()"), "bar");
        assert_eq!(gen.extract_name("const BAZ: i32"), "BAZ");
    }
}