//! GPU Detection and Auto-Selection
//!
//! Detects available GPUs (NVIDIA, AMD, Intel) and provides
//! interactive selection when multiple GPUs are present.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;
use tracing::{debug, info, warn};

/// GPU vendor types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown,
}

impl std::fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuVendor::Nvidia => write!(f, "NVIDIA"),
            GpuVendor::Amd => write!(f, "AMD"),
            GpuVendor::Intel => write!(f, "Intel"),
            GpuVendor::Apple => write!(f, "Apple"),
            GpuVendor::Unknown => write!(f, "Unknown"),
        }
    }
}

impl GpuVendor {
    pub fn as_str(&self) -> &'static str {
        match self {
            GpuVendor::Nvidia => "NVIDIA",
            GpuVendor::Amd => "AMD",
            GpuVendor::Intel => "Intel",
            GpuVendor::Apple => "Apple",
            GpuVendor::Unknown => "Unknown",
        }
    }
}

/// GPU backend compute capability
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuBackend {
    Cuda,
    Rocm,
    Metal,
    OpenCl,
    Cpu,
    Unknown,
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackend::Cuda => write!(f, "CUDA"),
            GpuBackend::Rocm => write!(f, "ROCm"),
            GpuBackend::Metal => write!(f, "Metal"),
            GpuBackend::OpenCl => write!(f, "OpenCL"),
            GpuBackend::Cpu => write!(f, "CPU"),
            GpuBackend::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Detected GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: usize,
    pub name: String,
    pub vendor: GpuVendor,
    pub backend: GpuBackend,
    pub vram_mb: Option<u64>,
    pub driver_version: Option<String>,
    pub compute_capability: Option<String>,
    pub is_integrated: bool,
    pub pci_id: Option<String>,
}

/// GPU detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDetectionResult {
    pub gpus: Vec<GpuInfo>,
    pub selected_gpu: Option<GpuInfo>,
    pub selection_method: SelectionMethod,
    pub backend: GpuBackend,
}

/// How the GPU was selected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionMethod {
    AutoSingle,      // Only one GPU available, auto-selected
    AutoBest,        // Multiple GPUs, best one auto-selected
    ManualPrompt,    // User selected via interactive prompt
    ManualConfig,    // Selected from config file
    Forced,          // Explicitly forced via CLI/env
}

/// Detect all available GPUs
pub fn detect_gpus() -> Result<Vec<GpuInfo>> {
    let mut gpus = Vec::new();

    // Try NVIDIA detection first
    if let Ok(nvidia_gpus) = detect_nvidia_gpus() {
        gpus.extend(nvidia_gpus);
    }

    // Try AMD detection
    if let Ok(amd_gpus) = detect_amd_gpus() {
        gpus.extend(amd_gpus);
    }

    // Try Intel detection
    if let Ok(intel_gpus) = detect_intel_gpus() {
        gpus.extend(intel_gpus);
    }

    // Try Apple Silicon (Metal)
    #[cfg(target_os = "macos")]
    if let Ok(apple_gpus) = detect_apple_gpus() {
        gpus.extend(apple_gpus);
    }

    // Fallback to OpenCL detection
    if gpus.is_empty() {
        if let Ok(opencl_gpus) = detect_opencl_gpus() {
            gpus.extend(opencl_gpus);
        }
    }

    // Sort by preference: discrete NVIDIA > discrete AMD > discrete Intel > integrated > CPU
    gpus.sort_by(|a, b| gpu_preference_score(b).cmp(&gpu_preference_score(a)));

    // Assign indices
    for (i, gpu) in gpus.iter_mut().enumerate() {
        gpu.index = i;
    }

    info!("Detected {} GPU(s)", gpus.len());
    for gpu in &gpus {
        info!(
            "  GPU {}: {} ({}) - VRAM: {} - Backend: {}",
            gpu.index,
            gpu.name,
            gpu.vendor,
            gpu.vram_mb.map_or("Unknown".to_string(), |v| format!("{} MB", v)),
            gpu.backend
        );
    }

    Ok(gpus)
}

/// Calculate preference score for GPU sorting (higher = better)
fn gpu_preference_score(gpu: &GpuInfo) -> i32 {
    let mut score = 0;

    // Vendor preference
    score += match gpu.vendor {
        GpuVendor::Nvidia => 1000,
        GpuVendor::Amd => 800,
        GpuVendor::Intel => 600,
        GpuVendor::Apple => 900,
        GpuVendor::Unknown => 100,
    };

    // Discrete GPU bonus
    if !gpu.is_integrated {
        score += 500;
    }

    // VRAM bonus
    if let Some(vram) = gpu.vram_mb {
        score += (vram / 1024).min(100) as i32; // Up to 100 points for VRAM
    }

    // Backend preference
    score += match gpu.backend {
        GpuBackend::Cuda => 200,
        GpuBackend::Metal => 150,
        GpuBackend::Rocm => 100,
        GpuBackend::OpenCl => 50,
        GpuBackend::Cpu => 0,
        GpuBackend::Unknown => 0,
    };

    score
}

/// Detect NVIDIA GPUs using nvidia-smi
fn detect_nvidia_gpus() -> Result<Vec<GpuInfo>> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,driver_version,compute_cap,pci.bus_id",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .context("Failed to run nvidia-smi")?;

    if !output.status.success() {
        return Err(anyhow!("nvidia-smi failed"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() >= 6 {
            let index = parts[0].parse::<usize>().unwrap_or(gpus.len());
            let name = parts[1].to_string();
            let vram_mb = parts[2].parse::<u64>().ok();
            let driver_version = Some(parts[3].to_string());
            let compute_capability = Some(parts[4].to_string());
            let pci_id = Some(parts[5].to_string());

            // Check if integrated (mobile/laptop GPUs often have lower VRAM)
            let is_integrated = name.to_lowercase().contains("notebook")
                || name.to_lowercase().contains("laptop")
                || name.to_lowercase().contains("mobile")
                || name.to_lowercase().contains("max-q");

            gpus.push(GpuInfo {
                index,
                name,
                vendor: GpuVendor::Nvidia,
                backend: GpuBackend::Cuda,
                vram_mb,
                driver_version,
                compute_capability,
                is_integrated,
                pci_id,
            });
        }
    }

    Ok(gpus)
}

/// Detect AMD GPUs using rocm-smi or lspci
fn detect_amd_gpus() -> Result<Vec<GpuInfo>> {
    let mut gpus = Vec::new();

    // Try rocm-smi first (Linux with ROCm)
    if let Ok(output) = Command::new("rocm-smi").args(["--showproductname", "--showmeminfo", "vram", "--csv"]).output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for (i, line) in stdout.lines().skip(1).enumerate() { // Skip header
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 3 {
                    let name = parts[1].to_string();
                    let vram_mb = parts[2].parse::<u64>().ok().map(|v| v / 1024 / 1024);

                    gpus.push(GpuInfo {
                        index: i,
                        name,
                        vendor: GpuVendor::Amd,
                        backend: GpuBackend::Rocm,
                        vram_mb,
                        driver_version: None,
                        compute_capability: None,
                        is_integrated: false, // ROCm typically discrete
                        pci_id: None,
                    });
                }
            }
            return Ok(gpus);
        }
    }

    // Fallback: lspci for AMD GPUs
    if let Ok(output) = Command::new("lspci").args(["-nn"]).output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for (i, line) in stdout.lines().enumerate() {
                if line.to_lowercase().contains("vga") || line.to_lowercase().contains("display") {
                    if line.contains("1002:") || line.contains("AMD") || line.contains("ATI") {
                        // Extract PCI ID
                        let pci_id = line.split('[').nth(1).and_then(|s| s.split(']').next()).map(|s| s.to_string());

                        gpus.push(GpuInfo {
                            index: i,
                            name: line.trim().to_string(),
                            vendor: GpuVendor::Amd,
                            backend: GpuBackend::Rocm,
                            vram_mb: None,
                            driver_version: None,
                            compute_capability: None,
                            is_integrated: line.to_lowercase().contains("integrated") || line.to_lowercase().contains("apu"),
                            pci_id,
                        });
                    }
                }
            }
        }
    }

    // Windows: use wmic
    #[cfg(target_os = "windows")]
    {
        if let Ok(output) = Command::new("wmic").args(["path", "win32_VideoController", "get", "Name,AdapterRAM,DriverVersion,PNPDeviceID", "/format:csv"]).output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for (i, line) in stdout.lines().skip(1).enumerate() {
                    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if parts.len() >= 4 && parts[1].to_lowercase().contains("amd") {
                        let vram_mb = parts[2].parse::<u64>().ok().map(|v| v / 1024 / 1024);
                        gpus.push(GpuInfo {
                            index: i,
                            name: parts[1].to_string(),
                            vendor: GpuVendor::Amd,
                            backend: GpuBackend::Rocm,
                            vram_mb,
                            driver_version: Some(parts[3].to_string()),
                            compute_capability: None,
                            is_integrated: parts[1].to_lowercase().contains("integrated"),
                            pci_id: Some(parts[4].to_string()),
                        });
                    }
                }
            }
        }
    }

    Ok(gpus)
}

/// Detect Intel GPUs
fn detect_intel_gpus() -> Result<Vec<GpuInfo>> {
    let mut gpus = Vec::new();

    // Linux: lspci
    if let Ok(output) = Command::new("lspci").args(["-nn"]).output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for (i, line) in stdout.lines().enumerate() {
                if line.to_lowercase().contains("vga") || line.to_lowercase().contains("display") {
                    if line.contains("8086:") { // Intel vendor ID
                        let pci_id = line.split('[').nth(1).and_then(|s| s.split(']').next()).map(|s| s.to_string());

                        gpus.push(GpuInfo {
                            index: i,
                            name: line.trim().to_string(),
                            vendor: GpuVendor::Intel,
                            backend: GpuBackend::OpenCl, // Intel typically uses OpenCL or oneAPI
                            vram_mb: None, // Shared memory
                            driver_version: None,
                            compute_capability: None,
                            is_integrated: true, // Intel GPUs are typically integrated
                            pci_id,
                        });
                    }
                }
            }
        }
    }

    // Windows: wmic
    #[cfg(target_os = "windows")]
    {
        if let Ok(output) = Command::new("wmic").args(["path", "win32_VideoController", "get", "Name,AdapterRAM,DriverVersion,PNPDeviceID", "/format:csv"]).output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for (i, line) in stdout.lines().skip(1).enumerate() {
                    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if parts.len() >= 4 && (parts[1].to_lowercase().contains("intel") || parts[1].to_lowercase().contains("uhd") || parts[1].to_lowercase().contains("iris") || parts[1].to_lowercase().contains("arc")) {
                        let vram_mb = parts[2].parse::<u64>().ok().map(|v| v / 1024 / 1024);
                        gpus.push(GpuInfo {
                            index: i,
                            name: parts[1].to_string(),
                            vendor: GpuVendor::Intel,
                            backend: GpuBackend::OpenCl,
                            vram_mb,
                            driver_version: Some(parts[3].to_string()),
                            compute_capability: None,
                            is_integrated: true,
                            pci_id: Some(parts[4].to_string()),
                        });
                    }
                }
            }
        }
    }

    Ok(gpus)
}

/// Detect Apple Silicon GPUs (Metal)
#[cfg(target_os = "macos")]
fn detect_apple_gpus() -> Result<Vec<GpuInfo>> {
    let mut gpus = Vec::new();

    // Use system_profiler for macOS
    if let Ok(output) = Command::new("system_profiler").args(["SPDisplaysDataType", "-json"]).output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
                if let Some(displays) = json.get("SPDisplaysDataType").and_then(|v| v.as_array()) {
                    for (i, display) in displays.iter().enumerate() {
                        if let Some(name) = display.get("sppci_model").and_then(|v| v.as_str()) {
                            let vram = display.get("sppci_vram").and_then(|v| v.as_str()).and_then(|s| s.split_whitespace().next()).and_then(|s| s.parse::<u64>().ok());
                            let is_metal = display.get("sppci_metal_support").and_then(|v| v.as_str()) == Some("Metal");

                            gpus.push(GpuInfo {
                                index: i,
                                name: name.to_string(),
                                vendor: GpuVendor::Apple,
                                backend: GpuBackend::Metal,
                                vram_mb: vram,
                                driver_version: None,
                                compute_capability: None,
                                is_integrated: true, // Apple Silicon is unified memory
                                pci_id: None,
                            });
                        }
                    }
                }
            }
        }
    }

    Ok(gpus)
}

/// Fallback: OpenCL detection
fn detect_opencl_gpus() -> Result<Vec<GpuInfo>> {
    // Try to use OpenCL via clinfo
    if let Ok(output) = Command::new("clinfo").output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut gpus = Vec::new();
            let mut current_gpu: Option<GpuInfo> = None;
            let mut index = 0;

            for line in stdout.lines() {
                let line = line.trim();
                if line.starts_with("Platform Name") {
                    // New platform
                } else if line.starts_with("Device Name") {
                    if let Some(gpu) = current_gpu.take() {
                        gpus.push(gpu);
                    }
                    let name = line.split(':').nth(1).unwrap_or("").trim().to_string();
                    current_gpu = Some(GpuInfo {
                        index,
                        name: name.clone(),
                        vendor: detect_vendor_from_name(&name),
                        backend: GpuBackend::OpenCl,
                        vram_mb: None,
                        driver_version: None,
                        compute_capability: None,
                        is_integrated: name.to_lowercase().contains("integrated") || name.to_lowercase().contains("intel"),
                        pci_id: None,
                    });
                    index += 1;
                } else if line.starts_with("Global Memory Size") && current_gpu.is_some() {
                    let size_str = line.split(':').nth(1).unwrap_or("").trim();
                    if let Some(bytes_str) = size_str.split_whitespace().next() {
                        if let Ok(bytes) = bytes_str.parse::<u64>() {
                            current_gpu.as_mut().unwrap().vram_mb = Some(bytes / 1024 / 1024);
                        }
                    }
                } else if line.starts_with("Driver Version") && current_gpu.is_some() {
                    let version = line.split(':').nth(1).unwrap_or("").trim().to_string();
                    current_gpu.as_mut().unwrap().driver_version = Some(version);
                }
            }
            if let Some(gpu) = current_gpu {
                gpus.push(gpu);
            }
            return Ok(gpus);
        }
    }

    Err(anyhow!("OpenCL detection failed"))
}

/// Detect vendor from GPU name
fn detect_vendor_from_name(name: &str) -> GpuVendor {
    let name = name.to_lowercase();
    if name.contains("nvidia") || name.contains("geforce") || name.contains("quadro") || name.contains("tesla") || name.contains("rtx") || name.contains("gtx") {
        GpuVendor::Nvidia
    } else if name.contains("amd") || name.contains("radeon") || name.contains("rx ") || name.contains("firepro") {
        GpuVendor::Amd
    } else if name.contains("intel") || name.contains("uhd") || name.contains("iris") || name.contains("arc ") || name.contains("hd graphics") {
        GpuVendor::Intel
    } else if name.contains("apple") || name.contains("m1") || name.contains("m2") || name.contains("m3") {
        GpuVendor::Apple
    } else {
        GpuVendor::Unknown
    }
}

/// Auto-select the best GPU
pub fn auto_select_gpu(gpus: &[GpuInfo]) -> Option<GpuInfo> {
    gpus.first().cloned()
}

/// Interactive GPU selection prompt
pub fn prompt_gpu_selection(gpus: &[GpuInfo]) -> Result<GpuInfo> {
    if gpus.is_empty() {
        return Err(anyhow!("No GPUs available for selection"));
    }

    if gpus.len() == 1 {
        info!("Only one GPU detected: {}", gpus[0].name);
        return Ok(gpus[0].clone());
    }

    println!("\n=== Multiple GPUs Detected ===");
    for (i, gpu) in gpus.iter().enumerate() {
        let vram = gpu.vram_mb.map_or("Unknown".to_string(), |v| format!("{} MB", v));
        let backend = gpu.backend.to_string();
        let integrated = if gpu.is_integrated { " (Integrated)" } else { "" };
        println!("  [{}] {} [{}] - VRAM: {}{}", i + 1, gpu.name, backend, vram, integrated);
    }
    println!("  [{}] CPU Only (No GPU acceleration)", gpus.len() + 1);
    println!("==============================\n");

    use std::io::{self, Write};
    print!("Select GPU [1-{}]: ", gpus.len() + 1);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let choice = input.trim().parse::<usize>().unwrap_or(0);

    if choice == 0 || choice > gpus.len() + 1 {
        println!("Invalid selection, defaulting to best GPU (auto)");
        return Ok(gpus[0].clone());
    }

    if choice == gpus.len() + 1 {
        // CPU only
        return Err(anyhow!("User selected CPU only"));
    }

    Ok(gpus[choice - 1].clone())
}

/// Detect GPUs without prompting
pub fn get_gpu_info() -> Result<Vec<GpuInfo>> {
    detect_gpus()
}

/// Detect and select GPU with optional prompt and preferred backend
pub fn detect_and_select_gpu(preferred_backend: Option<&str>, prompt: bool) -> Result<GpuDetectionResult> {
    let gpus = detect_gpus()?;

    if gpus.is_empty() {
        warn!("No GPUs detected, falling back to CPU");
        return Ok(GpuDetectionResult {
            gpus: Vec::new(),
            selected_gpu: None,
            selection_method: SelectionMethod::AutoSingle,
            backend: GpuBackend::Cpu,
        });
    }

    // Filter by preferred backend if specified
    let candidates = if let Some(backend_str) = preferred_backend {
        let preferred = match backend_str.to_lowercase().as_str() {
            "cuda" => Some(GpuBackend::Cuda),
            "rocm" => Some(GpuBackend::Rocm),
            "metal" => Some(GpuBackend::Metal),
            "opencl" => Some(GpuBackend::OpenCl),
            _ => {
                warn!("Unknown preferred backend '{}', considering all GPUs", backend_str);
                None
            }
        };
        if let Some(pref) = preferred {
            gpus.iter().filter(|g| g.backend == pref).cloned().collect::<Vec<_>>()
        } else {
            gpus.clone()
        }
    } else {
        gpus.clone()
    };

    let (selected_gpu, method) = if prompt && candidates.len() > 1 {
        match prompt_gpu_selection(&candidates) {
            Ok(gpu) => (Some(gpu), SelectionMethod::ManualPrompt),
            Err(_) => {
                info!("User chose CPU only or cancelled");
                (None, SelectionMethod::ManualPrompt)
            }
        }
    } else if candidates.len() == 1 {
        (Some(candidates[0].clone()), SelectionMethod::AutoSingle)
    } else if !candidates.is_empty() {
        (auto_select_gpu(&candidates), SelectionMethod::AutoBest)
    } else {
        // No GPUs matching preferred backend, fall back to all GPUs
        if gpus.len() == 1 {
            (Some(gpus[0].clone()), SelectionMethod::AutoSingle)
        } else {
            (auto_select_gpu(&gpus), SelectionMethod::AutoBest)
        }
    };

    // Determine the effective backend
    let backend = selected_gpu.as_ref().map(|g| g.backend).unwrap_or(GpuBackend::Cpu);

    if let Some(ref gpu) = selected_gpu {
        info!("Selected GPU: {} ({}) via {:?}", gpu.name, gpu.backend, method);
    } else {
        info!("No GPU selected, using CPU");
    }

    Ok(GpuDetectionResult {
        gpus,
        selected_gpu,
        selection_method: method,
        backend,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_preference_score() {
        let nvidia_discrete = GpuInfo {
            index: 0,
            name: "RTX 3080".to_string(),
            vendor: GpuVendor::Nvidia,
            backend: GpuBackend::Cuda,
            vram_mb: Some(10240),
            driver_version: None,
            compute_capability: None,
            is_integrated: false,
            pci_id: None,
        };

        let intel_integrated = GpuInfo {
            index: 1,
            name: "UHD Graphics 630".to_string(),
            vendor: GpuVendor::Intel,
            backend: GpuBackend::OpenCl,
            vram_mb: None,
            driver_version: None,
            compute_capability: None,
            is_integrated: true,
            pci_id: None,
        };

        assert!(gpu_preference_score(&nvidia_discrete) > gpu_preference_score(&intel_integrated));
    }

    #[test]
    fn test_vendor_detection() {
        assert_eq!(detect_vendor_from_name("NVIDIA GeForce RTX 3080"), GpuVendor::Nvidia);
        assert_eq!(detect_vendor_from_name("AMD Radeon RX 6800"), GpuVendor::Amd);
        assert_eq!(detect_vendor_from_name("Intel UHD Graphics 630"), GpuVendor::Intel);
        assert_eq!(detect_vendor_from_name("Apple M1 Pro"), GpuVendor::Apple);
        assert_eq!(detect_vendor_from_name("Unknown GPU"), GpuVendor::Unknown);
    }
}