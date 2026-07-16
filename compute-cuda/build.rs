//! Build script for compute-cuda - compiles CUDA kernels to PTX

use std::path::PathBuf;

fn main() {
    // Tell cargo to rerun if kernel files change
    println!("cargo:rerun-if-changed=src/kernels.cu");
    println!("cargo:rerun-if-changed=build.rs");

    // Check for CUDA toolkit
    let cuda_path = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let nvcc = PathBuf::from(&cuda_path).join("bin").join("nvcc");
    if !nvcc.exists() {
        println!("cargo:warning=CUDA compiler (nvcc) not found at {:?}. CUDA kernels will not be compiled.", nvcc);
        return;
    }

    // Compile kernels to PTX
    let kernel_src = PathBuf::from("src/kernels.cu");
    let ptx_out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("kernels.ptx");

    let status = std::process::Command::new(&nvcc)
        .args([
            "--ptx",
            "-O3",
            "-arch=sm_70", // Minimum Volta
            "-arch=sm_80", // Ampere
            "-arch=sm_90", // Hopper
            "--use_fast_math",
            "-lineinfo",
            "-o", ptx_out.to_str().unwrap(),
            kernel_src.to_str().unwrap(),
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-env=KERNELS_PTX={}", ptx_out.display());
            println!("cargo:rerun-if-changed=src/kernels.cu");
        }
        Ok(s) => {
            println!("cargo:warning=NVCC compilation failed with exit code: {:?}", s.code());
        }
        Err(e) => {
            println!("cargo:warning=Failed to execute nvcc: {}", e);
        }
    }

    // Link CUDA runtime
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
}