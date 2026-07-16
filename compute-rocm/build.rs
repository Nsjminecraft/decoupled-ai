//! Build script for compute-rocm — compiles HIP kernels to HSACO via hipcc.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels.hip");
    println!("cargo:rerun-if-changed=build.rs");

    // Skip on non-Linux (ROCm is Linux-only for our purposes)
    #[cfg(not(target_os = "linux"))]
    {
        println!("cargo:warning=ROCm backend is only supported on Linux. Skipping HSACO compilation.");
        return;
    }

    #[cfg(target_os = "linux")]
    {
        let hipcc = PathBuf::from("hipcc");
        if which::which(&hipcc).is_err() {
            println!("cargo:warning=hipcc not found in PATH. ROCm kernels will not be compiled.");
            return;
        }

        let kernel_src = PathBuf::from("src/kernels.hip");
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let hsaco_out = PathBuf::from(&out_dir).join("kernels.hsaco");

        let status = std::process::Command::new(&hipcc)
            .args([
                "--genco",
                "--target=gfx900,gfx906,gfx90a,gfx1030,gfx1100",
                "-O3",
                "-o", hsaco_out.to_str().unwrap(),
                kernel_src.to_str().unwrap(),
            ])
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:rustc-env=KERNELS_HSACO={}", hsaco_out.display());
            }
            Ok(s) => {
                println!("cargo:warning=hipcc compilation failed: {:?}", s.code());
            }
            Err(e) => {
                println!("cargo:warning=Failed to execute hipcc: {}", e);
            }
        }
    }
}
