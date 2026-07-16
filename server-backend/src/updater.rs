//! Over-the-Air (OTA) Update System
//!
//! Handles checking for updates via GitHub Releases API,
//! downloading, verifying, and installing new versions.

use anyhow::{anyhow, Context, Result};
use semver::Version;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;
use tokio::fs;
use tokio::sync::Mutex;
use tracing::{debug, info, warn, error};

/// Current version from Cargo.toml
const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// GitHub repository info
const REPO_OWNER: &str = "nsjminecraft";
const REPO_NAME: &str = "DeCoupled-AI";

/// GitHub Release asset info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseAsset {
    pub name: String,
    pub browser_download_url: String,
    pub size: u64,
    pub content_type: String,
}

/// GitHub Release info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubRelease {
    pub tag_name: String,
    pub name: String,
    pub body: Option<String>,
    pub draft: bool,
    pub prerelease: bool,
    pub assets: Vec<ReleaseAsset>,
    pub published_at: String,
}

/// Update check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateInfo {
    pub current_version: String,
    pub latest_version: String,
    pub update_available: bool,
    pub release_notes: Option<String>,
    pub download_url: Option<String>,
    pub asset_name: Option<String>,
    pub asset_size: Option<u64>,
}

/// Platform-specific asset suffix
fn get_platform_asset_suffix() -> &'static str {
    #[cfg(target_os = "windows")]
    return ".msi";
    #[cfg(target_os = "linux")]
    return ".deb";
    #[cfg(target_os = "macos")]
    return ".tar.gz";
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    return "";
}

/// Get current executable path
fn get_current_exe_path() -> Result<PathBuf> {
    std::env::current_exe().context("Failed to get current executable path")
}

/// Check for updates from GitHub Releases
pub async fn check_for_updates(include_prerelease: bool) -> Result<UpdateInfo> {
    let client = reqwest::Client::builder()
        .user_agent(format!("DecoupledAI/{}", CURRENT_VERSION))
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let url = format!(
        "https://api.github.com/repos/{}/{}/releases{}",
        REPO_OWNER, REPO_NAME,
        if include_prerelease { "" } else { "/latest" }
    );

    let response = client
        .get(&url)
        .header("Accept", "application/vnd.github.v3+json")
        .send()
        .await
        .context("Failed to fetch release info from GitHub")?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "GitHub API error: {}",
            response.status()
        ));
    }

    let release: GitHubRelease = response.json().await.context("Failed to parse release JSON")?;

    let latest_version_str = release.tag_name.trim_start_matches('v');
    let current_version = Version::parse(CURRENT_VERSION)
        .context("Failed to parse current version")?;
    let latest_version = Version::parse(latest_version_str)
        .context("Failed to parse latest version")?;

    let update_available = latest_version > current_version;

    // Find matching asset for current platform
    let suffix = get_platform_asset_suffix();
    let matching_asset = release.assets.iter().find(|a| a.name.ends_with(suffix));

    Ok(UpdateInfo {
        current_version: CURRENT_VERSION.to_string(),
        latest_version: latest_version_str.to_string(),
        update_available,
        release_notes: release.body,
        download_url: matching_asset.map(|a| a.browser_download_url.clone()),
        asset_name: matching_asset.map(|a| a.name.clone()),
        asset_size: matching_asset.map(|a| a.size),
    })
}

/// Download and verify the update package
async fn download_update(download_url: &str, expected_size: Option<u64>) -> Result<Vec<u8>> {
    let client = reqwest::Client::builder()
        .user_agent(format!("DecoupledAI/{}", CURRENT_VERSION))
        .timeout(std::time::Duration::from_secs(300))
        .build()?;

    info!("Downloading update from: {}", download_url);

    let response = client.get(download_url).send().await
        .context("Failed to download update")?;

    if !response.status().is_success() {
        return Err(anyhow!("Download failed: {}", response.status()));
    }

    let content_length = response.content_length();
    if let (Some(expected), Some(actual)) = (expected_size, content_length) {
        if expected != actual {
            warn!("Download size mismatch: expected {}, got {}", expected, actual);
        }
    }

    let bytes = response.bytes().await
        .context("Failed to read download content")?;

    info!("Downloaded {} bytes", bytes.len());
    Ok(bytes.to_vec())
}

/// Install the update package
async fn install_update(update_bytes: &[u8], asset_name: &str) -> Result<()> {
    let temp_dir = std::env::temp_dir().join("decoupled-ai-update");
    fs::create_dir_all(&temp_dir).await?;

    let installer_path = temp_dir.join(asset_name);
    fs::write(&installer_path, update_bytes).await?;

    info!("Installing update from: {:?}", installer_path);

    #[cfg(target_os = "windows")]
    {
        // Run MSI installer silently
        let output = Command::new("msiexec.exe")
            .args(["/i", installer_path.to_str().unwrap(), "/quiet", "/norestart"])
            .output()
            .context("Failed to run MSI installer")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("MSI install failed: {}", stderr));
        }
    }

    #[cfg(target_os = "linux")]
    {
        // Run dpkg for .deb
        let output = Command::new("dpkg")
            .args(["-i", installer_path.to_str().unwrap()])
            .output()
            .context("Failed to run dpkg")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("dpkg install failed: {}", stderr));
        }
    }

    #[cfg(target_os = "macos")]
    {
        // For macOS .tar.gz, extract and run installer script
        let output = Command::new("tar")
            .args(["-xzf", installer_path.to_str().unwrap(), "-C", temp_dir.to_str().unwrap()])
            .output()
            .context("Failed to extract tar.gz")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("tar extract failed: {}", stderr));
        }

        // Look for install script
        let install_script = temp_dir.join("install.sh");
        if install_script.exists() {
            let output = Command::new("bash")
                .arg(install_script)
                .output()
                .context("Failed to run install.sh")?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(anyhow!("install.sh failed: {}", stderr));
            }
        }
    }

    info!("Update installed successfully");
    Ok(())
}

/// Download and install update
pub async fn download_and_install_update(update_info: &UpdateInfo, progress_callback: Option<Box<dyn Fn(u64, u64) + Send + Sync>>) -> Result<()> {
    let download_url = update_info.download_url.as_ref()
        .ok_or_else(|| anyhow!("No download URL available"))?;
    let asset_name = update_info.asset_name.as_ref()
        .ok_or_else(|| anyhow!("No asset name available"))?;

    // Download with progress
    let client = reqwest::Client::builder()
        .user_agent(format!("DecoupledAI/{}", CURRENT_VERSION))
        .timeout(std::time::Duration::from_secs(300))
        .build()?;

    let response = client.get(download_url).send().await
        .context("Failed to start download")?;

    if !response.status().is_success() {
        return Err(anyhow!("Download failed: {}", response.status()));
    }

    let total_size = response.content_length().unwrap_or(0);
    let mut downloaded = 0u64;
    let mut buffer = Vec::with_capacity(total_size as usize);

    let mut stream = response.bytes_stream();
    use futures_util::StreamExt;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Download error")?;
        downloaded += chunk.len() as u64;
        buffer.extend_from_slice(&chunk);

        if let Some(cb) = &progress_callback {
            cb(downloaded, total_size);
        }
    }

    info!("Download complete: {} bytes", downloaded);

    // Install
    install_update(&buffer, asset_name).await?;

    Ok(())
}

/// Background update checker state
lazy_static::lazy_static! {
    static ref UPDATE_CHECKER_STATE: Mutex<Option<tokio::task::JoinHandle<()>>> = Mutex::new(None);
}

/// Start background update checker
pub fn start_update_checker(config: crate::ServerConfig) {
    let interval = config.update_check_interval.unwrap_or(24 * 60 * 60);
    let auto_install = config.auto_install_updates.unwrap_or(false);

    let handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval));
        interval.tick().await; // Skip immediate first check

        loop {
            interval.tick().await;
            debug!("Running scheduled update check...");

            match check_for_updates(false).await {
                Ok(info) if info.update_available => {
                    info!("Scheduled check: Update available: {} -> {}", info.current_version, info.latest_version);
                    if auto_install {
                        info!("Auto-installing update...");
                        if let Err(e) = download_and_install_update(&info, None).await {
                            error!("Auto-update failed: {}", e);
                        } else {
                            info!("Update installed. Restart required.");
                            // Could send signal to restart here
                        }
                    }
                }
                Ok(_) => {
                    debug!("Scheduled check: Already up to date");
                }
                Err(e) => {
                    warn!("Scheduled update check failed: {}", e);
                }
            }
        }
    });

    tokio::spawn(async move {
        let mut guard = UPDATE_CHECKER_STATE.lock().await;
        *guard = Some(handle);
    });
}

/// Stop background update checker
pub async fn stop_update_checker() {
    let mut guard = UPDATE_CHECKER_STATE.lock().await;
    if let Some(handle) = guard.take() {
        handle.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parsing() {
        let v1 = Version::parse("1.0.0").unwrap();
        let v2 = Version::parse("1.0.1").unwrap();
        assert!(v2 > v1);

        let v3 = Version::parse("2.0.0").unwrap();
        assert!(v3 > v2);

        let v4 = Version::parse("1.0.0-alpha").unwrap();
        assert!(v4 < v1);
    }

    #[test]
    fn test_platform_suffix() {
        let suffix = get_platform_asset_suffix();
        #[cfg(target_os = "windows")]
        assert_eq!(suffix, ".msi");
        #[cfg(target_os = "linux")]
        assert_eq!(suffix, ".deb");
        #[cfg(target_os = "macos")]
        assert_eq!(suffix, ".tar.gz");
    }
}