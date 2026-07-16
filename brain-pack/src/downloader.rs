//! Hugging Face Hub Downloader Module
//!
//! Native Rust downloader for fetching model files from Hugging Face Hub.
//! Supports parallel chunk downloads, progress bars, and redirect handling.

use anyhow::{anyhow, Context, Result};
use flume::{bounded, Receiver, Sender};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use std::path::Path;
use std::time::Duration;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

const HF_HUB_BASE: &str = "https://huggingface.co";
const CHUNK_SIZE: u64 = 1024 * 1024; // 1 MiB chunks
const MAX_CONCURRENT_CHUNKS: usize = 4;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Download configuration
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    pub repo_id: String,
    pub revision: String,
    pub files: Vec<String>,
    pub output_dir: std::path::PathBuf,
    pub token: Option<String>,
    pub max_concurrent_chunks: usize,
    pub chunk_size: u64,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            repo_id: String::new(),
            revision: "main".to_string(),
            files: Vec::new(),
            output_dir: std::path::PathBuf::from("."),
            token: None,
            max_concurrent_chunks: MAX_CONCURRENT_CHUNKS,
            chunk_size: CHUNK_SIZE,
        }
    }
}

/// Progress information for a single file download
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub file_name: String,
    pub bytes_downloaded: u64,
    pub total_bytes: u64,
    pub speed_bps: f64,
    pub finished: bool,
    pub error: Option<String>,
}

/// Hugging Face file metadata from the API
#[derive(Debug, serde::Deserialize)]
pub struct HFFileInfo {
    pub path: String,
    pub size: u64,
    pub lfs: Option<serde_json::Value>,
}

/// Hugging Face repository tree response - API returns array directly
#[derive(Debug, serde::Deserialize)]
pub struct HFRepoTree(pub Vec<HFFileInfo>);

/// Downloader for Hugging Face Hub models
pub struct HFDownloader {
    client: Client,
    config: DownloadConfig,
    progress_tx: Sender<DownloadProgress>,
    progress_rx: Receiver<DownloadProgress>,
}

impl HFDownloader {
    pub fn new(config: DownloadConfig) -> Self {
        let client = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .user_agent("DeCoupled-AI/brain-pack")
            .build()
            .expect("Failed to create HTTP client");

        let (progress_tx, progress_rx) = bounded(100);

        Self {
            client,
            config,
            progress_tx,
            progress_rx,
        }
    }

    /// Get the progress receiver for UI updates
    pub fn progress_receiver(&self) -> &Receiver<DownloadProgress> {
        &self.progress_rx
    }

    /// Download all files specified in config
    pub async fn download_all(&self) -> Result<()> {
        // Ensure output directory exists
        tokio::fs::create_dir_all(&self.config.output_dir).await?;

        // Fetch file metadata from HF Hub API
        let file_infos = self.fetch_file_metadata().await?;

        // Download each file
        for file_info in file_infos {
            if !self.config.files.is_empty() && !self.config.files.contains(&file_info.path) {
                continue;
            }
            self.download_file(&file_info).await?;
        }

        Ok(())
    }

    /// Fetch file metadata from Hugging Face Hub API
    async fn fetch_file_metadata(&self) -> Result<Vec<HFFileInfo>> {
        // Use the API endpoint directly (not the web UI endpoint)
        let url = format!(
            "{}/api/models/{}/tree/{}",
            HF_HUB_BASE, self.config.repo_id, self.config.revision
        );

        let mut request = self.client.get(&url);
        if let Some(token) = &self.config.token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await
            .context("Failed to fetch model tree from HF Hub")?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Failed to fetch model metadata: HTTP {}",
                response.status()
            ));
        }

        let tree: HFRepoTree = response.json().await?;
        Ok(tree.0)
    }

    /// Download a single file with parallel chunk support
    async fn download_file(&self, file_info: &HFFileInfo) -> Result<()> {
        let output_path = self.config.output_dir.join(&file_info.path);

        // Create parent directories if needed
        if let Some(parent) = output_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Check if file already exists and is complete
        if output_path.exists() {
            let metadata = tokio::fs::metadata(&output_path).await?;
            if metadata.len() == file_info.size {
                self.send_progress(DownloadProgress {
                    file_name: file_info.path.clone(),
                    bytes_downloaded: file_info.size,
                    total_bytes: file_info.size,
                    speed_bps: 0.0,
                    finished: true,
                    error: None,
                }).await;
                return Ok(());
            }
        }

        // Get download URL (handles redirects for LFS files)
        let download_url = self.get_download_url(file_info).await?;

        // Download with progress tracking
        self.download_with_progress(&download_url, &output_path, file_info).await
    }

    /// Get the actual download URL (follows redirects for LFS)
    async fn get_download_url(&self, file_info: &HFFileInfo) -> Result<String> {
        let url = format!(
            "{}/{}/resolve/{}/{}",
            HF_HUB_BASE, self.config.repo_id, self.config.revision, file_info.path
        );

        let mut request = self.client.head(&url);
        if let Some(token) = &self.config.token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await?;
        Ok(response.url().to_string())
    }

    /// Download file with parallel chunks and progress reporting
    async fn download_with_progress(
        &self,
        url: &str,
        output_path: &Path,
        file_info: &HFFileInfo,
    ) -> Result<()> {
        let total_size = file_info.size;
        let chunk_size = self.config.chunk_size.min(total_size);
        let num_chunks = (total_size + chunk_size - 1) / chunk_size;

        // For small files, use single-stream download
        if num_chunks <= 1 {
            return self.download_single_stream(url, output_path, file_info).await;
        }

        // Create progress bar
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message(format!("Downloading {}", file_info.path));

        // Channel for chunk results
        let (chunk_tx, chunk_rx) = bounded(self.config.max_concurrent_chunks);

        // Spawn chunk download tasks
        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_size;
            let end = std::cmp::min(start + chunk_size, total_size);
            let _chunk_len = end - start;

            let client = self.client.clone();
            let url = url.to_string();
            let token = self.config.token.clone();
            let tx = chunk_tx.clone();

            tokio::spawn(async move {
                let mut request = client.get(&url)
                    .header("Range", format!("bytes={}-{}", start, end - 1));
                if let Some(t) = &token {
                    request = request.bearer_auth(t);
                }

                let start_time = std::time::Instant::now();
                let response = request.send().await;

                match response {
                    Ok(resp) if resp.status().is_success() || resp.status().as_u16() == 206 => {
                        let bytes = resp.bytes().await.unwrap_or_default();
                        let elapsed = start_time.elapsed().as_secs_f64();
                        let speed = bytes.len() as f64 / elapsed.max(0.001);
                        let data_vec: Vec<u8> = bytes.to_vec();
                        let _ = tx.send_async(Ok((chunk_idx, data_vec, speed))).await;
                    }
                    Ok(resp) => {
                        let _ = tx.send_async(Err(anyhow!("HTTP {}", resp.status()))).await;
                    }
                    Err(e) => {
                        let _ = tx.send_async(Err(anyhow!(e))).await;
                    }
                }
            });
        }
        drop(chunk_tx); // Close sender

        // Collect chunks and write to file
        let mut file = File::create(output_path).await?;
        let mut chunks_received = 0u64;
        let mut total_speed = 0.0f64;
        let mut chunk_buffers: Vec<Option<(u64, Vec<u8>)>> = vec![None; num_chunks as usize];

        while let Ok(result) = chunk_rx.recv_async().await {
            match result {
                Ok((chunk_idx, data, speed)) => {
                    chunk_buffers[chunk_idx as usize] = Some((chunk_idx, data));
                    total_speed += speed;
                    chunks_received += 1;

                    // Write chunks in order
                    let mut next_expected = 0;
                    while next_expected < num_chunks {
                        if let Some((_, data)) = chunk_buffers[next_expected as usize].take() {
                            file.write_all(&data).await?;
                            pb.inc(data.len() as u64);
                            next_expected += 1;
                        } else {
                            break;
                        }
                    }

                    // Update progress
                    let downloaded = pb.position();
                    if downloaded > 0 {
                        let avg_speed = total_speed / chunks_received as f64;
                        self.send_progress(DownloadProgress {
                            file_name: file_info.path.clone(),
                            bytes_downloaded: downloaded,
                            total_bytes: total_size,
                            speed_bps: avg_speed,
                            finished: false,
                            error: None,
                        }).await;
                    }
                }
                Err(e) => {
                    pb.finish_and_clear();
                    return Err(e.context(format!("Failed to download chunk for {}", file_info.path)));
                }
            }
        }

        file.flush().await?;
        pb.finish_and_clear();

        self.send_progress(DownloadProgress {
            file_name: file_info.path.clone(),
            bytes_downloaded: total_size,
            total_bytes: total_size,
            speed_bps: 0.0,
            finished: true,
            error: None,
        }).await;

        Ok(())
    }

    /// Single-stream download for small files
    async fn download_single_stream(
        &self,
        url: &str,
        output_path: &Path,
        file_info: &HFFileInfo,
    ) -> Result<()> {
        let total_size = file_info.size;

        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message(format!("Downloading {}", file_info.path));

        let mut request = self.client.get(url);
        if let Some(token) = &self.config.token {
            request = request.bearer_auth(token);
        }

        let mut response = request.send().await
            .context("Failed to start download")?;

        if !response.status().is_success() {
            return Err(anyhow!("Download failed: HTTP {}", response.status()));
        }

        let mut file = File::create(output_path).await?;
        let mut downloaded: u64 = 0;
        let start_time = std::time::Instant::now();

        while let Some(chunk) = response.chunk().await? {
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            pb.inc(chunk.len() as u64);

            let elapsed = start_time.elapsed().as_secs_f64();
            let speed = downloaded as f64 / elapsed.max(0.001);

            self.send_progress(DownloadProgress {
                file_name: file_info.path.clone(),
                bytes_downloaded: downloaded,
                total_bytes: total_size,
                speed_bps: speed,
                finished: false,
                error: None,
            }).await;
        }

        file.flush().await?;
        pb.finish_and_clear();

        self.send_progress(DownloadProgress {
            file_name: file_info.path.clone(),
            bytes_downloaded: total_size,
            total_bytes: total_size,
            speed_bps: 0.0,
            finished: true,
            error: None,
        }).await;

        Ok(())
    }

    async fn send_progress(&self, progress: DownloadProgress) {
        let _ = self.progress_tx.send_async(progress).await;
    }
}

/// Convenience function to download a model from HF Hub
pub async fn download_model(
    repo_id: &str,
    files: &[&str],
    output_dir: &Path,
    token: Option<&str>,
    revision: Option<&str>,
) -> Result<Receiver<DownloadProgress>> {
    let config = DownloadConfig {
        repo_id: repo_id.to_string(),
        revision: revision.unwrap_or("main").to_string(),
        files: files.iter().map(|s| s.to_string()).collect(),
        output_dir: output_dir.to_path_buf(),
        token: token.map(|s| s.to_string()),
        ..Default::default()
    };

    let downloader = HFDownloader::new(config);
    let progress_rx = downloader.progress_receiver().clone();

    tokio::spawn(async move {
        if let Err(e) = downloader.download_all().await {
            eprintln!("Download failed: {}", e);
        }
    });

    Ok(progress_rx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_download_config_default() {
        let config = DownloadConfig::default();
        assert_eq!(config.revision, "main");
        assert_eq!(config.max_concurrent_chunks, MAX_CONCURRENT_CHUNKS);
        assert_eq!(config.chunk_size, CHUNK_SIZE);
    }
}