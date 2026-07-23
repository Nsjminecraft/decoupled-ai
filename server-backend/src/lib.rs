//! Web Server Backend for DeCoupled-AI
//!
//! High-performance Axum server with embedded frontend assets,
//! OpenAI-compatible API, and model management.

use anyhow::{anyhow, Result};
use api_openai::OpenAiApi;
use axum::{
    extract::{Path, State, WebSocketUpgrade},
    http::{HeaderMap, StatusCode},
    response::{Html, IntoResponse, Json, Response, sse::Event},
    routing::{get, post},
    Router,
};
use clap::Parser;
use engine_ipc::{InferenceEngine, select_backend, SpeculativeConfig, SpeculativeMetrics};
use rust_embed::RustEmbed;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tower_http::{cors::CorsLayer, trace::TraceLayer, compression::CompressionLayer, services::ServeDir};
use tracing::{info, error, warn};
use toml;

/// Embedded static files (React frontend build output)
#[derive(RustEmbed)]
#[folder = "static/"]
struct StaticAssets;

/// Get the local IP address (non-loopback) for binding to all interfaces
fn get_local_ip() -> Option<IpAddr> {
    // Try to get the local IP by connecting to a public DNS server
    // This doesn't actually send data, just determines the route
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    socket.local_addr().ok().map(|addr| addr.ip())
}

/// Get Tailscale IP if available (100.x.x.x CGNAT range)
fn get_tailscale_ip() -> Option<IpAddr> {
    // Tailscale uses 100.64.0.0/10 (CGNAT range) for its mesh network
    // Try to find a Tailscale interface by checking all local interfaces
    // On Linux, we can check /proc/net/if_inet6 and /proc/net/route
    // But a simpler approach: check if any interface has a 100.x.x.x IP
    use std::net::IpAddr;
    // Try common Tailscale interface patterns
    let test_ips = ["100.64.0.1", "100.64.0.2", "100.64.0.3", "100.64.0.4", "100.64.0.5",
                    "100.64.0.6", "100.64.0.7", "100.64.0.8", "100.64.0.9", "100.64.0.10"];
    for ip_str in test_ips {
        if let Ok(ip) = ip_str.parse::<IpAddr>() {
            // Try to bind to check if this IP exists on the system
            if std::net::UdpSocket::bind(format!("{}:0", ip)).is_ok() {
                return Some(ip);
            }
        }
    }
    None
}

/// Get the best bind address - prefers Tailscale IP, then private IP, falls back to 0.0.0.0
fn get_bind_address() -> String {
    // First check for Tailscale IP (100.64.0.0/10)
    if let Some(tailscale_ip) = get_tailscale_ip() {
        info!("Detected Tailscale IP: {}, binding to 0.0.0.0 for external access", tailscale_ip);
        return "0.0.0.0".to_string();
    }

    if let Some(ip) = get_local_ip() {
        // Check if it's a private IP (RFC 1918)
        match ip {
            IpAddr::V4(v4) => {
                let octets = v4.octets();
                // 10.x.x.x, 172.16-31.x.x, 192.168.x.x
                if octets[0] == 10
                    || (octets[0] == 172 && octets[1] >= 16 && octets[1] <= 31)
                    || (octets[0] == 192 && octets[1] == 168) {
                    info!("Detected private IP: {}, binding to 0.0.0.0 for external access", ip);
                    return "0.0.0.0".to_string();
                }
            }
            IpAddr::V6(v6) => {
                // Check for ULA (fc00::/7) or link-local (fe80::/10)
                let segments = v6.segments();
                if (segments[0] & 0xfe00) == 0xfc00 || (segments[0] & 0xffc0) == 0xfe80 {
                    info!("Detected private IPv6: {}, binding to [::] for external access", ip);
                    return "[::]".to_string();
                }
            }
        }
    }
    // Default to localhost if no private IP detected
    "127.0.0.1".to_string()
}

/// Get the accessible URL for the user to connect to
fn get_accessible_url(port: u16) -> String {
    // Try Tailscale IP first
    if let Some(tailscale_ip) = get_tailscale_ip() {
        return format!("http://{}:{}", tailscale_ip, port);
    }

    // Try local private IP
    if let Some(ip) = get_local_ip() {
        match ip {
            IpAddr::V4(v4) => {
                let octets = v4.octets();
                if octets[0] == 10
                    || (octets[0] == 172 && octets[1] >= 16 && octets[1] <= 31)
                    || (octets[0] == 192 && octets[1] == 168) {
                    return format!("http://{}:{}", ip, port);
                }
            }
            IpAddr::V6(v6) => {
                let segments = v6.segments();
                if (segments[0] & 0xfe00) == 0xfc00 || (segments[0] & 0xffc0) == 0xfe80 {
                    return format!("http://[{}]:{}", ip, port);
                }
            }
        }
    }

    // Fallback to localhost
    format!("http://127.0.0.1:{}", port)
}

// Brain-pack for model downloading
use brain_pack::downloader::HFDownloader;

// New modules
pub mod gpu_detect;
pub mod updater;

// Helper function to recursively find the first model file in a directory
fn find_first_model_file(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    fn is_model_file(path: &std::path::Path) -> bool {
        if let Some(ext) = path.extension() {
            if let Some(ext_str) = ext.to_str() {
                return matches!(ext_str, "safetensors" | "gguf" | "bin" | "pt" | "pth" | "brain");
            }
        }
        false
    }

    fn search_recursively(dir: &std::path::Path) -> Option<std::path::PathBuf> {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && is_model_file(&path) {
                    return Some(path);
                } else if path.is_dir() {
                    if let Some(found) = search_recursively(&path) {
                        return Some(found);
                    }
                }
            }
        }
        None
    }

    search_recursively(dir)
}

// ============================================================================
// Server State
// ============================================================================

#[derive(Clone)]
pub struct ServerState {
    pub engine: Arc<tokio::sync::Mutex<InferenceEngine>>,
    pub openai_api: OpenAiApi,
    pub config: ServerConfig,
    pub download_state: Arc<DownloadState>,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model_dir: PathBuf,
    pub backend: String,
    pub api_key: Option<String>,
    pub enable_cors: bool,
    pub max_request_size: usize,
    // GPU auto-detection
    pub gpu_index: Option<usize>,
    pub gpu_interactive: bool,
    // OTA updates
    pub auto_update: Option<bool>,
    pub auto_install_updates: Option<bool>,
    pub update_check_interval: Option<u64>, // seconds
    // Auto-load model on startup
    pub auto_load: Option<bool>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: get_bind_address(),
            port: 8080,
            model_dir: PathBuf::from("./models"),
            backend: "auto".to_string(),
            api_key: None,
            enable_cors: true,
            max_request_size: 100 * 1024 * 1024, // 100MB
            gpu_index: None,
            gpu_interactive: false,
            auto_update: Some(true),
            auto_install_updates: Some(false),
            update_check_interval: Some(24 * 60 * 60), // 24 hours
            auto_load: Some(true),
        }
    }
}

impl ServerConfig {
    pub fn load_from_file(path: &PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut config: ServerConfig = toml::from_str(&content)?;
        // Expand tilde in model_dir
        if config.model_dir.to_string_lossy().starts_with("~") {
            let expanded = shellexpand::tilde(&config.model_dir.to_string_lossy()).into_owned();
            config.model_dir = PathBuf::from(expanded);
        }
        Ok(config)
    }
}

// Download state for tracking model downloads
use std::collections::HashMap;
use flume::{bounded, Sender};
use tokio::sync::Mutex;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DownloadProgress {
    pub download_id: String,
    pub repo_id: String,
    pub files: Vec<FileProgress>,
    pub total_bytes: u64,
    pub downloaded_bytes: u64,
    pub status: DownloadStatus,
    pub error: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FileProgress {
    pub file_name: String,
    pub bytes_downloaded: u64,
    pub total_bytes: u64,
    pub speed_bps: f64,
    pub finished: bool,
    pub error: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum DownloadStatus {
    Pending,
    Downloading,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Clone)]
pub struct DownloadState {
    progress: Arc<Mutex<HashMap<String, DownloadProgress>>>,
    progress_senders: Arc<Mutex<HashMap<String, Sender<DownloadProgress>>>>,
}

impl DownloadState {
    fn new() -> Self {
        Self {
            progress: Arc::new(Mutex::new(HashMap::new())),
            progress_senders: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    async fn create_progress(&self, download_id: String, repo_id: String) -> Sender<DownloadProgress> {
        let (tx, rx) = bounded(100);
        let mut senders = self.progress_senders.lock().await;
        senders.insert(download_id.clone(), tx.clone());

        // Spawn a task to update progress from the receiver
        let progress = self.progress.clone();
        let download_id_clone = download_id.clone();
        tokio::spawn(async move {
            while let Ok(p) = rx.recv_async().await {
                let mut map = progress.lock().await;
                map.insert(download_id_clone.clone(), p);
            }
        });

        tx
    }

    pub async fn get_progress(&self, download_id: &str) -> Option<DownloadProgress> {
        let map = self.progress.lock().await;
        map.get(download_id).cloned()
    }

    pub async fn get_progress_sender(&self, download_id: &str) -> Option<Sender<DownloadProgress>> {
        let senders = self.progress_senders.lock().await;
        senders.get(download_id).cloned()
    }

    pub async fn remove_download(&self, download_id: &str) {
        let mut map = self.progress.lock().await;
        map.remove(download_id);
        let mut senders = self.progress_senders.lock().await;
        senders.remove(download_id);
    }
}

// ============================================================================
// Routes
// ============================================================================

pub fn create_router(state: ServerState) -> Router {
    let mut app = Router::new()
        // Health & Info
        .route("/health", get(health_check))
        .route("/v1/models", get(list_models))
        .route("/v1/models/:model_id", get(get_model))
        .route("/v1/models/:model_id/load", post(load_model))
        .route("/v1/models/:model_id/unload", post(unload_model))

        // Model Download
        .route("/v1/models/download", post(download_model))
        .route("/v1/models/download/:download_id", get(download_progress))
        .route("/v1/models/download/:download_id/sse", get(download_progress_sse))

        // OpenAI-compatible API
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))

        // Streaming stats (for sharded models)
        .route("/v1/streaming/stats", get(streaming_stats))

        // Speculative Decoding API
        .route("/v1/speculative/config", get(speculative_config))
        .route("/v1/speculative/config", post(update_speculative_config))
        .route("/v1/speculative/metrics", get(speculative_metrics))
        .route("/v1/speculative/metrics/sse", get(speculative_metrics_sse))

        // GPU Detection API
        .route("/v1/system/gpus", get(list_gpus))
        .route("/v1/system/gpus/detect", post(detect_gpus))

        // OTA Update API
        .route("/v1/system/update/check", get(check_updates))
        .route("/v1/system/update/install", post(install_update))
        .route("/v1/system/update/progress", get(update_progress))

        // System Config API
        .route("/v1/system/config", get(get_system_config))
        .route("/v1/system/config", post(update_system_config))

        // WebSocket for streaming
        .route("/v1/ws/chat", get(ws_chat_handler))

        // Serve React SPA from embedded assets (catch-all route for client-side routing)
        .route("/assets/*path", get(serve_embedded_asset))
        .route("/favicon.svg", get(serve_favicon))
        .route("/robots.txt", get(serve_robots))
        .route("/manifest.json", get(serve_manifest))
        .fallback(get(serve_spa_fallback));

    // Add middleware
    app = app
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new());

    if state.config.enable_cors {
        app = app.layer(CorsLayer::permissive());
    }

    app.with_state(state)
}

// ============================================================================
// Health Check
// ============================================================================

#[axum::debug_handler]
async fn health_check(State(state): State<ServerState>) -> impl IntoResponse {
    let engine = state.engine.lock().await;
    let models = engine.list_models();
    Json(serde_json::json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
        "models_loaded": models.len(),
        "backend": models.first().map(|m| m.backend.as_str()).unwrap_or("none"),
    }))
}

#[axum::debug_handler]
async fn list_models(State(state): State<ServerState>) -> impl IntoResponse {
    let engine = state.engine.lock().await;
    let models = engine.list_models();
    let data: Vec<api_openai::OpenAiModelInfo> = models.iter()
        .map(api_openai::OpenAiModelInfo::from_engine)
        .collect();
    Json(serde_json::json!({
        "object": "list",
        "data": data
    }))
}

async fn get_model(State(state): State<ServerState>, Path(model_id): Path<String>) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let engine = state.engine.lock().await;
    let models = engine.list_models();
    models.iter()
        .find(|m| m.id == model_id)
        .map(|m| Json(api_openai::OpenAiModelInfo::from_engine(m)))
        .ok_or_else(|| (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "Model not found"}))))
}

async fn load_model(
    State(state): State<ServerState>,
    Path(_model_id): Path<String>,
    Json(payload): Json<serde_json::Value>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let filename = payload.get("filename")
        .and_then(|v| v.as_str())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "filename required"}))))?;

    let mut engine = state.engine.lock().await;
    match engine.load_model(filename).await {
        Ok(id) => {
            let models = engine.list_models();
            let info = models.iter().find(|m| m.id == id).cloned().unwrap();
            Ok(Json(serde_json::json!({"model_id": id, "info": api_openai::OpenAiModelInfo::from_engine(&info)})))
        }
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))))
    }
}

async fn unload_model(
    State(state): State<ServerState>,
    Path(model_id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let mut engine = state.engine.lock().await;
    match engine.unload_model(&model_id) {
        Ok(_) => Ok(Json(serde_json::json!({"model_id": model_id, "status": "unloaded"}))),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))))
    }
}

// ============================================================================
// OpenAI Compatible Endpoints
// ============================================================================

#[axum::debug_handler]
async fn chat_completions(
    State(state): State<ServerState>,
    headers: HeaderMap,
    Json(request): Json<api_openai::ChatCompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    // Verify API key
    if let Some(api_key) = &state.config.api_key {
        let auth = headers.get("authorization")
            .and_then(|h| h.to_str().ok())
            .and_then(|h| h.strip_prefix("Bearer "));
        if auth != Some(api_key) {
            return Err((StatusCode::UNAUTHORIZED, Json(serde_json::json!({
                "error": {"message": "Invalid API key", "type": "authentication_error"}
            }))));
        }
    }

    state.openai_api.chat_completions(request).await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))))
}

#[axum::debug_handler]
async fn completions(
    State(state): State<ServerState>,
    headers: HeaderMap,
    Json(request): Json<api_openai::CompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    if let Some(api_key) = &state.config.api_key {
        let auth = headers.get("authorization")
            .and_then(|h| h.to_str().ok())
            .and_then(|h| h.strip_prefix("Bearer "));
        if auth != Some(api_key) {
            return Err((StatusCode::UNAUTHORIZED, Json(serde_json::json!({
                "error": {"message": "Invalid API key", "type": "authentication_error"}
            }))));
        }
    }

    state.openai_api.completions(request).await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))))
}

#[axum::debug_handler]
async fn embeddings(
    State(state): State<ServerState>,
    headers: HeaderMap,
    Json(request): Json<api_openai::EmbeddingRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    if let Some(api_key) = &state.config.api_key {
        let auth = headers.get("authorization")
            .and_then(|h| h.to_str().ok())
            .and_then(|h| h.strip_prefix("Bearer "));
        if auth != Some(api_key) {
            return Err((StatusCode::UNAUTHORIZED, Json(serde_json::json!({
                "error": {"message": "Invalid API key", "type": "authentication_error"}
            }))));
        }
    }

    state.openai_api.embeddings(request).await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))))
}

#[axum::debug_handler]
async fn streaming_stats(
    State(state): State<ServerState>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let engine = state.engine.lock().await;
    match engine.streaming_stats().await {
        Some(stats) => Ok(Json(stats)),
        None => Err((StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "No sharded model loaded"}))))
    }
}

// ============================================================================
// Model Download Handlers
// ============================================================================

#[derive(serde::Deserialize)]
struct DownloadRequest {
    repo_id: String,
    revision: Option<String>,
    files: Option<Vec<String>>,
    token: Option<String>,
}

#[axum::debug_handler]
async fn download_model(
    State(state): State<ServerState>,
    Json(request): Json<DownloadRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let download_id = uuid::Uuid::new_v4().to_string();
    let download_id_for_spawn = download_id.clone(); // Clone for the spawned task
    let download_id_for_response = download_id.clone(); // Clone for the response

    // Create progress tracking channel
    let progress_tx = state.download_state.create_progress(download_id.clone(), request.repo_id.clone()).await;

    // Spawn download task using brain-pack's downloader
    let model_dir = state.config.model_dir.clone();
    let repo_id = request.repo_id.clone();
    let revision = request.revision.unwrap_or_else(|| "main".to_string());
    let files = request.files.unwrap_or_default();
    let token = request.token.clone();

    tokio::spawn(async move {
        // Use brain-pack's downloader
        let config = brain_pack::downloader::DownloadConfig {
            repo_id: repo_id.clone(),
            revision: revision.clone(),
            files: files.clone(),
            output_dir: model_dir.join(&repo_id),
            token,
            ..Default::default()
        };

        let downloader = brain_pack::downloader::HFDownloader::new(config);
        let progress_rx = downloader.progress_receiver().clone();

        // Update progress from downloader
        let mut file_progress_map: std::collections::HashMap<String, brain_pack::downloader::DownloadProgress> = std::collections::HashMap::new();

        // Run download and process progress concurrently
        let download_handle = tokio::spawn(async move {
            downloader.download_all().await
        });

        while let Ok(progress) = progress_rx.recv_async().await {
            file_progress_map.insert(progress.file_name.clone(), progress.clone());

            // Convert to server progress format
            let files: Vec<FileProgress> = file_progress_map.values().map(|p| FileProgress {
                file_name: p.file_name.clone(),
                bytes_downloaded: p.bytes_downloaded,
                total_bytes: p.total_bytes,
                speed_bps: p.speed_bps,
                finished: p.finished,
                error: p.error.clone(),
            }).collect();

            let total_bytes: u64 = file_progress_map.values().map(|p| p.total_bytes).sum();
            let downloaded_bytes: u64 = file_progress_map.values().map(|p| p.bytes_downloaded).sum();
            let all_finished = file_progress_map.values().all(|p| p.finished);
            let any_error = file_progress_map.values().any(|p| p.error.is_some());
            let error_msg = file_progress_map.values().find_map(|p| p.error.clone());

            let server_progress = DownloadProgress {
                download_id: download_id_for_spawn.clone(),
                repo_id: repo_id.clone(),
                files,
                total_bytes,
                downloaded_bytes,
                status: if any_error {
                    DownloadStatus::Failed
                } else if all_finished && !file_progress_map.is_empty() {
                    DownloadStatus::Completed
                } else {
                    DownloadStatus::Downloading
                },
                error: error_msg,
            };

            let _ = progress_tx.send_async(server_progress).await;
        }

        // Wait for download to complete
        if let Err(e) = download_handle.await.unwrap_or(Err(anyhow::anyhow!("Download task panicked"))) {
            eprintln!("Download failed: {}", e);
            let _ = progress_tx.send_async(DownloadProgress {
                download_id: download_id_for_spawn.clone(),
                repo_id: repo_id.clone(),
                files: Vec::new(),
                total_bytes: 0,
                downloaded_bytes: 0,
                status: DownloadStatus::Failed,
                error: Some(e.to_string()),
            }).await;
        } else {
            // Send final completion
            let _ = progress_tx.send_async(DownloadProgress {
                download_id: download_id_for_spawn,
                repo_id: repo_id.clone(),
                files: Vec::new(),
                total_bytes: 0,
                downloaded_bytes: 0,
                status: DownloadStatus::Completed,
                error: None,
            }).await;
        }
    });

    Ok(Json(serde_json::json!({
        "download_id": download_id_for_response,
        "status": "started"
    })))
}

#[axum::debug_handler]
async fn download_progress(
    State(state): State<ServerState>,
    Path(download_id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    match state.download_state.get_progress(&download_id).await {
        Some(progress) => Ok(Json(progress)),
        None => Err((StatusCode::NOT_FOUND, Json(serde_json::json!({
            "error": "Download not found"
        })))),
    }
}

#[axum::debug_handler]
async fn download_progress_sse(
    State(state): State<ServerState>,
    Path(download_id): Path<String>,
) -> impl IntoResponse {
    use axum::response::sse::{Event, KeepAlive, Sse};
    use std::time::Duration;

    let download_state = state.download_state.clone();
    let download_id_clone = download_id.clone();

    let stream = async_stream::stream! {
        // Send initial progress if available
        if let Some(progress) = download_state.get_progress(&download_id_clone).await {
            yield Ok::<Event, std::convert::Infallible>(Event::default().data(serde_json::to_string(&progress).unwrap()));
        }

        // Wait for updates
        let mut interval = tokio::time::interval(Duration::from_millis(500));
        loop {
            interval.tick().await;
            if let Some(progress) = download_state.get_progress(&download_id_clone).await {
                yield Ok::<Event, std::convert::Infallible>(Event::default().data(serde_json::to_string(&progress).unwrap()));

                if progress.status == DownloadStatus::Completed
                    || progress.status == DownloadStatus::Failed
                    || progress.status == DownloadStatus::Cancelled {
                    break;
                }
            }
        }

        // Keep connection alive for a bit more
        loop {
            tokio::time::sleep(Duration::from_secs(30)).await;
            yield Ok::<Event, std::convert::Infallible>(Event::default().event("ping").data(""));
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(30)))
}

// ============================================================================
// WebSocket Streaming
// ============================================================================

async fn ws_chat_handler(
    ws: WebSocketUpgrade,
    State(state): State<ServerState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_ws_chat(socket, state))
}

async fn handle_ws_chat(
    mut socket: axum::extract::ws::WebSocket,
    state: ServerState,
) {
    use axum::extract::ws::Message;
    use futures_util::{SinkExt, StreamExt};

    while let Some(msg) = socket.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                if let Ok(request) = serde_json::from_str::<api_openai::ChatCompletionRequest>(&text) {
                    match state.openai_api.chat_completions_stream(request).await {
                        Ok(mut stream) => {
                            while let Some(chunk) = stream.next().await {
                                if let Ok(chunk) = chunk {
                                    if let Ok(data) = serde_json::to_string(&chunk) {
                                        if socket.send(Message::Text(format!("data: {}\n\n", data))).await.is_err() {
                                            break;
                                        }
                                    }
                                }
                            }
                            let _ = socket.send(Message::Text("data: [DONE]\n\n".to_string())).await;
                        }
                        Err(e) => {
                            let _ = socket.send(Message::Text(format!("error: {}", e))).await;
                        }
                    }
                }
            }
            Ok(Message::Close(_)) => break,
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }
}

// ============================================================================
// SPA Fallback - serves embedded React assets
// ============================================================================

async fn serve_spa_fallback() -> Response {
    // Serve embedded index.html for SPA client-side routing
    match StaticAssets::get("index.html") {
        Some(content) => Response::builder()
            .header("Content-Type", "text/html")
            .body(axum::body::Body::from(content.data.as_ref().to_vec()))
            .unwrap(),
        None => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(axum::body::Body::from(b"Frontend not built. Run 'npm run build' in frontend-react/ and rebuild server.".to_vec()))
            .unwrap(),
    }
}

// ============================================================================
// Embedded Static Asset Handler
// ============================================================================

async fn serve_embedded_asset(Path(path): Path<String>) -> impl IntoResponse {
    // Handle /assets/* requests from embedded files
    // The path parameter captures only the part after "/assets/", so we need to prepend "assets/"
    // Also handle root-level files like favicon.svg, robots.txt, manifest.json (these come from their own dedicated routes)
    let asset_path = format!("assets/{}", path);
    match StaticAssets::get(&asset_path) {
        Some(content) => {
            let mime_type = mime_guess::from_path(&asset_path).first_or_octet_stream();
            Response::builder()
                .header("Content-Type", mime_type.as_ref())
                .header("Cache-Control", "public, max-age=31536000, immutable")
                .body(axum::body::Body::from(content.data.as_ref().to_vec()))
                .unwrap()
        }
        None => {
            // Fall through to SPA fallback for client-side routing
            serve_spa_fallback().await
        }
    }
}

async fn serve_favicon() -> impl IntoResponse {
    match StaticAssets::get("favicon.svg") {
        Some(content) => {
            let mime_type = mime_guess::from_path("favicon.svg").first_or_octet_stream();
            Response::builder()
                .header("Content-Type", mime_type.as_ref())
                .header("Cache-Control", "public, max-age=31536000, immutable")
                .body(axum::body::Body::from(content.data.as_ref().to_vec()))
                .unwrap()
        }
        None => serve_spa_fallback().await
    }
}

async fn serve_robots() -> impl IntoResponse {
    match StaticAssets::get("robots.txt") {
        Some(content) => {
            let mime_type = mime_guess::from_path("robots.txt").first_or_octet_stream();
            Response::builder()
                .header("Content-Type", mime_type.as_ref())
                .header("Cache-Control", "public, max-age=31536000, immutable")
                .body(axum::body::Body::from(content.data.as_ref().to_vec()))
                .unwrap()
        }
        None => serve_spa_fallback().await
    }
}

async fn serve_manifest() -> impl IntoResponse {
    match StaticAssets::get("manifest.json") {
        Some(content) => {
            let mime_type = mime_guess::from_path("manifest.json").first_or_octet_stream();
            Response::builder()
                .header("Content-Type", mime_type.as_ref())
                .header("Cache-Control", "public, max-age=31536000, immutable")
                .body(axum::body::Body::from(content.data.as_ref().to_vec()))
                .unwrap()
        }
        None => serve_spa_fallback().await
    }
}

// ============================================================================
// Server Builder
// ============================================================================

pub struct ServerBuilder {
    config: ServerConfig,
}

impl ServerBuilder {
    pub fn new() -> Self {
        Self { config: ServerConfig::default() }
    }

    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.config.host = host.into();
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.config.port = port;
        self
    }

    pub fn model_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.model_dir = dir.into();
        self
    }

    pub fn backend(mut self, backend: impl Into<String>) -> Self {
        self.config.backend = backend.into();
        self
    }

    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        let key = key.into();
        if !key.is_empty() {
            self.config.api_key = Some(key);
        }
        self
    }

    pub fn cors(mut self, enable: bool) -> Self {
        self.config.enable_cors = enable;
        self
    }

    pub async fn build(self) -> Result<Server> {
        let backend = select_backend(&self.config.backend)?;
        let engine = Arc::new(tokio::sync::Mutex::new(InferenceEngine::new(&self.config.model_dir, backend)?));
        let openai_api = OpenAiApi::new(engine.clone());

        let state = ServerState {
            engine,
            openai_api,
            config: self.config.clone(),
            download_state: Arc::new(DownloadState::new()),
        };

        let app = create_router(state.clone());
        // Resolve "auto" host to actual bind address
        let resolved_host = if self.config.host == "auto" {
            get_bind_address()
        } else {
            self.config.host.clone()
        };
        let addr = format!("{}:{}", resolved_host, self.config.port).parse::<SocketAddr>()?;
        let port = self.config.port;

        Ok(Server { app, addr, state, port })
    }
}

pub struct Server {
    app: Router,
    addr: SocketAddr,
    state: ServerState,
    port: u16,
}

impl Server {
    pub async fn run(self) -> Result<()> {
        let listener = tokio::net::TcpListener::bind(self.addr).await?;

        // Print accessible URL for the user
        let accessible_url = get_accessible_url(self.port);
        info!("Server listening on http://{}", self.addr);
        info!("Web UI accessible at: {}", accessible_url);
        if self.addr.ip().is_unspecified() {
            info!("(Bound to all interfaces - accessible from other machines on the network)");
        }

        axum::serve(listener, self.app)
            .with_graceful_shutdown(shutdown_signal())
            .await?;

        info!("Server shut down gracefully");
        Ok(())
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c().await.expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("Shutdown signal received");
}

// ============================================================================
// Speculative Decoding API Endpoints
// ============================================================================

#[axum::debug_handler]
async fn speculative_config(
    State(state): State<ServerState>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let engine = state.engine.lock().await;
    let config = engine.speculative_config().await;
    Ok(Json(serde_json::json!({
        "enabled": config.enabled,
        "max_draft_tokens": config.max_draft_tokens,
        "confidence_threshold": config.confidence_threshold,
        "ngram_order": config.ngram_order,
        "max_ngram_entries": config.max_ngram_entries,
        "draft_temperature": config.draft_temperature,
        "draft_top_k": config.draft_top_k,
        "draft_top_p": config.draft_top_p,
        "max_ngram_context": config.max_ngram_context,
        "verification_threshold": config.verification_threshold,
    })))
}

#[axum::debug_handler]
async fn update_speculative_config(
    State(state): State<ServerState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let config = SpeculativeConfig {
        enabled: payload.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true),
        max_draft_tokens: payload.get("max_draft_tokens").and_then(|v| v.as_u64()).unwrap_or(8) as usize,
        confidence_threshold: payload.get("confidence_threshold").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32,
        ngram_order: payload.get("ngram_order").and_then(|v| v.as_u64()).unwrap_or(4) as usize,
        max_ngram_entries: payload.get("max_ngram_entries").and_then(|v| v.as_u64()).unwrap_or(4_000_000) as usize,
        draft_temperature: payload.get("draft_temperature").and_then(|v| v.as_f64()).unwrap_or(0.1) as f32,
        draft_top_k: payload.get("draft_top_k").and_then(|v| v.as_u64()).unwrap_or(50) as usize,
        draft_top_p: payload.get("draft_top_p").and_then(|v| v.as_f64()).unwrap_or(0.9) as f32,
        max_ngram_context: payload.get("max_ngram_context").and_then(|v| v.as_u64()).unwrap_or(256) as usize,
        verification_threshold: payload.get("verification_threshold").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32,
    };

    if let Err(e) = config.validate() {
        return Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": e.to_string()}))));
    }

    // Need mutable reference to engine
    let mut engine = state.engine.lock().await;
    engine.set_speculative_config(config).await;

    Ok(Json(serde_json::json!({"status": "updated"})))
}

#[axum::debug_handler]
async fn speculative_metrics(
    State(state): State<ServerState>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let engine = state.engine.lock().await;
    let metrics = engine.speculative_metrics().await;
    if let Some(metrics) = metrics {
        Ok(Json(serde_json::json!({
            "tokens_indexed": metrics.tokens_indexed,
            "unique_ngrams": metrics.unique_ngrams,
            "config": {
                "enabled": metrics.config.enabled,
                "max_draft_tokens": metrics.config.max_draft_tokens,
                "confidence_threshold": metrics.config.confidence_threshold,
            }
        })))
    } else {
        Err((StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": "Speculative decoding not initialized"}))))
    }
}

#[axum::debug_handler]
async fn speculative_metrics_sse(
    State(state): State<ServerState>,
) -> impl IntoResponse {
    use axum::response::sse::{Event, KeepAlive, Sse};
    use std::time::Duration;
    use async_stream::stream;

    let engine = state.engine.clone();

    let stream = stream! {
        // Send initial metrics
        {
            let engine = engine.lock().await;
            if let Some(metrics) = engine.speculative_metrics().await {
                yield Ok::<Event, std::convert::Infallible>(Event::default().data(serde_json::to_string(&serde_json::json!({
                    "tokens_indexed": metrics.tokens_indexed,
                    "unique_ngrams": metrics.unique_ngrams,
                    "config": {
                        "enabled": metrics.config.enabled,
                        "max_draft_tokens": metrics.config.max_draft_tokens,
                        "confidence_threshold": metrics.config.confidence_threshold,
                    }
                })).unwrap()));
            }
        }

        // Poll for updates every 500ms
        let mut interval = tokio::time::interval(Duration::from_millis(500));
        loop {
            interval.tick().await;
            let engine = engine.lock().await;
            if let Some(metrics) = engine.speculative_metrics().await {
                yield Ok::<Event, std::convert::Infallible>(Event::default().data(serde_json::to_string(&serde_json::json!({
                    "tokens_indexed": metrics.tokens_indexed,
                    "unique_ngrams": metrics.unique_ngrams,
                    "config": {
                        "enabled": metrics.config.enabled,
                        "max_draft_tokens": metrics.config.max_draft_tokens,
                        "confidence_threshold": metrics.config.confidence_threshold,
                    }
                })).unwrap()));
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(30)))
}

// ============================================================================
// GPU Detection API
// ============================================================================

#[axum::debug_handler]
async fn list_gpus() -> impl IntoResponse {
    match crate::gpu_detect::get_gpu_info() {
        Ok(gpus) => Json(serde_json::json!({
            "gpus": gpus,
            "count": gpus.len()
        })),
        Err(e) => Json(serde_json::json!({
            "error": e.to_string(),
            "gpus": [],
            "count": 0
        }))
    }
}

#[axum::debug_handler]
async fn detect_gpus(
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    let preferred_backend = payload.get("preferred_backend").and_then(|v| v.as_str());
    let interactive = payload.get("interactive").and_then(|v| v.as_bool()).unwrap_or(false);

    match crate::gpu_detect::detect_and_select_gpu(preferred_backend, interactive) {
        Ok(result) => Json(serde_json::json!({
            "selected_gpu": result.selected_gpu,
            "backend": result.backend,
            "selection_method": result.selection_method,
            "gpus": result.gpus
        })),
        Err(e) => Json(serde_json::json!({
            "error": e.to_string()
        }))
    }
}

// ============================================================================
// OTA Update API
// ============================================================================

#[axum::debug_handler]
async fn check_updates(
    State(state): State<ServerState>,
) -> impl IntoResponse {
    let include_prerelease = state.config.auto_update.unwrap_or(false);

    match crate::updater::check_for_updates(include_prerelease).await {
        Ok(info) => Json(serde_json::json!({
            "current_version": info.current_version,
            "latest_version": info.latest_version,
            "update_available": info.update_available,
            "release_notes": info.release_notes,
            "download_url": info.download_url,
            "asset_name": info.asset_name,
            "asset_size": info.asset_size
        })),
        Err(e) => Json(serde_json::json!({
            "error": e.to_string()
        }))
    }
}

#[axum::debug_handler]
async fn install_update(
    State(state): State<ServerState>,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    let update_info = match crate::updater::check_for_updates(false).await {
        Ok(info) => info,
        Err(e) => return Json(serde_json::json!({"error": e.to_string()})),
    };

    if !update_info.update_available {
        return Json(serde_json::json!({"error": "No update available"}));
    }

    let restart_after = payload.get("restart_after").and_then(|v| v.as_bool()).unwrap_or(true);

    match crate::updater::download_and_install_update(&update_info, None).await {
        Ok(_) => Json(serde_json::json!({
            "status": "success",
            "message": "Update installed successfully",
            "restart_required": restart_after
        })),
        Err(e) => Json(serde_json::json!({
            "error": e.to_string()
        }))
    }
}

#[axum::debug_handler]
async fn update_progress() -> impl IntoResponse {
    // TODO: Implement progress tracking via SSE
    Json(serde_json::json!({
        "status": "not_implemented"
    }))
}

// ============================================================================
// System Config API
// ============================================================================

#[axum::debug_handler]
async fn get_system_config(
    State(state): State<ServerState>,
) -> impl IntoResponse {
    let config = &state.config;
    Json(serde_json::json!({
        "config": {
            "host": config.host,
            "port": config.port,
            "model_dir": config.model_dir.to_string_lossy(),
            "backend": config.backend,
            "api_key": config.api_key.as_deref().unwrap_or(""),
            "enable_cors": config.enable_cors,
            "max_request_size": config.max_request_size,
            "gpu_index": config.gpu_index,
            "gpu_interactive": config.gpu_interactive,
            "auto_update": config.auto_update,
            "auto_install_updates": config.auto_install_updates,
            "update_check_interval": config.update_check_interval,
            "auto_load": config.auto_load,
        }
    }))
}

#[axum::debug_handler]
async fn update_system_config(
    State(state): State<ServerState>,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    // For now, we'll just return the updated config
    // In a real implementation, this would write to the config file
    Json(serde_json::json!({
        "status": "success",
        "message": "Configuration updated (requires restart to take effect)",
        "config": payload
    }))
}

#[derive(clap::Parser)]
#[command(name = "decoupled-ai-server", version, about = "DeCoupled-AI Web Server", disable_help_flag = true)]
struct Cli {
    #[arg(long, default_value = "auto", help = "Bind address (auto = detect private IP and bind to 0.0.0.0)")]
    host: String,

    #[arg(short, long, default_value = "8080")]
    port: u16,

    #[arg(long, default_value = "./models")]
    model_dir: PathBuf,

    #[arg(long, default_value = "auto")]
    backend: String,

    #[arg(long)]
    api_key: Option<String>,

    #[arg(long)]
    no_cors: bool,

    // Config file
    #[arg(long, help = "Path to config.toml file")]
    config: Option<PathBuf>,

    // GPU auto-detection
    #[arg(long, help = "GPU index to use (auto-detected if not specified)")]
    gpu_index: Option<usize>,

    #[arg(long, help = "Interactively select GPU when multiple are available")]
    gpu_interactive: bool,

    // OTA updates
    #[arg(long, help = "Enable automatic update checks", default_value = "true")]
    auto_update: bool,

    #[arg(long, help = "Automatically install updates when available")]
    auto_install_updates: bool,

    #[arg(long, help = "Check for updates on startup")]
    check_updates: bool,

    // Auto-load models
    #[arg(long, help = "Auto-load first available model on startup")]
    auto_load: bool,

    #[arg(short, long, action = clap::ArgAction::Help)]
    help: Option<bool>,
}

pub async fn run_cli() -> Result<()> {
    let cli = Cli::parse();

    // Load config file if provided
    let file_config = if let Some(config_path) = &cli.config {
        info!("Loading config from: {}", config_path.display());
        Some(ServerConfig::load_from_file(config_path)?)
    } else {
        None
    };

    // Handle GPU auto-detection if backend is "auto"
    let (backend, gpu_index) = if cli.backend == "auto" {
        // If config file specifies a backend, use that instead of auto-detect
        let config_backend = file_config.as_ref().and_then(|c| {
            if c.backend != "auto" {
                Some(c.backend.clone())
            } else {
                None
            }
        });

        if let Some(backend_str) = config_backend {
            // Use backend from config file
            let gpu_idx = cli.gpu_index.or(file_config.as_ref().and_then(|c| c.gpu_index));
            (backend_str, gpu_idx)
        } else {
            // Auto-detect
            use crate::gpu_detect::{detect_and_select_gpu, GpuDetectionResult, GpuBackend};

            let preferred_backend = if cli.gpu_index.is_some() { None } else { Some("cuda") };
            let detection_result = detect_and_select_gpu(preferred_backend, cli.gpu_interactive)?;

            info!("GPU Detection: {} GPU(s) found, backend: {}", detection_result.gpus.len(), detection_result.backend);
            for (i, gpu) in detection_result.gpus.iter().enumerate() {
                let vram = gpu.vram_mb.map(|v| format!("{} MB", v)).unwrap_or_else(|| "Shared".to_string());
                info!("  GPU {}: {} ({}) - {}", i, gpu.name, gpu.vendor.as_str(), vram);
            }

            let selected_index = cli.gpu_index.or_else(|| {
                detection_result.selected_gpu.as_ref().map(|gpu| {
                    detection_result.gpus.iter().position(|g| g.name == gpu.name && g.vendor == gpu.vendor).unwrap_or(0)
                })
            });
            if let Some(idx) = selected_index {
                info!("Selected GPU {}: {}", idx, detection_result.gpus[idx].name);
            } else {
                info!("Using CPU backend");
            }

            // Convert GpuBackend enum to string
            // Note: Engine only supports cuda, rocm, metal, cpu. OpenCL falls back to CPU.
            let backend_str = match detection_result.backend {
                GpuBackend::Cuda => "cuda",
                GpuBackend::Rocm => "rocm",
                GpuBackend::Metal => "metal",
                GpuBackend::OpenCl => "cpu",  // No OpenCL backend in engine yet
                GpuBackend::Cpu => "cpu",
                GpuBackend::Unknown => "cpu",
            };
            (backend_str.to_string(), selected_index)
        }
    } else {
        (cli.backend.clone(), cli.gpu_index)
    };

    let mut builder = ServerBuilder::new()
        .host(cli.host.clone())
        .port(cli.port)
        .model_dir(cli.model_dir.clone())
        .backend(backend)
        .api_key(cli.api_key.clone().unwrap_or_default())
        .cors(!cli.no_cors);

    // Merge config file values (CLI overrides config file)
    if let Some(fc) = file_config {
        // Apply config file values that weren't overridden by CLI
        let host_default = "auto".to_string();
        let port_default = 8080;
        let model_dir_default = PathBuf::from("./models");
        let backend_default = "auto".to_string();

        if cli.host == host_default {
            builder.config.host = fc.host;
        }
        if cli.port == port_default {
            builder.config.port = fc.port;
        }
        if cli.model_dir == model_dir_default {
            builder.config.model_dir = fc.model_dir;
        }
        if cli.backend == backend_default {
            builder.config.backend = fc.backend;
        }
        builder.config.api_key = builder.config.api_key.or(fc.api_key);
        builder.config.enable_cors = fc.enable_cors;
        builder.config.max_request_size = fc.max_request_size;
        builder.config.gpu_index = builder.config.gpu_index.or(fc.gpu_index);
        builder.config.gpu_interactive = fc.gpu_interactive;
        builder.config.auto_update = fc.auto_update;
        builder.config.auto_install_updates = fc.auto_install_updates;
        builder.config.update_check_interval = fc.update_check_interval;
        builder.config.auto_load = fc.auto_load;
    }

    // Apply CLI overrides for values that were explicitly set
    if let Some(idx) = gpu_index {
        builder.config.gpu_index = Some(idx);
    }
    builder.config.gpu_interactive = cli.gpu_interactive;
    builder.config.auto_update = Some(cli.auto_update);
    builder.config.auto_install_updates = Some(cli.auto_install_updates);
    builder.config.auto_load = Some(cli.auto_load);

    let server = builder.build().await?;

    // Auto-load model on startup if enabled
    if server.state.config.auto_load.unwrap_or(true) {
        info!("Auto-loading first available model from: {}", server.state.config.model_dir.display());
        let mut engine = server.state.engine.lock().await;

        // Find first model file in the directory (recursively search subdirectories)
        let model_dir = &server.state.config.model_dir;
        if let Some(model_path) = find_first_model_file(model_dir) {
            let filename = model_path.strip_prefix(model_dir).unwrap_or(&model_path).to_string_lossy();
            info!("Found model file: {}", filename);
            if let Err(e) = engine.load_model(&filename).await {
                warn!("Auto-load model failed: {}", e);
            } else {
                info!("Model auto-loaded successfully: {}", filename);
            }
        } else {
            info!("No model files found in {:?}", model_dir);
        }
    }

    // Check for updates on startup if enabled
    if cli.check_updates || cli.auto_update {
        info!("Checking for updates...");
        match crate::updater::check_for_updates(false).await {
            Ok(update_info) if update_info.update_available => {
                info!("Update available: {} -> {}", update_info.current_version, update_info.latest_version);
                if let Some(notes) = &update_info.release_notes {
                    info!("Release notes: {}", notes.lines().take(3).collect::<Vec<_>>().join(" | "));
                }
                if cli.auto_install_updates {
                    info!("Auto-installing update...");
                    if let Err(e) = crate::updater::download_and_install_update(&update_info, None).await {
                        error!("Auto-update failed: {}", e);
                    } else {
                        info!("Update installed successfully. Please restart the server.");
                        return Ok(());
                    }
                }
            }
            Ok(_) => {
                info!("Already up to date ({})", env!("CARGO_PKG_VERSION"));
            }
            Err(e) => {
                warn!("Update check failed: {}", e);
            }
        }
    }

    // Start background update checker if enabled
    if cli.auto_update {
        let config = server.state.config.clone();
        crate::updater::start_update_checker(config);
    }

    server.run().await
}

fn is_default_backend(backend: &str) -> bool {
    backend == "auto"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_server_builder() {
        let server = ServerBuilder::new()
            .port(0) // Random port
            .backend("cpu")
            .build()
            .await;
        assert!(server.is_ok());
    }
}