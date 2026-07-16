//! Embedded Frontend Assets
//!
//! Single-page application for the DeCoupled-AI dashboard.
//! Includes chat interface, model manager, and settings.

use std::collections::HashMap;

/// Frontend asset container
#[derive(Clone)]
pub struct FrontendAssets {
    assets: HashMap<String, Vec<u8>>,
}

impl FrontendAssets {
    pub fn new() -> Self {
        let mut assets = HashMap::new();

        // Main CSS
        assets.insert("style.css".to_string(), include_bytes!("../assets/style.css").to_vec());
        // Main JS
        assets.insert("app.js".to_string(), include_bytes!("../assets/app.js").to_vec());
        // Chart.js for visualizations
        assets.insert("chart.min.js".to_string(), include_bytes!("../assets/chart.min.js").to_vec());
        // Fonts
        assets.insert("inter-var.woff2".to_string(), include_bytes!("../assets/inter-var.woff2").to_vec());

        Self { assets }
    }

    pub fn get(&self, path: &str) -> Option<&[u8]> {
        self.assets.get(path).map(|v| v.as_slice())
    }

    // ========================================================================
    // HTML Pages (inline for zero dependencies)
    // ========================================================================

    pub fn index_html() -> &'static str {
        r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeCoupled-AI Dashboard</title>
    <link rel="stylesheet" href="/assets/style.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
</head>
<body>
    <div id="app">
        <header class="header">
            <div class="header-left">
                <svg class="logo" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect width="32" height="32" rx="8" fill="url(#grad)"/>
                    <path d="M8 16L14 22L24 10" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
                    <defs>
                        <linearGradient id="grad" x1="0" y1="0" x2="32" y2="32">
                            <stop offset="0%" stop-color="#6366f1"/>
                            <stop offset="100%" stop-color="#8b5cf6"/>
                        </linearGradient>
                    </defs>
                </svg>
                <span class="title">DeCoupled-AI</span>
            </div>
            <nav class="nav">
                <a href="/chat" class="nav-item active" data-page="chat">Chat</a>
                <a href="/models" class="nav-item" data-page="models">Models</a>
                <a href="/speculative" class="nav-item" data-page="speculative">Speculative</a>
                <a href="/download" class="nav-item" data-page="download">Download</a>
                <a href="/settings" class="nav-item" data-page="settings">Settings</a>
            </nav>
            <div class="header-right">
                <div class="status-indicator" id="statusIndicator">
                    <span class="dot"></span>
                    <span class="text">Connecting...</span>
                </div>
            </div>
        </header>
        <main class="main" id="mainContent">
            <!-- Page content loaded dynamically -->
        </main>
    </div>
    <script src="/assets/app.js"></script>
    <script>initApp();</script>
</body>
</html>"##
    }

    pub fn chat_html() -> &'static str {
        r#"<div class="chat-page">
    <div class="chat-sidebar" id="chatSidebar">
        <div class="sidebar-header">
            <h2>Conversations</h2>
            <button class="btn btn-primary btn-sm" id="newChatBtn">New Chat</button>
        </div>
        <div class="conversation-list" id="conversationList">
            <div class="conversation-item active" data-id="current">
                <span class="conv-title">New Conversation</span>
                <span class="conv-time">Just now</span>
            </div>
        </div>
    </div>
    <div class="chat-main">
        <div class="chat-header">
            <div class="model-selector">
                <label>Model:</label>
                <select id="modelSelect" class="select">
                    <option value="">Select a model...</option>
                </select>
            </div>
            <div class="chat-actions">
                <button class="btn btn-ghost" id="clearChatBtn" title="Clear chat">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                </button>
            </div>
        </div>
        <div class="chat-messages" id="chatMessages">
            <div class="welcome-message">
                <svg class="welcome-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>
                <h3>Welcome to DeCoupled-AI</h3>
                <p>Select a model and start chatting. Your conversation runs entirely locally.</p>
                <div class="example-prompts">
                    <button class="example-prompt" data-prompt="Explain quantum computing in simple terms">Quantum Computing</button>
                    <button class="example-prompt" data-prompt="Write a Rust function for binary search">Rust Binary Search</button>
                    <button class="example-prompt" data-prompt="Create a Dockerfile for a Node.js app">Dockerfile for Node.js</button>
                    <button class="example-prompt" data-prompt="Explain the transformer architecture">Transformer Architecture</button>
                </div>
            </div>
        </div>
        <div class="chat-input-area">
            <div class="input-wrapper">
                <textarea id="messageInput" placeholder="Message DeCoupled-AI..." rows="1"></textarea>
                <div class="input-actions">
                    <button class="btn btn-ghost" id="attachBtn" title="Attach file">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
                        </svg>
                    </button>
                    <button class="btn btn-primary" id="sendBtn" disabled>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="22" y1="2" x2="11" y2="13"></line>
                            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                        </svg>
                    </button>
                </div>
            </div>
            <div class="input-hints">
                <span>Press <kbd>Enter</kbd> to send, <kbd>Shift+Enter</kbd> for new line</span>
                <span id="tokenCount">0 tokens</span>
            </div>
        </div>
    </div>
</div>"#
    }

    pub fn models_html() -> &'static str {
        r#"<div class="models-page">
    <div class="page-header">
        <h1>Model Manager</h1>
        <button class="btn btn-primary" id="loadModelBtn">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            Load Model
        </button>
    </div>
    <div class="models-grid" id="modelsGrid">
        <div class="loading-state">
            <div class="spinner"></div>
            <p>Loading models...</p>
        </div>
    </div>
</div>

<!-- Load Model Modal -->
<div class="modal" id="loadModelModal">
    <div class="modal-backdrop"></div>
    <div class="modal-content">
        <div class="modal-header">
            <h2>Load Model</h2>
            <button class="modal-close" id="closeLoadModal">&times;</button>
        </div>
        <form id="loadModelForm">
            <div class="form-group">
                <label for="modelPath">Model Path (.brain file)</label>
                <input type="text" id="modelPath" name="modelPath" placeholder="/models/llama-3-8b-q4_k_m.brain" required>
            </div>
            <div class="form-group">
                <label for="modelId">Model ID (optional)</label>
                <input type="text" id="modelId" name="modelId" placeholder="auto-generated">
            </div>
            <div class="modal-actions">
                <button type="button" class="btn btn-secondary" id="cancelLoadModel">Cancel</button>
                <button type="submit" class="btn btn-primary">Load Model</button>
            </div>
        </form>
    </div>
</div>"#
    }

    pub fn settings_html() -> &'static str {
        r#"<div class="settings-page">
    <div class="page-header">
        <h1>Settings</h1>
    </div>
    <div class="settings-grid">
        <section class="settings-section">
            <h2>Server</h2>
            <div class="setting-item">
                <label>Host</label>
                <input type="text" id="settingHost" class="setting-input" value="127.0.0.1" readonly>
            </div>
            <div class="setting-item">
                <label>Port</label>
                <input type="number" id="settingPort" class="setting-input" value="8080" readonly>
            </div>
            <div class="setting-item">
                <label>Backend</label>
                <select id="settingBackend" class="setting-select">
                    <option value="auto">Auto</option>
                    <option value="cpu">CPU</option>
                    <option value="cuda">CUDA</option>
                    <option value="rocm">ROCm</option>
                    <option value="metal">Metal</option>
                </select>
            </div>
            <div class="setting-item">
                <label>API Key</label>
                <div class="input-with-toggle">
                    <input type="password" id="settingApiKey" class="setting-input" placeholder="Optional">
                    <button type="button" class="btn btn-ghost toggle-visibility">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                            <circle cx="12" cy="12" r="3"></circle>
                        </svg>
                    </button>
                </div>
            </div>
        </section>
        <section class="settings-section">
            <h2>Inference</h2>
            <div class="setting-item">
                <label>Default Temperature</label>
                <input type="range" id="settingTemp" class="setting-range" min="0" max="2" step="0.1" value="0.7">
                <span id="settingTempValue">0.7</span>
            </div>
            <div class="setting-item">
                <label>Default Max Tokens</label>
                <input type="number" id="settingMaxTokens" class="setting-input" value="2048">
            </div>
            <div class="setting-item">
                <label>Default Top-p</label>
                <input type="range" id="settingTopP" class="setting-range" min="0" max="1" step="0.05" value="0.9">
                <span id="settingTopPValue">0.9</span>
            </div>
        </section>
        <section class="settings-section">
            <h2>Advanced</h2>
            <div class="setting-item">
                <label>Enable CORS</label>
                <label class="toggle">
                    <input type="checkbox" id="settingCors" checked>
                    <span class="toggle-slider"></span>
                </label>
            </div>
            <div class="setting-item">
                <label>Log Level</label>
                <select id="settingLogLevel" class="setting-select">
                    <option value="trace">Trace</option>
                    <option value="debug">Debug</option>
                    <option value="info" selected>Info</option>
                    <option value="warn">Warn</option>
                    <option value="error">Error</option>
                </select>
            </div>
        </section>
        <section class="settings-section">
            <h2>Streaming Cache (Sharded Models)</h2>
            <div class="streaming-stats" id="streamingStats">
                <div class="stat-card">
                    <div class="stat-label">RAM Resident</div>
                    <div class="stat-value" id="statResident">—</div>
                    <div class="stat-unit">/ <span id="statMaxResident">—</span></div>
                    <div class="stat-bar"><div class="stat-fill" id="statResidentFill" style="width: 0%"></div></div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Streaming Shards Mapped</div>
                    <div class="stat-value" id="statShardCount">—</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">NVMe Throughput (lifetime)</div>
                    <div class="stat-value" id="statTotalPulled">—</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Evictions</div>
                    <div class="stat-value" id="statEvictions">—</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Prefetch Hits</div>
                    <div class="stat-value" id="statPrefetchHits">—</div>
                </div>
            </div>
            <p class="stat-note" id="streamingNote">No sharded model loaded — monolithic .brain files show no streaming stats.</p>
        </section>
    </div>
</div>"#
    }

    pub fn download_html() -> &'static str {
        r#"<div class="download-page">
    <div class="page-header">
        <h1>Model Downloader</h1>
        <p class="page-subtitle">Download models from Hugging Face Hub and convert them for local inference</p>
    </div>

    <div class="download-grid">
        <!-- Download Form -->
        <section class="download-section">
            <h2>Download from Hugging Face</h2>
            <form id="downloadForm" class="download-form">
                <div class="form-group">
                    <label for="repoId">Repository ID <span class="required">*</span></label>
                    <input type="text" id="repoId" name="repoId" class="form-input"
                           placeholder="e.g., meta-llama/Llama-3.2-3B-Instruct" required>
                    <span class="help-text">Format: organization/model-name</span>
                </div>

                <div class="form-group">
                    <label for="revision">Revision (branch/tag/commit)</label>
                    <input type="text" id="revision" name="revision" class="form-input"
                           placeholder="main" value="main">
                    <span class="help-text">Branch name, tag, or commit hash</span>
                </div>

                <div class="form-group">
                    <label for="files">Files to Download (optional)</label>
                    <textarea id="files" name="files" class="form-input" rows="4"
                              placeholder="model.safetensors&#10;config.json&#10;tokenizer.json&#10;tokenizer.model"></textarea>
                    <span class="help-text">One file per line. Leave empty to download all files.</span>
                </div>

                <div class="form-group">
                    <label for="token">HF Token (for private/gated repos)</label>
                    <input type="password" id="token" name="token" class="form-input" placeholder="hf_...">
                    <span class="help-text">Get your token at <a href="https://huggingface.co/settings/tokens" target="_blank">huggingface.co/settings/tokens</a></span>
                </div>

                <div class="form-actions">
                    <button type="submit" class="btn btn-primary btn-lg" id="downloadBtn">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="17 8 12 3 7 8"></polyline>
                            <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                        Start Download
                    </button>
                </div>
            </form>
        </section>

        <!-- Active Downloads -->
        <section class="download-section">
            <h2>Active Downloads</h2>
            <div id="activeDownloads" class="downloads-list">
                <div class="empty-state">
                    <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                    <p>No active downloads</p>
                    <span class="hint">Start a download from the form on the left</span>
                </div>
            </div>
        </section>
    </div>

    <!-- Download History -->
    <section class="download-section history-section">
        <h2>Download History</h2>
        <div id="downloadHistory" class="downloads-list">
            <div class="empty-state">
                <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
                <p>No download history</p>
            </div>
        </div>
    </section>

    <!-- Convert to .brain Section -->
    <section class="download-section convert-section">
        <h2>Convert to .brain Format</h2>
        <p class="section-description">Convert downloaded safetensors models to DeCoupled-AI's optimized .brain format for fast local inference</p>
        <div class="convert-form">
            <div class="form-group">
                <label for="convertModelPath">Model Directory</label>
                <input type="text" id="convertModelPath" class="form-input" placeholder="/models/meta-llama/Llama-3.2-3B-Instruct">
            </div>
            <div class="form-group">
                <label for="convertOutputName">Output Model Name</label>
                <input type="text" id="convertOutputName" class="form-input" placeholder="llama-3.2-3b-instruct">
            </div>
            <div class="form-group">
                <label for="convertQuantization">Quantization</label>
                <select id="convertQuantization" class="form-select">
                    <option value="q4_k_m">Q4_K_M (Recommended - 4-bit)</option>
                    <option value="q8_0">Q8_0 (8-bit)</option>
                    <option value="f16">F16 (Half precision)</option>
                    <option value="f32">F32 (Full precision)</option>
                </select>
            </div>
            <div class="form-group class="form-actions">
                <button class="btn btn-secondary" id="convertBtn">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polygon points="18 17 18 23 22 23 22 17"></polygon>
                        <path d="M2 20v-7a4 4 0 0 1 4-4h12a4 4 0 0 1 4 4v7"></path>
                        <path d="M2 14h20"></path>
                    </svg>
                    Convert to .brain
                </button>
            </div>
            <div id="convertProgress" class="convert-progress hidden">
                <div class="progress-bar">
                    <div class="progress-fill" id="convertProgressFill" style="width: 0%"></div>
                </div>
                <div class="progress-info">
                    <span id="convertStatus">Converting...</span>
                    <span id="convertPercent">0%</span>
                </div>
            </div>
        </div>
    </section>
</div>"#
    }

    pub fn speculative_html() -> &'static str {
        r#"<div class="speculative-page">
    <div class="page-header">
        <h1>Speculative Decoding Dashboard</h1>
        <p class="page-subtitle">Real-time N-gram speculative decoding metrics and configuration</p>
    </div>

    <div class="speculative-grid">
        <!-- Configuration Panel -->
        <section class="speculative-section">
            <h2>Configuration</h2>
            <div class="config-form" id="configForm">
                <div class="form-group">
                    <label class="toggle-label">
                        <input type="checkbox" id="specEnabled" class="toggle-input">
                        <span class="toggle-slider"></span>
                        <span class="toggle-text">Enable Speculative Decoding</span>
                    </label>
                </div>

                <div class="form-group">
                    <label for="specMaxDraftTokens">Max Draft Tokens</label>
                    <input type="number" id="specMaxDraftTokens" class="form-input" min="1" max="32" value="8">
                    <span class="help-text">Maximum number of draft tokens per step</span>
                </div>

                <div class="form-group">
                    <label for="specConfidenceThreshold">Confidence Threshold</label>
                    <input type="range" id="specConfidenceThreshold" class="form-range" min="0" max="1" step="0.05" value="0.4">
                    <span class="range-value" id="specConfidenceValue">0.40</span>
                    <span class="help-text">Minimum confidence to accept draft tokens</span>
                </div>

                <div class="form-group">
                    <label for="specMinMatchLength">Min N-gram Match Length</label>
                    <input type="number" id="specMinMatchLength" class="form-input" min="2" max="4" value="3">
                    <span class="help-text">Minimum n-gram length to consider for drafting</span>
                </div>

                <div class="form-group">
                    <label for="specMaxNgramSize">Max N-gram Size</label>
                    <input type="number" id="specMaxNgramSize" class="form-input" min="2" max="4" value="4">
                    <span class="help-text">Maximum n-gram size in the index</span>
                </div>

                <div class="form-group">
                    <label for="specMaxIndexEntries">Max Index Entries</label>
                    <input type="number" id="specMaxIndexEntries" class="form-input" min="10000" max="10000000" value="100000" step="10000">
                    <span class="help-text">Maximum entries in the n-gram index (LRU eviction)</span>
                </div>

                <div class="form-group">
                    <label for="specVerificationBatchSize">Verification Batch Size</label>
                    <input type="number" id="specVerificationBatchSize" class="form-input" min="1" max="16" value="4">
                    <span class="help-text">Batch size for target model verification</span>
                </div>

                <div class="form-group">
                    <label for="specTemperature">Temperature</label>
                    <input type="range" id="specTemperature" class="form-range" min="0" max="1" step="0.05" value="0.7">
                    <span class="range-value" id="specTemperatureValue">0.70</span>
                    <span class="help-text">Sampling temperature for draft generation</span>
                </div>

                <div class="form-group">
                    <label for="specTopP">Top-p (Nucleus Sampling)</label>
                    <input type="range" id="specTopP" class="form-range" min="0" max="1" step="0.05" value="0.9">
                    <span class="range-value" id="specTopPValue">0.90</span>
                    <span class="help-text">Nucleus sampling threshold</span>
                </div>

                <div class="form-group">
                    <label for="specTopK">Top-k</label>
                    <input type="number" id="specTopK" class="form-input" min="1" max="100" value="40">
                    <span class="help-text">Top-k sampling limit</span>
                </div>

                <div class="form-actions">
                    <button type="button" class="btn btn-primary" id="saveSpecConfigBtn">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path>
                            <polyline points="17 21 17 13 7 13 7 21"></polyline>
                            <polyline points="7 3 7 8 15 8"></polyline>
                        </svg>
                        Save Configuration
                    </button>
                    <button type="button" class="btn btn-secondary" id="resetSpecConfigBtn">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path>
                            <path d="M3 3v5h5"></path>
                        </svg>
                        Reset to Defaults
                    </button>
                </div>

                <div id="configStatus" class="config-status hidden"></div>
            </div>
        </section>

        <!-- Real-time Metrics Panel -->
        <section class="speculative-section metrics-panel">
            <div class="section-header">
                <h2>Real-time Metrics</h2>
                <div class="connection-status" id="sseStatus">
                    <span class="status-dot"></span>
                    <span class="status-text">Connecting...</span>
                </div>
            </div>

            <div class="metrics-grid">
                <div class="metric-card primary">
                    <div class="metric-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label">Tokens Indexed</div>
                        <div class="metric-value" id="metricTokensIndexed">—</div>
                    </div>
                </div>

                <div class="metric-card primary">
                    <div class="metric-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <path d="M12 6v6l4 2"></path>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label">Unique N-grams</div>
                        <div class="metric-value" id="metricUniqueNgrams">—</div>
                    </div>
                </div>

                <div class="metric-card success">
                    <div class="metric-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                            <polyline points="22 4 12 14.01 9 11.01"></polyline>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label">Acceptance Rate</div>
                        <div class="metric-value" id="metricAcceptanceRate">—</div>
                    </div>
                </div>

                <div class="metric-card warning">
                    <div class="metric-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="12" y1="8" x2="12" y2="12"></line>
                            <line x1="12" y1="16" x2="12.01" y2="16"></line>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label">Avg Tokens/Step</div>
                        <div class="metric-value" id="metricAvgTokensPerStep">—</div>
                    </div>
                </div>

                <div class="metric-card info">
                    <div class="metric-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="2" y="3" width="20" height="14" rx="2"></rect>
                            <path d="M8 21h8"></path>
                            <path d="M12 17v4"></path>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label">Verification Batches</div>
                        <div class="metric-value" id="metricVerificationBatches">—</div>
                    </div>
                </div>

                <div class="metric-card danger">
                    <div class="metric-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 6h18"></path>
                            <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
                            <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label">Rejections</div>
                        <div class="metric-value" id="metricRejections">—</div>
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                            <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                            <line x1="12" y1="22.08" x2="12" y2="12"></line>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label">Index Evictions</div>
                        <div class="metric-value" id="metricIndexEvictions">—</div>
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"></path>
                        </svg>
                    </div>
                    <div class="metric-content">
                        <div class="metric-label">Speedup Ratio</div>
                        <div class="metric-value" id="metricSpeedup">—</div>
                    </div>
                </div>
            </div>

            <!-- Acceptance Rate Chart -->
            <div class="chart-container">
                <h3>Acceptance Rate History</h3>
                <canvas id="acceptanceChart"></canvas>
            </div>

            <!-- Tokens per Step Chart -->
            <div class="chart-container">
                <h3>Tokens per Step</h3>
                <canvas id="tokensPerStepChart"></canvas>
            </div>
        </section>

        <!-- N-gram Index Inspector -->
        <section class="speculative-section index-inspector">
            <h2>N-gram Index Inspector</h2>
            <div class="inspector-controls">
                <div class="form-group">
                    <label for="inspectNgramSize">N-gram Size</label>
                    <select id="inspectNgramSize" class="form-select">
                        <option value="4">4-gram</option>
                        <option value="3">3-gram</option>
                        <option value="2">2-gram</option>
                        <option value="1">1-gram</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="inspectLimit">Max Entries</label>
                    <input type="number" id="inspectLimit" class="form-input" min="10" max="500" value="50">
                </div>
                <button class="btn btn-secondary" id="refreshIndexBtn">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M23 4v6h-6"></path>
                        <path d="M1 20v-6h6"></path>
                        <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
                    </svg>
                    Refresh
                </button>
            </div>
            <div class="index-table-container">
                <table class="index-table" id="indexTable">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>N-gram</th>
                            <th>Count</th>
                            <th>Next Tokens (Top 5)</th>
                            <th>Last Seen</th>
                        </tr>
                    </thead>
                    <tbody id="indexTableBody">
                        <tr><td colspan="5" class="loading-row">Loading index data...</td></tr>
                    </tbody>
                </table>
            </div>
        </section>
    </div>
</div>"#
    }
}

impl Default for FrontendAssets {
    fn default() -> Self {
        Self::new()
    }
}