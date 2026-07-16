/* =============================================================================
 * DeCoupled-AI Dashboard — Client Controller
 *
 * Single-page app: chat interface, model manager, metrics, settings.
 * Talks to the embedded server via fetch + SSE/WS. No external dependencies.
 * ========================================================================== */

(() => {
"use strict";

const API_BASE = "";
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
    activeModel: null,
    models: [],
    messages: [],
    streaming: false,
    metrics: { tokensPerSec: 0, totalTokens: 0, peakMem: 0, requests: 0 },
    settings: loadSettings(),
};

// --------------------------------------------------------------------------
// Settings persistence
// --------------------------------------------------------------------------
function loadSettings() {
    const def = {
        temperature: 0.7,
        topP: 0.9,
        topK: 40,
        maxTokens: 1024,
        seed: 0,
        stream: true,
        backend: "auto",
        repeatPenalty: 1.10,
    };
    try {
        return Object.assign(def, JSON.parse(localStorage.getItem("dai_settings") || "{}"));
    } catch { return def; }
}
function saveSettings() {
    try { localStorage.setItem("dai_settings", JSON.stringify(state.settings)); } catch {}
}

// --------------------------------------------------------------------------
// Toast
// --------------------------------------------------------------------------
function toast(msg, ms = 2400) {
    const el = document.createElement("div");
    el.className = "toast";
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => {
        el.style.transition = "opacity 360ms";
        el.style.opacity = "0";
        setTimeout(() => el.remove(), 400);
    }, ms);
}

// --------------------------------------------------------------------------
// API helpers
// --------------------------------------------------------------------------
async function api(path, opts = {}) {
    const res = await fetch(API_BASE + path, {
        headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
        ...opts,
    });
    if (!res.ok) {
        const body = await res.text();
        throw new Error(`${res.status} ${res.statusText} — ${body}`);
    }
    return res;
}

async function loadModels() {
    try {
        const res = await api("/v1/models");
        const data = await res.json();
        state.models = data.data || [];
        renderModels();
    } catch (e) {
        console.warn("model list failed", e);
        $$(".status-pill,.status").forEach(p => {
            p.classList.add("offline");
            p.textContent = "Offline";
        });
    }
}

async function pushStats() {
    try {
        const res = await api("/metrics");
        const m = await res.json();
        state.metrics = { ...state.metrics, ...m };
        renderMetrics();
    } catch { /* ignore */ }
}

// --------------------------------------------------------------------------
// Rendering
// --------------------------------------------------------------------------
function renderModels() {
    const list = $(".model-list");
    if (!list) return;
    list.innerHTML = "";
    for (const m of state.models) {
        const item = document.createElement("div");
        item.className = "model-item" + (m.id === state.activeModel ? " active" : "");
        const isLoaded = (m.status || "loaded") === "loaded";
        item.innerHTML = `
            <div class="model-dot ${isLoaded ? "" : "unloaded"}"></div>
            <div class="model-meta">
                <div class="model-name">${escapeHtml(m.id)}</div>
                <div class="model-info">${formatBytes(m.bytes || 0)} · ${m.quant || "?"}</div>
            </div>`;
        item.addEventListener("click", () => selectModel(m.id));
        list.appendChild(item);
    }
}

function renderMetrics() {
    $$(".metric-card").forEach(card => {
        const name = card.dataset.metric;
        const valEl = card.querySelector(".value");
        const deltaEl = card.querySelector(".delta");
        const v = state.metrics[name];
        if (v === undefined) return;
        if (typeof v === "number") {
            valEl.textContent = name === "peakMem" ? formatBytes(v) : v.toFixed(2);
            if (deltaEl && state.previousMetrics) {
                const prev = state.previousMetrics[name] || 0;
                const d = v - prev;
                deltaEl.textContent = (d >= 0 ? "▲ " : "▼ ") + Math.abs(d).toFixed(2);
                deltaEl.classList.toggle("down", d < 0);
            }
        } else {
            valEl.textContent = v;
        }
    });
    state.previousMetrics = { ...state.metrics };
}

function renderChat() {
    const stream = $(".chat-stream");
    if (!stream) return;
    stream.innerHTML = "";
    for (const msg of state.messages) {
        stream.appendChild(renderMessage(msg));
    }
    stream.scrollTop = stream.scrollHeight;
}

function renderMessage(msg) {
    const el = document.createElement("div");
    el.className = "message";
    const avatar = msg.role === "user" ? "U" : "AI";
    el.innerHTML = `
        <div class="msg-avatar ${msg.role}">${avatar}</div>
        <div class="msg-body">
            ${renderMarkdown(msg.content)}
            <div class="msg-meta">
                <span>${msg.role}</span>
                ${msg.tokens ? `<span>· ${msg.tokens} tok</span>` : ""}
                ${msg.tps ? `<span>· ${msg.tps.toFixed(1)} tok/s</span>` : ""}
            </div>
        </div>`;
    return el;
}

// Minimal markdown subset: ```code```, `inline`, **bold**, *italic*
function renderMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);
    html = html.replace(/```([\s\S]*?)```/g, (_, c) => `<pre>${c}</pre>`);
    html = html.replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`);
    html = html.replace(/\*\*([^*]+)\*\*/g, (_, c) => `<b>${c}</b>`);
    html = html.replace(/\*([^*]+)\*/g, (_, c) => `<i>${c}</i>`);
    html = html.replace(/\n/g, "<br>");
    return html;
}

function escapeHtml(s) {
    return String(s)
        .replace(/&/g, "&").replace(/</g, "<").replace(/>/g, ">")
        .replace(/"/g, """).replace(/'/g, "&#39;");
}

function formatBytes(b) {
    if (b === 0) return "0 B";
    const u = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(b) / Math.log(1024));
    return (b / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 2) + " " + u[i];
}

// --------------------------------------------------------------------------
// Model selection
// --------------------------------------------------------------------------
async function selectModel(id) {
    if (state.activeModel === id) return;
    state.activeModel = id;
    renderModels();
    try {
        await api(`/v1/models/${encodeURIComponent(id)}/load`, { method: "POST" });
        toast(`Model loaded: ${id}`);
    } catch (e) {
        toast(`Load failed: ${e.message}`);
    }
}

// --------------------------------------------------------------------------
// Chat submission
// --------------------------------------------------------------------------
async function submitChat(text) {
    if (!text || state.streaming) return;
    if (!state.activeModel) {
        toast("Select a model first");
        return;
    }

    state.messages.push({ role: "user", content: text });
    renderChat();

    const assistantMsg = { role: "assistant", content: "", tokens: 0, tps: 0 };
    state.messages.push(assistantMsg);
    state.streaming = true;
    setSubmitState(true);

    const start = performance.now();
    let lastChunk = start;

    const body = {
        model: state.activeModel,
        messages: state.messages.filter(m => m.role !== "assistant" || m.content)
                                .map(m => ({ role: m.role, content: m.content })),
        stream: !!state.settings.stream,
        temperature: parseFloat(state.settings.temperature),
        top_p: parseFloat(state.settings.topP),
        max_tokens: parseInt(state.settings.maxTokens, 10),
        seed: parseInt(state.settings.seed, 10) || undefined,
    };

    if (body.stream) {
        await streamChat(body, assistantMsg, (delta) => {
            assistantMsg.content += delta;
            const now = performance.now();
            if (now - lastChunk > 60) { renderChat(); lastChunk = now; }
        });
    } else {
        try {
            const res = await api("/v1/chat/completions", {
                method: "POST",
                body: JSON.stringify(body),
            });
            const data = await res.json();
            assistantMsg.content = data.choices?.[0]?.message?.content || "";
            if (data.usage) {
                assistantMsg.tokens = data.usage.completion_tokens;
                const dur = (performance.now() - start) / 1000;
                assistantMsg.tps = dur > 0 ? assistantMsg.tokens / dur : 0;
            }
        } catch (e) {
            assistantMsg.content = `[error] ${e.message}`;
        }
    }

    state.streaming = false;
    setSubmitState(false);
    renderChat();
    pushStats();
}

async function streamChat(body, assistantMsg, onDelta) {
    return new Promise(async (resolve) => {
        try {
            const res = await fetch(API_BASE + "/v1/chat/completions", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                body: JSON.stringify({ ...body, stream: true }),
            });
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buf = "";
            let tokens = 0;
            const start = performance.now();

            for (;;) {
                const { done, value } = await reader.read();
                if (done) break;
                buf += decoder.decode(value, { stream: true });
                let idx;
                while ((idx = buf.indexOf("\n\n")) !== -1) {
                    const raw = buf.slice(0, idx);
                    buf = buf.slice(idx + 2);
                    const dataLine = raw.split("\n").find(l => l.startsWith("data: "));
                    if (!dataLine) continue;
                    const payload = dataLine.slice(6);
                    if (payload === "[DONE]") { resolve(); return; }
                    try {
                        const chunk = JSON.parse(payload);
                        const delta = chunk.choices?.[0]?.delta?.content || "";
                        if (delta) { onDelta(delta); tokens++; }
                    } catch { /* ignore */ }
                }
            }
            const dur = (performance.now() - start) / 1000;
            assistantMsg.tokens = tokens;
            assistantMsg.tps = dur > 0 ? tokens / dur : 0;
        } catch (e) {
            assistantMsg.content = `[stream error] ${e.message}`;
        }
        resolve();
    });
}

function setSubmitState(streaming) {
    const btn = $("#send-btn");
    const ta = $("#chat-input");
    if (btn) btn.disabled = streaming;
    if (ta) ta.disabled = streaming;
}

// --------------------------------------------------------------------------
// Settings page
// --------------------------------------------------------------------------
function bindSettings() {
    const form = $("#settings-form");
    if (!form) return;
    ["temperature", "topP", "topK", "maxTokens", "seed", "repeatPenalty"].forEach(name => {
        const el = form.elements[name];
        if (el) {
            el.value = state.settings[name];
            const out = form.elements[name + "_val"];
            if (out) out.textContent = el.value;
            el.addEventListener("input", () => {
                state.settings[name] = parseFloat(el.value);
                if (out) out.textContent = el.value;
                saveSettings();
            });
        }
    });
    const streamEl = form.elements.stream;
    if (streamEl) {
        streamEl.checked = state.settings.stream;
        streamEl.addEventListener("change", () => {
            state.settings.stream = streamEl.checked;
            saveSettings();
        });
    }
    const backendEl = form.elements.backend;
    if (backendEl) {
        backendEl.value = state.settings.backend;
        backendEl.addEventListener("change", () => {
            state.settings.backend = backendEl.value;
            saveSettings();
        });
    }
}

// --------------------------------------------------------------------------
// Streaming Cache Stats
// --------------------------------------------------------------------------
async function pushStreamingStats() {
    try {
        const res = await api("/v1/streaming/stats");
        if (res.ok) {
            const s = await res.json();
            updateStreamingStatsUI(s);
        } else {
            // No streaming model loaded - keep note visible
            showStreamingNote(true);
        }
    } catch {
        showStreamingNote(true);
    }
}

function updateStreamingStatsUI(s) {
    const note = $("#streamingNote");
    if (note) note.style.display = "none";

    const fmt = formatBytes;

    const resident = $("#statResident");
    const maxResident = $("#statMaxResident");
    const residentFill = $("#statResidentFill");
    const shardCount = $("#statShardCount");
    const totalPulled = $("#statTotalPulled");
    const evictions = $("#statEvictions");
    const prefetchHits = $("#statPrefetchHits");

    if (resident && s.current_resident_bytes !== undefined) {
        resident.textContent = fmt(s.current_resident_bytes);
        maxResident.textContent = fmt(s.max_resident_bytes);
        const pct = s.max_resident_bytes > 0
            ? Math.min(100, (s.current_resident_bytes / s.max_resident_bytes) * 100)
            : 0;
        residentFill.style.width = pct + "%";
    }
    if (shardCount && s.resident_shard_count !== undefined) {
        shardCount.textContent = s.resident_shard_count.toString();
    }
    if (totalPulled && s.total_bytes_pulled !== undefined) {
        totalPulled.textContent = fmt(s.total_bytes_pulled);
    }
    if (evictions && s.evictions !== undefined) {
        evictions.textContent = s.evictions.toString();
    }
    if (prefetchHits && s.prefetch_hits !== undefined) {
        prefetchHits.textContent = s.prefetch_hits.toString();
    }
}

function showStreamingNote(show) {
    const note = $("#streamingNote");
    if (note) note.style.display = show ? "block" : "none";
    const grid = $("#streamingStats");
    if (grid) grid.style.display = show ? "none" : "grid";
}

// --------------------------------------------------------------------------
// Download Page
// --------------------------------------------------------------------------
async function initDownloadPage() {
    const downloadForm = $("#downloadForm");
    if (!downloadForm) return;

    // Load download history
    await loadDownloadHistory();

    // Handle download form submission
    downloadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const formData = new FormData(downloadForm);
        const repoId = formData.get("repoId").trim();
        if (!repoId) {
            toast("Repository ID is required");
            return;
        }

        const request = {
            repo_id: repoId,
            revision: formData.get("revision").trim() || "main",
            files: formData.get("files").trim()
                ? formData.get("files").trim().split("\n").map(f => f.trim()).filter(f => f)
                : undefined,
            token: formData.get("token").trim() || undefined,
        };

        try {
            const btn = $("#downloadBtn");
            if (btn) btn.disabled = true;
            btn.innerHTML = '<svg class="spin" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke-opacity="0.75"></path></svg> Starting...';

            const res = await api("/v1/models/download", {
                method: "POST",
                body: JSON.stringify(request),
            });
            const data = await res.json();

            if (data.download_id) {
                toast(`Download started: ${data.download_id}`);
                startProgressTracking(data.download_id);
                downloadForm.reset();
            } else {
                toast("Failed to start download");
            }
        } catch (e) {
            toast(`Download failed: ${e.message}`);
        } finally {
            const btn = $("#downloadBtn");
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg> Start Download`;
            }
        }
    });

    // Handle convert form
    const convertBtn = $("#convertBtn");
    if (convertBtn) {
        convertBtn.addEventListener("click", async () => {
            const modelPath = $("#convertModelPath")?.value?.trim();
            const outputName = $("#convertOutputName")?.value?.trim();
            const quantization = $("#convertQuantization")?.value;

            if (!modelPath) {
                toast("Model directory path is required");
                return;
            }

            // For now, show a toast since conversion is a CLI command
            toast("Conversion is currently a CLI command. Run: brain-pack convert --input <path> --output <name> --quantization " + quantization);
        });
    }
}

async function loadDownloadHistory() {
    const historyEl = $("#downloadHistory");
    if (!historyEl) return;

    try {
        // In a real implementation, we'd fetch from a history endpoint
        // For now, show empty state
        historyEl.innerHTML = `
            <div class="empty-state">
                <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
                <p>No download history</p>
            </div>
        `;
    } catch (e) {
        console.warn("Failed to load download history:", e);
    }
}

function startProgressTracking(downloadId) {
    const activeEl = $("#activeDownloads");
    if (!activeEl) return;

    // Remove empty state
    activeEl.innerHTML = "";

    // Create download item
    const item = document.createElement("div");
    item.className = "download-item";
    item.id = `download-${downloadId}`;
    item.innerHTML = `
        <div class="download-header">
            <div class="download-info">
                <span class="download-id">${downloadId.slice(0, 8)}...</span>
                <span class="download-repo"></span>
            </div>
            <span class="download-status pending">Starting...</span>
        </div>
        <div class="download-progress">
            <div class="progress-bar"><div class="progress-fill" style="width: 0%"></div></div>
            <div class="progress-details">
                <span class="progress-bytes">0 B / 0 B</span>
                <span class="progress-speed">0 B/s</span>
            </div>
        </div>
        <div class="download-files"></div>
    `;
    activeEl.appendChild(item);

    // Connect to SSE for progress updates
    const eventSource = new EventSource(`${API_BASE}/v1/models/download/${downloadId}/sse`);
    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            updateDownloadProgress(downloadId, data);
        } catch (e) {
            console.warn("Failed to parse SSE data:", e);
        }
    };
    eventSource.onerror = () => {
        eventSource.close();
    };

    // Also poll as fallback
    const pollInterval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/v1/models/download/${downloadId}`);
            if (res.ok) {
                const data = await res.json();
                updateDownloadProgress(downloadId, data);
                if (data.status === "completed" || data.status === "failed") {
                    clearInterval(pollInterval);
                    eventSource.close();
                }
            }
        } catch (e) {
            console.warn("Progress poll failed:", e);
        }
    }, 2000);
}

function updateDownloadProgress(downloadId, data) {
    const item = $(`#download-${downloadId}`);
    if (!item) return;

    const statusEl = item.querySelector(".download-status");
    const progressFill = item.querySelector(".progress-fill");
    const progressBytes = item.querySelector(".progress-bytes");
    const progressSpeed = item.querySelector(".progress-speed");
    const filesEl = item.querySelector(".download-files");

    if (statusEl) {
        statusEl.textContent = data.status || "Unknown";
        statusEl.className = "download-status " + (data.status || "").toLowerCase();
    }

    if (data.downloaded_bytes !== undefined && data.total_bytes !== undefined) {
        if (progressFill) progressFill.style.width = data.total_bytes > 0
            ? `${Math.min(100, (data.downloaded_bytes / data.total_bytes) * 100)}%`
            : "0%";
        if (progressBytes) progressBytes.textContent = `${formatBytes(data.downloaded_bytes)} / ${formatBytes(data.total_bytes)}`;
        if (progressSpeed) progressSpeed.textContent = `${formatBytes(data.speed_bps || 0)}/s`;
    }

    if (data.files && filesEl) {
        filesEl.innerHTML = data.files.map(f => `
            <div class="file-progress">
                <span class="file-name">${escapeHtml(f.file_name)}</span>
                <div class="file-progress-bar"><div class="file-progress-fill" style="width: ${f.total_bytes > 0 ? (f.bytes_downloaded / f.total_bytes) * 100 : 0}%"></div></div>
                <span class="file-status">${f.finished ? "✓" : "⟳"}</span>
            </div>
        `).join("");
    }

    // If completed or failed, update history after a delay
    if (data.status === "completed" || data.status === "failed") {
        setTimeout(() => {
            loadDownloadHistory();
        }, 2000);
    }
}

// --------------------------------------------------------------------------
// Speculative Decoding Dashboard
// --------------------------------------------------------------------------
async function initSpeculativePage() {
    const configForm = $("#configForm");
    if (!configForm) return;

    // SSE connection for real-time metrics
    let speculativeSSE = null;
    let acceptanceChart = null;
    let tokensPerStepChart = null;
    let acceptanceHistory = [];
    let tokensPerStepHistory = [];
    const MAX_HISTORY = 60;

    // Initialize charts
    function initCharts() {
        const acceptanceCtx = $("#acceptanceChart");
        const tokensCtx = $("#tokensPerStepChart");
        if (!acceptanceCtx || !tokensCtx) return;

        acceptanceChart = new Chart(acceptanceCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Acceptance Rate',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { display: false },
                    y: { min: 0, max: 1, ticks: { stepSize: 0.2 } }
                },
                plugins: { legend: { display: false } }
            }
        });

        tokensPerStepChart = new Chart(tokensCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Tokens/Step',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { display: false },
                    y: { min: 0, suggestedMax: 10 }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    // Update metrics from SSE
    function updateMetrics(metrics) {
        // Update metric cards
        updateMetricCard("metricTokensIndexed", metrics.tokens_indexed);
        updateMetricCard("metricUniqueNgrams", metrics.unique_ngrams);
        updateMetricCard("metricAcceptanceRate", metrics.acceptance_rate !== undefined ? (metrics.acceptance_rate * 100).toFixed(1) + "%" : "—");
        updateMetricCard("metricAvgTokensPerStep", metrics.avg_tokens_per_step !== undefined ? metrics.avg_tokens_per_step.toFixed(2) : "—");
        updateMetricCard("metricVerificationBatches", metrics.verification_batches);
        updateMetricCard("metricRejections", metrics.rejections);
        updateMetricCard("metricIndexEvictions", metrics.index_evictions);
        updateMetricCard("metricSpeedup", metrics.speedup_ratio !== undefined ? metrics.speedup_ratio.toFixed(2) + "x" : "—");

        // Update charts
        if (acceptanceChart && metrics.acceptance_rate !== undefined) {
            const now = new Date().toLocaleTimeString();
            acceptanceHistory.push({ time: now, value: metrics.acceptance_rate });
            if (acceptanceHistory.length > MAX_HISTORY) acceptanceHistory.shift();

            acceptanceChart.data.labels = acceptanceHistory.map(h => h.time);
            acceptanceChart.data.datasets[0].data = acceptanceHistory.map(h => h.value);
            acceptanceChart.update('none');
        }

        if (tokensPerStepChart && metrics.avg_tokens_per_step !== undefined) {
            const now = new Date().toLocaleTimeString();
            tokensPerStepHistory.push({ time: now, value: metrics.avg_tokens_per_step });
            if (tokensPerStepHistory.length > MAX_HISTORY) tokensPerStepHistory.shift();

            tokensPerStepChart.data.labels = tokensPerStepHistory.map(h => h.time);
            tokensPerStepChart.data.datasets[0].data = tokensPerStepHistory.map(h => h.value);
            tokensPerStepChart.update('none');
        }
    }

    function updateMetricCard(id, value) {
        const el = $(`#${id}`);
        if (el) el.textContent = value !== undefined && value !== null ? value : "—";
    }

    // SSE connection
    function connectSSE() {
        const statusEl = $("#sseStatus");
        speculativeSSE = new EventSource(`${API_BASE}/v1/speculative/metrics/sse`);

        speculativeSSE.onopen = () => {
            if (statusEl) {
                statusEl.querySelector(".status-dot").classList.add("connected");
                statusEl.querySelector(".status-text").textContent = "Connected";
            }
        };

        speculativeSSE.onmessage = (event) => {
            try {
                const metrics = JSON.parse(event.data);
                updateMetrics(metrics);
            } catch (e) {
                console.warn("Failed to parse SSE metrics:", e);
            }
        };

        speculativeSSE.onerror = () => {
            if (statusEl) {
                statusEl.querySelector(".status-dot").classList.remove("connected");
                statusEl.querySelector(".status-text").textContent = "Reconnecting...";
            }
            speculativeSSE.close();
            setTimeout(connectSSE, 3000);
        };
    }

    // Load current config
    async function loadConfig() {
        try {
            const res = await api("/v1/speculative/config");
            const config = await res.json();
            populateConfigForm(config);
        } catch (e) {
            console.warn("Failed to load speculative config:", e);
        }
    }

    function populateConfigForm(config) {
        const fields = {
            "specEnabled": "enabled",
            "specMaxDraftTokens": "max_draft_tokens",
            "specConfidenceThreshold": "confidence_threshold",
            "specMinMatchLength": "min_match_length",
            "specMaxNgramSize": "max_ngram_size",
            "specMaxIndexEntries": "max_index_entries",
            "specVerificationBatchSize": "verification_batch_size",
            "specTemperature": "temperature",
            "specTopP": "top_p",
            "specTopK": "top_k",
        };

        for (const [id, key] of Object.entries(fields)) {
            const el = $(`#${id}`);
            if (el && config[key] !== undefined) {
                if (el.type === "checkbox") {
                    el.checked = config[key];
                } else if (el.type === "range") {
                    el.value = config[key];
                    updateRangeValue(id);
                } else {
                    el.value = config[key];
                }
            }
        }
    }

    function updateRangeValue(id) {
        const el = $(`#${id}`);
        const valueEl = $(`#${id}Value`);
        if (el && valueEl) {
            valueEl.textContent = parseFloat(el.value).toFixed(2);
        }
    }

    // Range input handlers
    $$(".form-range").forEach(input => {
        input.addEventListener("input", () => updateRangeValue(input.id));
    });

    // Save config
    $("#saveSpecConfigBtn")?.addEventListener("click", async () => {
        const statusEl = $("#configStatus");
        const btn = $("#saveSpecConfigBtn");

        const config = {
            enabled: $("#specEnabled")?.checked,
            max_draft_tokens: parseInt($("#specMaxDraftTokens")?.value) || 8,
            confidence_threshold: parseFloat($("#specConfidenceThreshold")?.value) || 0.4,
            min_match_length: parseInt($("#specMinMatchLength")?.value) || 3,
            max_ngram_size: parseInt($("#specMaxNgramSize")?.value) || 4,
            max_index_entries: parseInt($("#specMaxIndexEntries")?.value) || 100000,
            verification_batch_size: parseInt($("#specVerificationBatchSize")?.value) || 4,
            temperature: parseFloat($("#specTemperature")?.value) || 0.7,
            top_p: parseFloat($("#specTopP")?.value) || 0.9,
            top_k: parseInt($("#specTopK")?.value) || 40,
        };

        try {
            btn.disabled = true;
            btn.innerHTML = '<svg class="spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke-opacity="0.75"></path></svg> Saving...';

            const res = await api("/v1/speculative/config", {
                method: "POST",
                body: JSON.stringify(config),
            });
            const result = await res.json();

            if (statusEl) {
                statusEl.textContent = "Configuration saved successfully!";
                statusEl.className = "config-status success";
                statusEl.classList.remove("hidden");
            }
        } catch (e) {
            if (statusEl) {
                statusEl.textContent = `Failed: ${e.message}`;
                statusEl.className = "config-status error";
                statusEl.classList.remove("hidden");
            }
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path><polyline points="17 21 17 13 7 13 7 21"></polyline><polyline points="7 3 7 8 15 8"></polyline></svg> Save Configuration';
        }
    });

    // Reset config
    $("#resetSpecConfigBtn")?.addEventListener("click", async () => {
        const defaults = {
            enabled: false,
            max_draft_tokens: 8,
            confidence_threshold: 0.4,
            min_match_length: 3,
            max_ngram_size: 4,
            max_index_entries: 100000,
            verification_batch_size: 4,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
        };
        populateConfigForm(defaults);
        toast("Reset to defaults - click Save to apply");
    });

    // Index inspector
    $("#refreshIndexBtn")?.addEventListener("click", async () => {
        const ngramSize = parseInt($("#inspectNgramSize")?.value) || 4;
        const limit = parseInt($("#inspectLimit")?.value) || 50;
        const tbody = $("#indexTableBody");

        tbody.innerHTML = '<tr><td colspan="5" class="loading-row">Loading index data...</td></tr>';

        try {
            const res = await api(`/v1/speculative/index?size=${ngramSize}&limit=${limit}`);
            const data = await res.json();

            if (data.entries && data.entries.length > 0) {
                tbody.innerHTML = data.entries.map((entry, i) => `
                    <tr>
                        <td>${i + 1}</td>
                        <td><code>${escapeHtml(entry.ngram.join(", "))}</code></td>
                        <td>${entry.count}</td>
                        <td>${entry.next_tokens.map(t => `<code>${escapeHtml(t.token)}</code> (${t.count})`).join(", ")}</td>
                        <td>${new Date(entry.last_seen).toLocaleString()}</td>
                    </tr>
                `).join("");
            } else {
                tbody.innerHTML = '<tr><td colspan="5" class="loading-row">No entries found</td></tr>';
            }
        } catch (e) {
            tbody.innerHTML = `<tr><td colspan="5" class="loading-row error">Failed to load: ${escapeHtml(e.message)}</td></tr>`;
        }
    });

    // Initialize
    initCharts();
    await loadConfig();
    connectSSE();

    // Initial index load
    $("#refreshIndexBtn")?.click();
}

// --------------------------------------------------------------------------
// Bootstrap
// --------------------------------------------------------------------------
function attachChatInput() {
    const ta = $("#chat-input");
    if (!ta) return;
    ta.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            const text = ta.value.trim();
            if (!text) return;
            ta.value = "";
            submitChat(text);
        }
    });
    const btn = $("#send-btn");
    if (btn) btn.addEventListener("click", () => {
        const text = (ta.value || "").trim();
        if (!text) return;
        ta.value = "";
        submitChat(text);
    });
}

function initNav() {
    $$(".nav a").forEach(a => {
        a.addEventListener("click", (e) => {
            e.preventDefault();
            const href = a.getAttribute("href");
            $$(".nav a").forEach(n => n.classList.remove("active"));
            a.classList.add("active");
            // Server-side already serves different pages, but for the embedded
            // SPA we navigate via fetch and patch main content.
            location.href = href;
        });
    });
}

function init() {
    bindSettings();
    attachChatInput();
    initNav();
    loadModels();
    pushStats();
    pushStreamingStats();
    initDownloadPage();
    initSpeculativePage();
    setInterval(pushStats, 4000);
    setInterval(pushStreamingStats, 4000);
    if (window.InterSectionObserver === undefined && location.pathname !== "/") {
        // opt: re-render stream on load
        renderChat();
    }
}

document.addEventListener("DOMContentLoaded", init);

// Expose for inline onclick handlers/tests
window.DAI = { state, submitChat, loadModels, toast };

})();
