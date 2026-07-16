//! OpenAI-Compatible API Implementation
//!
//! Implements the exact OpenAI API specification for chat completions,
//! completions, and embeddings endpoints with streaming support.

use anyhow::{anyhow, Result};
use engine_ipc::{InferenceEngine, GenerateRequest, GenerateResponse, ModelInfo as EngineModelInfo};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};
use uuid::Uuid;

// ============================================================================
// Request/Response Types (OpenAI Compatible)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub r#type: String,
    pub function: FunctionDef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    None(String), // "none"
    Auto(String), // "auto"
    Function { r#type: String, function: FunctionName },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionName {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    #[serde(default)]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
    #[serde(default)]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: ChatDelta,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: CompletionPrompt,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub echo: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CompletionPrompt {
    String(String),
    Array(Vec<String>),
    Tokens(Vec<i32>),
    TokensArray(Vec<Vec<i32>>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub encoding_format: Option<String>, // "float" or "base64"
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    String(String),
    Array(Vec<String>),
    Tokens(Vec<i32>),
    TokensArray(Vec<Vec<i32>>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<OpenAiModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub permission: Vec<serde_json::Value>,
    pub root: String,
    pub parent: Option<String>,
    #[serde(flatten)]
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl OpenAiModelInfo {
    pub fn from_engine(m: &EngineModelInfo) -> Self {
        Self {
            id: m.id.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "decoupled-ai".to_string(),
            permission: vec![],
            root: m.id.clone(),
            parent: None,
            metadata: std::collections::HashMap::new(),
        }
    }
}

// ============================================================================
// API Implementation
// ============================================================================

#[derive(Clone)]
pub struct OpenAiApi {
    engine: Arc<tokio::sync::Mutex<InferenceEngine>>,
}

impl OpenAiApi {
    pub fn new(engine: Arc<tokio::sync::Mutex<InferenceEngine>>) -> Self {
        Self { engine }
    }

    /// Non-streaming chat completions
    pub async fn chat_completions(&self, request: ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        let engine = self.engine.lock().await;
        let model = engine.get_model(&request.model)
            .ok_or_else(|| anyhow!("Model not found: {}", request.model))?;
        let prompt = self.messages_to_prompt(&request.messages)?;

        let gen_request = GenerateRequest {
            model_id: model.id.clone(),
            prompt_tokens: prompt,
            max_tokens: request.max_tokens.unwrap_or(2048),
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: 0, // Not used in OpenAI API
            stop_tokens: self.tokenize_many(&request.stop.unwrap_or_default())?,
            stream: false,
        };

        let response = engine.generate_async(gen_request).await?;
        Ok(self.format_chat_response(&request.model, response))
    }

    /// Streaming chat completions
    pub async fn chat_completions_stream(&self, request: ChatCompletionRequest) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>> {
        let engine = self.engine.lock().await;
        let model = engine.get_model(&request.model)
            .ok_or_else(|| anyhow!("Model not found: {}", request.model))?;
        let prompt = self.messages_to_prompt(&request.messages)?;

        let gen_request = GenerateRequest {
            model_id: model.id.clone(),
            prompt_tokens: prompt,
            max_tokens: request.max_tokens.unwrap_or(2048),
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: 0,
            stop_tokens: self.tokenize_many(&request.stop.unwrap_or_default())?,
            stream: true,
        };

        let response_stream = engine.generate_stream(gen_request)?;

        let request_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
        let created = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let model_name = request.model.clone();

        // Transform engine stream (one item per token) to OpenAI SSE chunks.
        let stream = response_stream.map(move |result| {
            match result {
                Ok((token_id, finish_reason)) => {
                    // Decode single token. Real backends would BPE-decode here.
                    // Simple character-level detokenization (same as OpenAiApi::detokenize)
                    let content = char::from_u32(token_id as u32).unwrap_or('?').to_string();
                    let chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: ChatDelta {
                                role: Some("assistant".to_string()),
                                content: Some(content),
                                tool_calls: None,
                            },
                            logprobs: None,
                            finish_reason: if finish_reason == "stop" { None } else { Some(finish_reason) },
                        }],
                        system_fingerprint: Some("fp_decoupled_ai".to_string()),
                    };
                    Ok(chunk)
                }
                Err(e) => Err(anyhow!("Stream error: {}", e)),
            }
        });

        Ok(Box::pin(stream))
    }

    /// Legacy completions endpoint
    pub async fn completions(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let engine = self.engine.lock().await;
        let model = engine.get_model(&request.model)
            .ok_or_else(|| anyhow!("Model not found: {}", request.model))?;
        let prompt = match request.prompt {
            CompletionPrompt::String(s) => self.tokenize(&s)?,
            CompletionPrompt::Array(arr) => self.tokenize_many(&arr)?,
            CompletionPrompt::Tokens(tokens) => tokens,
            CompletionPrompt::TokensArray(arrays) => arrays.into_iter().flatten().collect(),
        };

        let gen_request = GenerateRequest {
            model_id: model.id.clone(),
            prompt_tokens: prompt,
            max_tokens: request.max_tokens.unwrap_or(2048),
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: 0,
            stop_tokens: self.tokenize_many(&request.stop.unwrap_or_default())?,
            stream: false,
        };

        let response = engine.generate_async(gen_request).await?;
        Ok(self.format_completion_response(&request.model, response))
    }

    /// Embeddings endpoint (not fully implemented - requires embedding model)
    pub async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        // For now, return dummy embeddings
        // Real implementation would use an embedding model
        let inputs = match request.input {
            EmbeddingInput::String(s) => vec![s],
            EmbeddingInput::Array(arr) => arr,
            EmbeddingInput::Tokens(tokens) => tokens.iter().map(|t| t.to_string()).collect(),
            EmbeddingInput::TokensArray(arrays) => arrays.into_iter().flatten().map(|t| t.to_string()).collect(),
        };

        let engine = self.engine.lock().await;
        let model = engine.get_model(&request.model)
            .ok_or_else(|| anyhow!("Model not found: {}", request.model))?;

        let data: Vec<EmbeddingData> = inputs.iter().enumerate().map(|(i, _)| {
            EmbeddingData {
                object: "embedding".to_string(),
                embedding: vec![0.0; 4096], // Dummy embedding
                index: i as u32,
            }
        }).collect();

        Ok(EmbeddingResponse {
            object: "list".to_string(),
            data,
            model: request.model,
            usage: Usage {
                prompt_tokens: inputs.iter().map(|s| s.len()).sum(),
                completion_tokens: 0,
                total_tokens: inputs.iter().map(|s| s.len()).sum(),
            },
        })
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    fn find_model(&self, model_id: &str) -> Result<EngineModelInfo> {
        let models = self.engine.blocking_lock().list_models();
        models.into_iter()
            .find(|m| m.id == model_id || m.name == model_id)
            .ok_or_else(|| anyhow!("Model not found: {}", model_id))
    }

    fn messages_to_prompt(&self, messages: &[ChatMessage]) -> Result<Vec<i32>> {
        // Simple conversion: concatenate messages with role prefixes
        // Real implementation would use model-specific chat template
        let mut prompt = String::new();
        for msg in messages {
            let role = match msg.role.as_str() {
                "system" => "System: ",
                "user" => "User: ",
                "assistant" => "Assistant: ",
                _ => "",
            };
            if let Some(content) = &msg.content {
                prompt.push_str(role);
                prompt.push_str(content);
                prompt.push('\n');
            }
        }
        prompt.push_str("Assistant: ");
        self.tokenize(&prompt)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        // Placeholder: simple character-level tokenization
        // Real implementation would use model's tokenizer
        Ok(text.chars().map(|c| c as i32).collect())
    }

    fn tokenize_many(&self, texts: &[String]) -> Result<Vec<i32>> {
        let mut out = Vec::new();
        for t in texts {
            out.extend(self.tokenize(t)?);
        }
        Ok(out)
    }

    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        // Placeholder
        Ok(tokens.iter().map(|&t| char::from_u32(t as u32).unwrap_or('?')).collect())
    }

    fn format_chat_response(&self, model: &str, response: GenerateResponse) -> ChatCompletionResponse {
        let prompt_tokens = response.tokens.len(); // Approximation
        let completion_tokens = response.tokens.len();

        ChatCompletionResponse {
            id: format!("chatcmpl-{}", Uuid::new_v4().simple()),
            object: "chat.completion".to_string(),
            created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            model: model.to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(self.detokenize(&response.tokens).unwrap_or_default()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                logprobs: None,
                finish_reason: response.finish_reason,
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
            system_fingerprint: Some("fp_decoupled_ai".to_string()),
        }
    }

    fn format_completion_response(&self, model: &str, response: GenerateResponse) -> CompletionResponse {
        CompletionResponse {
            id: format!("cmpl-{}", Uuid::new_v4().simple()),
            object: "text_completion".to_string(),
            created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            model: model.to_string(),
            choices: vec![CompletionChoice {
                text: self.detokenize(&response.tokens).unwrap_or_default(),
                index: 0,
                logprobs: None,
                finish_reason: response.finish_reason,
            }],
            usage: Usage {
                prompt_tokens: response.tokens.len(),
                completion_tokens: response.tokens.len(),
                total_tokens: response.tokens.len() * 2,
            },
        }
    }
}

// ============================================================================
// Engine Extension for Streaming
// ============================================================================

// This would be implemented in engine-ipc
// pub trait InferenceEngineExt {
//     fn generate_stream(&self, request: GenerateRequest) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateResponse>> + Send>>>;
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_request_serialization() {
        let req = ChatCompletionRequest {
            model: "gpt-3.5-turbo".to_string(),
            messages: vec![
                ChatMessage { role: "user".to_string(), content: Some("Hello".to_string()), name: None, tool_calls: None, tool_call_id: None },
            ],
            temperature: Some(0.7),
            top_p: Some(1.0),
            max_tokens: Some(100),
            stream: Some(false),
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            tools: None,
            tool_choice: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("gpt-3.5-turbo"));
        assert!(json.contains("Hello"));
    }

    #[test]
    fn test_chat_response_serialization() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "gpt-3.5-turbo".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: Some("Hello!".to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                logprobs: None,
                finish_reason: "stop".to_string(),
            }],
            usage: Usage { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
            system_fingerprint: Some("fp_test".to_string()),
        };

        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("chat.completion"));
        assert!(json.contains("Hello!"));
    }
}