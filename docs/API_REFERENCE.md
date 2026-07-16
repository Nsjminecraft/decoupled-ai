# DeCoupled-AI OpenAI-Compatible API Reference

## Overview

The DeCoupled-AI engine exposes a REST API fully compatible with OpenAI's v1 specification, enabling drop-in integration with frameworks like **Hermes Agent**, **LangChain**, **AutoGPT**, and any OpenAI SDK client.

**Base URL:** `http://localhost:8080/v1` (configurable via `--port` flag)

**Content-Type:** `application/json` (request/response)
**Streaming:** `text/event-stream` (SSE for chat completions)

---

## Endpoints

### 1. List Models
**GET** `/v1/models`

Returns available `.brain` models loaded in the engine.

#### Request
```bash
curl -X GET http://localhost:8080/v1/models \
  -H "Authorization: Bearer <api_key>"
```

#### Response (200 OK)
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-3-8b-q4_k_m",
      "object": "model",
      "created": 1704067200,
      "owned_by": "local",
      "permission": [],
      "root": "llama-3-8b-q4_k_m",
      "parent": null,
      "metadata": {
        "parameter_count": "8B",
        "quantization": "q4_k_m",
        "context_length": 8192,
        "architecture": "llama"
      }
    },
    {
      "id": "mistral-7b-v0.3-q4_k_m",
      "object": "model",
      "created": 1704067200,
      "owned_by": "local",
      "permission": [],
      "root": "mistral-7b-v0.3-q4_k_m",
      "parent": null,
      "metadata": {
        "parameter_count": "7B",
        "quantization": "q4_k_m",
        "context_length": 32768,
        "architecture": "mistral"
      }
    }
  ]
}
```

---

### 2. Retrieve Model
**GET** `/v1/models/{model_id}`

Returns details for a specific model.

#### Request
```bash
curl -X GET http://localhost:8080/v1/models/llama-3-8b-q4_k_m \
  -H "Authorization: Bearer <api_key>"
```

#### Response (200 OK)
```json
{
  "id": "llama-3-8b-q4_k_m",
  "object": "model",
  "created": 1704067200,
  "owned_by": "local",
  "permission": [],
  "root": "llama-3-8b-q4_k_m",
  "parent": null,
  "metadata": {
    "parameter_count": "8B",
    "quantization": "q4_k_m",
    "context_length": 8192,
    "architecture": "llama",
    "file_path": "/models/llama-3-8b-q4_k_m.brain",
    "memory_mapped": true,
    "compute_backend": "cuda"
  }
}
```

---

### 3. Chat Completions (Non-Streaming)
**POST** `/v1/chat/completions`

Standard OpenAI chat completions endpoint.

#### Request
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <api_key>" \
  -d '{
    "model": "llama-3-8b-q4_k_m",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum entanglement in simple terms."}
    ],
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "stream": false,
    "stop": ["\n\n"],
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "logit_bias": {},
    "user": "user-123"
  }'
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Yes | - | Model ID from `/v1/models` |
| `messages` | array | Yes | - | Array of message objects |
| `messages[].role` | enum | Yes | - | `system`, `user`, `assistant`, `tool` |
| `messages[].content` | string | Yes | - | Message content |
| `messages[].name` | string | No | - | Optional name for role |
| `messages[].tool_calls` | array | No | - | Tool calls (assistant messages) |
| `messages[].tool_call_id` | string | No | - | For tool response messages |
| `temperature` | float | No | 1.0 | Sampling temperature (0.0-2.0) |
| `top_p` | float | No | 1.0 | Nucleus sampling |
| `max_tokens` | int | No | inf | Maximum tokens to generate |
| `stream` | bool | No | false | Enable SSE streaming |
| `stop` | string[] | No | null | Stop sequences |
| `presence_penalty` | float | No | 0.0 | -2.0 to 2.0 |
| `frequency_penalty` | float | No | 0.0 | -2.0 to 2.0 |
| `logit_bias` | object | No | {} | Token bias map |
| `user` | string | No | - | User identifier for abuse tracking |

#### Response (200 OK)
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "llama-3-8b-q4_k_m",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quantum entanglement is a phenomenon where two particles become linked...",
        "tool_calls": null
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 156,
    "total_tokens": 198
  },
  "system_fingerprint": "fp_decoupled_ai_v1"
}
```

---

### 4. Chat Completions (Streaming)
**POST** `/v1/chat/completions` with `"stream": true`

Returns Server-Sent Events (SSE) stream.

#### Request
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <api_key>" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "llama-3-8b-q4_k_m",
    "messages": [{"role": "user", "content": "Count to 10"}],
    "stream": true,
    "max_tokens": 100
  }' \
  --no-buffer
```

#### Response Stream (SSE Format)
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1704067200,"model":"llama-3-8b-q4_k_m","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1704067200,"model":"llama-3-8b-q4_k_m","choices":[{"index":0,"delta":{"content":"1"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1704067200,"model":"llama-3-8b-q4_k_m","choices":[{"index":0,"delta":{"content":" 2"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1704067200,"model":"llama-3-8b-q4_k_m","choices":[{"index":0,"delta":{"content":" 3"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1704067200,"model":"llama-3-8b-q4_k_m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

#### Chunk Format
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "created": 1704067200,
  "model": "llama-3-8b-q4_k_m",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "token fragment"
      },
      "finish_reason": null
    }
  ]
}
```

**Final chunk:** `finish_reason` = `"stop"` | `"length"` | `"tool_calls"` | `"content_filter"`

**Termination:** `data: [DONE]\n\n`

---

### 5. Completions (Legacy / Text)
**POST** `/v1/completions`

Legacy text completion endpoint (GPT-3 style).

#### Request
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <api_key>" \
  -d '{
    "model": "llama-3-8b-q4_k_m",
    "prompt": "The capital of France is",
    "max_tokens": 50,
    "temperature": 0.7,
    "stream": false
  }'
```

#### Response
```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1704067200,
  "model": "llama-3-8b-q4_k_m",
  "choices": [
    {
      "text": " Paris, known for the Eiffel Tower and...",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 32,
    "total_tokens": 40
  }
}
```

---

### 6. Embeddings
**POST** `/v1/embeddings`

Generate embeddings (if model supports it).

#### Request
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <api_key>" \
  -d '{
    "model": "llama-3-8b-q4_k_m",
    "input": "Hello world",
    "encoding_format": "float"
  }'
```

#### Response
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0023, -0.0145, ...],
      "index": 0
    }
  ],
  "model": "llama-3-8b-q4_k_m",
  "usage": {
    "prompt_tokens": 2,
    "total_tokens": 2
  }
}
```

---

### 7. Health Check
**GET** `/health`

Engine health status (no auth required).

#### Request
```bash
curl http://localhost:8080/health
```

#### Response
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "loaded_models": 2,
  "active_backend": "cuda",
  "memory_usage": {
    "vram_used_mb": 4096,
    "vram_total_mb": 8192,
    "ram_used_mb": 1024
  }
}
```

---

## Authentication

All `/v1/*` endpoints require Bearer token authentication (configurable):

```bash
# Default: no auth required in development
# Production: set DECOUPLED_AI_API_KEY environment variable
Authorization: Bearer sk-decoupled-ai-xxxxx
```

---

## Error Responses

All errors follow OpenAI format:

```json
{
  "error": {
    "message": "Model not found: unknown-model",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

| HTTP Code | Error Type | Description |
|-----------|------------|-------------|
| 400 | `invalid_request_error` | Malformed request |
| 401 | `authentication_error` | Invalid/missing API key |
| 404 | `not_found_error` | Model/endpoint not found |
| 429 | `rate_limit_error` | Too many requests |
| 500 | `server_error` | Internal engine error |
| 503 | `service_unavailable` | Engine starting/overloaded |

---

## Integration Examples

### Hermes Agent Configuration
```yaml
# hermes-config.yaml
llm:
  provider: "openai"
  model: "llama-3-8b-q4_k_m"
  base_url: "http://localhost:8080/v1"
  api_key: "sk-decoupled-ai-dev"
  temperature: 0.7
  max_tokens: 2048
```

### Python (OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-decoupled-ai-dev"
)

# Non-streaming
response = client.chat.completions.create(
    model="llama-3-8b-q4_k_m",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="llama-3-8b-q4_k_m",
    messages=[{"role": "user", "content": "Count to 10"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### JavaScript (OpenAI SDK)
```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "sk-decoupled-ai-dev",
  dangerouslyAllowBrowser: true
});

// Streaming
for await (const chunk of await client.chat.completions.create({
  model: "llama-3-8b-q4_k_m",
  messages: [{ role: "user", content: "Hello!" }],
  stream: true
})) {
  process.stdout.write(chunk.choices[0]?.delta?.content || "");
}
```

### cURL Streaming with jq
```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"model":"llama-3-8b-q4_k_m","messages":[{"role":"user","content":"Hello"}],"stream":true}' | \
  while IFS= read -r line; do
    if [[ $line == data:* ]]; then
      data="${line#data: }"
      if [[ $data != "[DONE]" ]]; then
        echo "$data" | jq -r '.choices[0].delta.content // empty' | tr -d '\n'
      fi
    fi
  done
echo
```

---

## Rate Limits (Default)

| Endpoint | Requests/min | Tokens/min |
|----------|--------------|------------|
| `/v1/chat/completions` | 60 | 100,000 |
| `/v1/completions` | 60 | 100,000 |
| `/v1/embeddings` | 120 | 500,000 |
| `/v1/models` | 600 | - |

Configurable via `DECOUPLED_AI_RATE_LIMIT_*` environment variables.

---

## Model Hot-Swapping

Models can be loaded/unloaded at runtime via the dashboard or API:

```bash
# Load a model
curl -X POST http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"path": "/models/new-model.brain", "id": "new-model"}'

# Unload a model
curl -X POST http://localhost:8080/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"id": "old-model"}'
```

---

*API Version: 1.0 | Compatible with OpenAI API v1.0.0+*