# DeCoupled-AI Agent Interaction Patterns

## Overview

This document defines the interaction patterns for **Hermes Agent** and other autonomous agents integrating with DeCoupled-AI. It covers agent orchestration, tool calling, context management, and multi-agent coordination protocols.

---

## 1. Agent Architecture

### 1.1 Core Agent Interface

```python
class BaseAgent:
    """Base interface for all agents interacting with DeCoupled-AI"""
    
    def __init__(
        self,
        model: str = "llama-3-8b-q4_k_m",
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "sk-decoupled-ai-dev",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: str = None,
        tools: List[Tool] = None
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.conversation_history = []
    
    async def chat(self, messages: List[Message], stream: bool = False) -> Union[str, AsyncGenerator]:
        pass
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        pass
    
    def add_to_history(self, message: Message):
        self.conversation_history.append(message)
```

---

## 2. Hermes Agent Integration Patterns

### 2.1 Basic Hermes Configuration

```yaml
# hermes.yaml
agent:
  name: "hermes-main"
  type: "autonomous"
  
llm:
  provider: "openai"
  model: "llama-3-8b-q4_k_m"
  base_url: "http://localhost:8080/v1"
  api_key: "sk-decoupled-ai-dev"
  temperature: 0.7
  max_tokens: 4096
  top_p: 0.9
  presence_penalty: 0.1
  frequency_penalty: 0.1

memory:
  type: "vector"
  backend: "ruvector"
  collection: "hermes_memory"
  embedding_model: "nomic-embed-text"
  max_context_tokens: 8192

tools:
  - name: "web_search"
    type: "function"
    function:
      name: "search_web"
      description: "Search the web for current information"
      parameters:
        type: "object"
        properties:
          query:
            type: "string"
            description: "Search query"
          max_results:
            type: "integer"
            default: 5
  - name: "code_exec"
    type: "function"
    function:
      name: "execute_code"
      description: "Execute Python code in sandbox"
      parameters:
        type: "object"
        properties:
          code:
            type: "string"
            description: "Python code to execute"
          timeout:
            type: "integer"
            default: 30

orchestration:
  max_iterations: 10
  reflection_interval: 3
  confidence_threshold: 0.8
```

### 2.2 Hermes Tool Calling Loop

```python
class HermesAgent(BaseAgent):
    """Hermes-specific agent with autonomous tool execution"""
    
    def __init__(self, config: HermesConfig):
        super().__init__(
            model=config.llm.model,
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            system_prompt=config.system_prompt,
            tools=config.tools
        )
        self.max_iterations = config.orchestration.max_iterations
        self.reflection_interval = config.orchestration.reflection_interval
        self.tool_executor = ToolExecutor(config.tools)
    
    async def run(self, user_input: str) -> AgentResult:
        """Execute autonomous agent loop"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        for iteration in range(self.max_iterations):
            # 1. Generate response with tool calls
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.openai_tools_schema,
                tool_choice="auto",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            
            message = response.choices[0].message
            messages.append(message.model_dump())
            
            # 2. Execute tool calls if any
            if message.tool_calls:
                tool_results = []
                for tool_call in message.tool_calls:
                    result = await self.tool_executor.execute(
                        tool_call.function.name,
                        json.loads(tool_call.function.arguments)
                    )
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                messages.extend(tool_results)
                
                # Continue loop for next iteration
                continue
            
            # 3. No tool calls - final response
            return AgentResult(
                success=True,
                output=message.content,
                iterations=iteration + 1,
                messages=messages
            )
        
        return AgentResult(
            success=False,
            output="Max iterations reached",
            iterations=self.max_iterations,
            messages=messages
        )
```

### 2.3 Streaming with Tool Calls

```python
async def run_streaming(self, user_input: str) -> AsyncGenerator[str, None]:
    """Stream tokens while handling tool calls"""
    messages = [
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    while True:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.openai_tools_schema,
            tool_choice="auto",
            stream=True
        )
        
        tool_calls_buffer = []
        content_buffer = ""
        
        async for chunk in stream:
            delta = chunk.choices[0].delta
            
            # Handle content streaming
            if delta.content:
                content_buffer += delta.content
                yield delta.content
            
            # Handle tool call streaming
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.index >= len(tool_calls_buffer):
                        tool_calls_buffer.append({
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": "", "arguments": ""}
                        })
                    if tc.function.name:
                        tool_calls_buffer[tc.index]["function"]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_buffer[tc.index]["function"]["arguments"] += tc.function.arguments
            
            # Check for finish
            if chunk.choices[0].finish_reason == "tool_calls":
                # Execute all tool calls
                for tc in tool_calls_buffer:
                    result = await self.tool_executor.execute(
                        tc["function"]["name"],
                        json.loads(tc["function"]["arguments"])
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result)
                    })
                
                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": content_buffer,
                    "tool_calls": tool_calls_buffer
                })
                break  # Restart loop with new messages
            
            elif chunk.choices[0].finish_reason in ("stop", "length"):
                messages.append({
                    "role": "assistant",
                    "content": content_buffer
                })
                return
```

---

## 3. Multi-Agent Orchestration

### 3.1 Swarm Coordinator Pattern

```python
class SwarmCoordinator:
    """Coordinates multiple specialized agents"""
    
    def __init__(self, agents: Dict[str, BaseAgent], router: BaseAgent):
        self.agents = agents  # {"researcher": agent, "coder": agent, "critic": agent}
        self.router = router  # Decides which agent handles each task
        self.shared_memory = SharedMemory()
    
    async def execute(self, task: str) -> SwarmResult:
        # 1. Router analyzes task and creates plan
        plan = await self.router.create_plan(task)
        
        # 2. Execute plan steps with appropriate agents
        results = []
        for step in plan.steps:
            agent = self.agents[step.agent_role]
            agent.shared_memory = self.shared_memory
            
            result = await agent.execute(step.description, context={
                "previous_results": results,
                "shared_memory": self.shared_memory.get_context()
            })
            results.append(result)
            
            # Store in shared memory for other agents
            self.shared_memory.store(f"step_{step.id}", result)
        
        # 3. Synthesize final output
        synthesis = await self.router.synthesize(task, results)
        return SwarmResult(task=task, plan=plan, steps=results, final=synthesis)
```

### 3.2 Agent Role Definitions

```python
AGENT_ROLES = {
    "planner": AgentConfig(
        system_prompt="""You are a strategic planner. Break complex tasks into 
        clear, executable steps. Assign each step to the most appropriate specialist.
        Output JSON: {"steps": [{"id": 1, "agent_role": "researcher", "description": "..."}]}""",
        model="llama-3-8b-q4_k_m",
        temperature=0.3
    ),
    "researcher": AgentConfig(
        system_prompt="""You are a research specialist. Use web_search tool to gather 
        current information. Provide comprehensive, cited findings.""",
        model="llama-3-8b-q4_k_m",
        temperature=0.5,
        tools=["web_search"]
    ),
    "coder": AgentConfig(
        system_prompt="""You are a senior software engineer. Write clean, tested code.
        Use execute_code tool to verify your solutions.""",
        model="llama-3-8b-q4_k_m",
        temperature=0.2,
        tools=["code_exec", "file_ops"]
    ),
    "critic": AgentConfig(
        system_prompt="""You are a critical reviewer. Find bugs, security issues, 
        and improvements. Be thorough but constructive.""",
        model="llama-3-8b-q4_k_m",
        temperature=0.3
    ),
    "synthesizer": AgentConfig(
        system_prompt="""You synthesize outputs from multiple agents into a coherent
        final deliverable. Maintain consistency and completeness.""",
        model="llama-3-8b-q4_k_m",
        temperature=0.5
    )
}
```

---

## 4. Context Management

### 4.1 Sliding Window with Summarization

```python
class ContextManager:
    """Manages conversation context within token limits"""
    
    def __init__(self, max_tokens: int = 8192, summarizer: BaseAgent = None):
        self.max_tokens = max_tokens
        self.summarizer = summarizer
        self.messages = []
        self.summary = ""
    
    def add_message(self, message: Message):
        self.messages.append(message)
        self._maintain_window()
    
    def _maintain_window(self):
        """Keep context within token budget"""
        while self._count_tokens() > self.max_tokens * 0.9:
            if len(self.messages) <= 2:
                break  # Keep system + first user message
            
            # Summarize oldest messages
            if self.summarizer and not self.summary:
                old_messages = self.messages[:3]
                self.summary = await self.summarizer.summarize(old_messages)
                self.messages = [{"role": "system", "content": f"Previous context: {self.summary}"}] + self.messages[3:]
            else:
                # Drop oldest non-system message
                for i, msg in enumerate(self.messages):
                    if msg["role"] != "system":
                        self.messages.pop(i)
                        break
    
    def get_context(self) -> List[Message]:
        return self.messages
```

### 4.2 RAG-Enhanced Context

```python
class RAGContextManager(ContextManager):
    """Context manager with vector retrieval"""
    
    def __init__(self, *args, vector_store: VectorStore, **kwargs):
        super().__init__(*args, **kwargs)
        self.vector_store = vector_store
        self.embedder = EmbeddingModel("nomic-embed-text")
    
    async def get_relevant_context(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant documents from vector store"""
        query_embedding = await self.embedder.embed(query)
        results = await self.vector_store.search(query_embedding, k=k)
        return [r.content for r in results]
    
    async def build_context(self, query: str) -> List[Message]:
        """Build context with retrieved documents"""
        relevant_docs = await self.get_relevant_context(query)
        
        context_messages = self.get_context()
        if relevant_docs:
            context_messages.insert(-1, {  # Before last user message
                "role": "system",
                "content": "Relevant context:\n" + "\n---\n".join(relevant_docs)
            })
        return context_messages
```

---

## 5. Tool Definition Schema (OpenAI Functions)

### 5.1 Standard Tool Format

```python
TOOL_SCHEMAS = {
    "web_search": {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information. Returns top results with snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "recency_days": {
                        "type": "integer",
                        "description": "Limit results to last N days",
                        "default": 365
                    }
                },
                "required": ["query"]
            }
        }
    },
    "execute_code": {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code in a sandboxed environment. Returns stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds",
                        "default": 30,
                        "minimum": 1,
                        "maximum": 300
                    },
                    "packages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional pip packages to install"
                    }
                },
                "required": ["code"]
            }
        }
    },
    "file_read": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "encoding": {"type": "string", "default": "utf-8"}
                },
                "required": ["path"]
            }
        }
    },
    "file_write": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"},
                    "mode": {"type": "string", "enum": ["w", "a"], "default": "w"}
                },
                "required": ["path", "content"]
            }
        }
    }
}
```

### 5.2 Tool Executor

```python
class ToolExecutor:
    """Executes tool calls safely"""
    
    def __init__(self, tools: List[Dict]):
        self.tools = {t["function"]["name"]: t for t in tools}
        self.handlers = {
            "search_web": self._search_web,
            "execute_code": self._execute_code,
            "read_file": self._read_file,
            "write_file": self._write_file
        }
    
    async def execute(self, name: str, args: Dict) -> Any:
        if name not in self.handlers:
            return {"error": f"Unknown tool: {name}"}
        
        try:
            return await self.handlers[name](**args)
        except Exception as e:
            return {"error": str(e), "tool": name}
    
    async def _search_web(self, query: str, max_results: int = 5, recency_days: int = 365) -> Dict:
        # Implementation using your search provider
        pass
    
    async def _execute_code(self, code: str, timeout: int = 30, packages: List[str] = None) -> Dict:
        # Sandbox execution (Docker/nsjail)
        pass
```

---

## 6. Streaming Patterns for Real-Time UX

### 6.1 Token Streaming with Thinking

```python
async def stream_with_thinking(self, messages: List[Message]) -> AsyncGenerator[StreamEvent, None]:
    """Stream tokens with optional thinking display"""
    
    thinking_buffer = ""
    content_buffer = ""
    in_thinking = False
    
    stream = await self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        stream=True
    )
    
    async for chunk in stream:
        delta = chunk.choices[0].delta
        
        # Handle thinking tokens (if model supports <thinking> tags)
        if delta.content:
            if "<thinking>" in delta.content:
                in_thinking = True
                thinking_buffer = ""
                continue
            elif "</thinking>" in delta.content:
                in_thinking = False
                yield StreamEvent(type="thinking_complete", content=thinking_buffer)
                continue
            
            if in_thinking:
                thinking_buffer += delta.content
                yield StreamEvent(type="thinking", content=delta.content)
            else:
                content_buffer += delta.content
                yield StreamEvent(type="content", content=delta.content)
        
        if chunk.choices[0].finish_reason:
            yield StreamEvent(type="done", finish_reason=chunk.choices[0].finish_reason)
            break
```

### 6.2 Progressive Tool Results

```python
async def stream_with_progressive_tools(self, messages: List[Message]):
    """Stream tool execution results progressively"""
    
    while True:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            stream=True
        )
        
        tool_calls = []
        content = ""
        
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
                yield {"type": "content", "data": delta.content}
            if delta.tool_calls:
                tool_calls.extend(delta.tool_calls)
        
        if not tool_calls:
            yield {"type": "done", "content": content}
            break
        
        # Execute tools and stream results
        for tc in tool_calls:
            yield {"type": "tool_start", "tool": tc.function.name, "args": tc.function.arguments}
            
            result = await self.tool_executor.execute(
                tc.function.name,
                json.loads(tc.function.arguments)
            )
            
            yield {"type": "tool_result", "tool": tc.function.name, "result": result}
            
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [tc.model_dump() for tc in tool_calls]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result)
            })
        
        # Continue loop for next iteration
```

---

## 7. Error Handling & Retry Logic

```python
class ResilientAgent(BaseAgent):
    """Agent with built-in retry and error handling"""
    
    def __init__(self, *args, max_retries: int = 3, retry_delay: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    async def chat_with_retry(self, messages: List[Message], **kwargs) -> ChatCompletion:
        for attempt in range(self.max_retries):
            try:
                return await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs
                )
            except RateLimitError:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                raise
            except APIConnectionError:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                raise
            except BadRequestError as e:
                # Don't retry bad requests
                raise AgentError(f"Invalid request: {e}")
        
        raise AgentError("Max retries exceeded")
```

---

## 8. Configuration Reference

### 8.1 Environment Variables

```bash
# DeCoupled-AI Engine
DECOUPLED_AI_HOST=0.0.0.0
DECOUPLED_AI_PORT=8080
DECOUPLED_AI_API_KEY=sk-decoupled-ai-dev
DECOUPLED_AI_MODEL_PATH=/models
DECOUPLED_AI_BACKEND=cuda  # cuda, cpu, metal

# Agent Configuration
HERMES_MODEL=llama-3-8b-q4_k_m
HERMES_TEMPERATURE=0.7
HERMES_MAX_TOKENS=4096
HERMES_SYSTEM_PROMPT_PATH=./prompts/hermes.md
HERMES_MAX_ITERATIONS=10
HERMES_TOOLS=web_search,execute_code,file_read,file_write

# Vector Memory
RUVECTOR_HOST=localhost
RUVECTOR_PORT=6333
RUVECTOR_COLLECTION=hermes_memory
EMBEDDING_MODEL=nomic-embed-text
```

### 8.2 Docker Compose for Full Stack

```yaml
# docker-compose.yml
version: '3.8'

services:
  decoupled-ai:
    image: decoupled-ai:latest
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models:ro
      - ./cache:/cache
    environment:
      - DECOUPLED_AI_BACKEND=cuda
      - DECOUPLED_AI_API_KEY=${DECOUPLED_AI_API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  ruvector:
    image: ruvector:latest
    ports:
      - "6333:6333"
    volumes:
      - ruvector_data:/data

  hermes-agent:
    build: ./hermes-agent
    environment:
      - DECOUPLED_AI_BASE_URL=http://decoupled-ai:8080/v1
      - RUVECTOR_HOST=ruvector
    depends_on:
      - decoupled-ai
      - ruvector

volumes:
  ruvector_data:
```

---

## 9. Testing Patterns

### 9.1 Unit Test for Agent Loop

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_hermes_agent_tool_loop():
    """Test agent executes tools and returns final answer"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        # First call: tool call
        MagicMock(choices=[MagicMock(message=MagicMock(
            tool_calls=[MagicMock(
                id="call_1",
                function=MagicMock(name="search_web", arguments='{"query": "test"}')
            )],
            content=None
        ))]),
        # Second call: final answer
        MagicMock(choices=[MagicMock(message=MagicMock(
            tool_calls=None,
            content="Final answer"
        ))])
    ]
    
    agent = HermesAgent(client=mock_client, tools=[web_search_tool])
    result = await agent.run("Search for test")
    
    assert result.success
    assert result.output == "Final answer"
    assert mock_client.chat.completions.create.call_count == 2
```

### 9.2 Integration Test with Real Engine

```python
@pytest.mark.integration
async def test_full_chat_completion():
    """Test complete chat flow against running engine"""
    async with OpenAI(base_url="http://localhost:8080/v1", api_key="test") as client:
        response = await client.chat.completions.create(
            model="llama-3-8b-q4_k_m",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=50
        )
        
        assert response.choices[0].message.content
        assert response.usage.total_tokens > 0
```

---

## 10. Monitoring & Observability

### 10.1 Agent Metrics

```python
class AgentMetrics:
    """Collect agent performance metrics"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "tool_calls": Counter(),
            "errors": Counter(),
            "latency_ms": Histogram()
        }
    
    def record_request(self, tokens: int, latency: float, tools_used: List[str]):
        self.metrics["total_requests"] += 1
        self.metrics["total_tokens"] += tokens
        self.metrics["latency_ms"].observe(latency)
        for tool in tools_used:
            self.metrics["tool_calls"][tool] += 1
    
    def record_error(self, error_type: str):
        self.metrics["errors"][error_type] += 1
```

---

*Agent Patterns Version: 1.0 | For DeCoupled-AI Engine v1.0+*