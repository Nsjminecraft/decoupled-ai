import { useState, useRef, useEffect } from 'react'
import { Send, Copy, Check, Loader2, Trash2, Sparkles, MessageSquare } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { clsx } from 'clsx'

const API_BASE = ''

export function ChatPage() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [model, setModel] = useState(null)
  const [models, setModels] = useState([])
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(2048)
  const [stream, setStream] = useState(true)
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    fetchModels()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_BASE}/v1/models`)
      const data = await res.json()
      setModels(data.data || [])
      if (data.data?.length > 0 && !model) {
        setModel(data.data[0].id)
      }
    } catch (e) {
      console.error('Failed to fetch models:', e)
    }
  }

  const handleSend = async (e) => {
    e.preventDefault()
    if (!input.trim() || streaming || !model) return

    const userMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setStreaming(true)

    try {
      const res = await fetch(`${API_BASE}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('apiKey') || 'sk-decoupled-ai-dev'}`,
        },
        body: JSON.stringify({
          model,
          messages: [...messages, userMessage],
          temperature,
          max_tokens: maxTokens,
          stream,
        }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      if (stream) {
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let assistantMessage = { role: 'assistant', content: '' }
        const messageIndex = messages.length + 1
        setMessages(prev => [...prev, assistantMessage])

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value)
          const lines = chunk.split('\n')

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6)
              if (data === '[DONE]') continue
              try {
                const parsed = JSON.parse(data)
                const delta = parsed.choices?.[0]?.delta?.content
                if (delta) {
                  setMessages(prev => {
                    const newMsgs = [...prev]
                    newMsgs[messageIndex] = {
                      ...newMsgs[messageIndex],
                      content: newMsgs[messageIndex].content + delta
                    }
                    return newMsgs
                  })
                }
              } catch (e) {}
            }
          }
        }
      } else {
        const data = await res.json()
        const assistantMessage = {
          role: 'assistant',
          content: data.choices?.[0]?.message?.content || 'No response'
        }
        setMessages(prev => [...prev, assistantMessage])
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${error.message}`,
        error: true
      }])
    } finally {
      setStreaming(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend(e)
    }
  }

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text)
  }

  const clearChat = () => {
    setMessages([])
  }

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto w-full">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-dark-900 dark:text-white flex items-center gap-2">
          <Sparkles className="w-6 h-6 text-primary-600" />
          Chat
        </h1>
        <p className="text-dark-500 dark:text-dark-400 text-sm mt-1">
          Chat with your local models via OpenAI-compatible API
        </p>
      </div>

      <div className="flex-1 overflow-y-auto space-y-4 pb-4 pr-2">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-center text-dark-400 dark:text-dark-500">
            <MessageSquare className="w-16 h-16 mb-4 opacity-50" />
            <h3 className="text-lg font-medium mb-2">Start a conversation</h3>
            <p className="text-sm max-w-md">Select a model and start chatting. Your conversation stays local.</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <MessageBubble
              key={idx}
              message={msg}
              isStreaming={streaming && idx === messages.length - 1 && msg.role === 'assistant'}
              onCopy={copyToClipboard}
            />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="border-t border-dark-200 dark:border-dark-700 p-4 bg-white/50 dark:bg-dark-900/50 backdrop-blur-sm">
        <div className="flex items-center gap-3 mb-3">
          <select
            value={model || ''}
            onChange={(e) => setModel(e.target.value)}
            disabled={streaming || models.length === 0}
            className="input flex-1 max-w-xs"
            aria-label="Select model"
          >
            <option value="">Select a model...</option>
            {models.map(m => (
              <option key={m.id} value={m.id}>{m.id}</option>
            ))}
          </select>

          <div className="flex items-center gap-2 text-sm text-dark-500 dark:text-dark-400">
            <label className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={stream}
                onChange={(e) => setStream(e.target.checked)}
                disabled={streaming}
                className="w-4 h-4 rounded border-dark-300 text-primary-600 focus:ring-primary-500"
              />
              Stream
            </label>
            <label className="flex items-center gap-1">
              Temp: {temperature.toFixed(1)}
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                disabled={streaming}
                className="w-24 h-1 bg-dark-200 dark:bg-dark-700 rounded-lg appearance-none cursor-pointer accent-primary-600"
              />
            </label>
            <label className="flex items-center gap-1">
              Max: {maxTokens}
              <input
                type="range"
                min="100"
                max="8192"
                step="100"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                disabled={streaming}
                className="w-24 h-1 bg-dark-200 dark:bg-dark-700 rounded-lg appearance-none cursor-pointer accent-primary-600"
              />
            </label>
          </div>
        </div>

        <form onSubmit={handleSend} className="flex gap-2">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={streaming || !model}
            placeholder={!model ? 'Select a model first...' : streaming ? 'Generating...' : 'Type a message... (Shift+Enter for new line)'}
            rows={1}
            className="flex-1 input resize-none min-h-[44px] max-h-48"
            style={{ height: 'auto' }}
            aria-label="Chat input"
          />
          <button
            type="submit"
            disabled={streaming || !input.trim() || !model}
            className="btn-primary self-end mb-1 px-6"
            aria-label="Send message"
          >
            {streaming ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
              </>
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </form>

        {messages.length > 0 && (
          <button
            onClick={clearChat}
            className="btn-ghost w-full mt-2 text-dark-500 dark:text-dark-400 hover:text-red-600 dark:hover:text-red-400"
          >
            <Trash2 className="w-4 h-4 mr-2" />
            Clear Chat
          </button>
        )}
      </div>
    </div>
  )
}

function MessageBubble({ message, isStreaming, onCopy }) {
  const [copied, setCopied] = useState(false)

  const renderContent = () => {
    if (message.error) {
      return <div className="text-red-500 dark:text-red-400">{message.content}</div>
    }

    return (
      <ReactMarkdown
        components={{
          code: ({ children, ...props }) => {
            const code = String(children).trim()
            const language = props.className?.replace('language-', '') || 'text'
            return (
              <div className="relative group my-2">
                <button
                  onClick={() => {
                    onCopy(code)
                    setCopied(true)
                    setTimeout(() => setCopied(false), 2000)
                  }}
                  className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded text-dark-400 hover:text-dark-600 dark:hover:text-dark-200"
                  aria-label="Copy code"
                >
                  {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                </button>
                <SyntaxHighlighter language={language} style={oneDark} customStyle={{ margin: 0 }}>
                  {code}
                </SyntaxHighlighter>
              </div>
            )
          },
          pre: ({ children }) => children,
        }}
      >
        {message.content}
      </ReactMarkdown>
    )
  }

  return (
    <div className={clsx('flex gap-3 animate-fade-in', message.role === 'user' && 'flex-row-reverse')}>
      <div
        className={clsx(
          'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0',
          message.role === 'user'
            ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400'
            : 'bg-dark-100 dark:bg-dark-800 text-dark-600 dark:text-dark-400'
        )}
        aria-hidden="true"
      >
        {message.role === 'user' ? (
          <MessageSquare className="w-4 h-4" />
        ) : (
          <Sparkles className="w-4 h-4" />
        )}
      </div>

      <div
        className={clsx(
          'max-w-[85%] rounded-2xl px-4 py-3',
          message.role === 'user'
            ? 'bg-primary-600 text-white rounded-br-md'
            : 'bg-white dark:bg-dark-800 border border-dark-200 dark:border-dark-700 rounded-bl-md shadow-sm'
        )}
      >
        {renderContent()}
        {isStreaming && <span className="inline-block w-2 h-2 bg-current opacity-50 animate-pulse ml-1" />}
      </div>
    </div>
  )
}