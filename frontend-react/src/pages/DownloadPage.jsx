import { useState, useEffect } from 'react'
import { DownloadCloud, Search, X, Loader2, CheckCircle, AlertCircle, ChevronDown, Database, Globe, Download } from 'lucide-react'
import { clsx } from 'clsx'

export function DownloadPage() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [downloading, setDownloading] = useState({})
  const [progress, setProgress] = useState({})
  const [localModels, setLocalModels] = useState([])
  const [activeTab, setActiveTab] = useState('search')

  useEffect(() => {
    fetchLocalModels()
  }, [])

  const fetchLocalModels = async () => {
    try {
      const res = await fetch('/v1/models')
      const data = await res.json()
      setLocalModels(data.data?.filter(m => m.local) || [])
    } catch (e) {
      console.error('Failed to fetch local models:', e)
    }
  }

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    try {
      const res = await fetch(`/v1/models/search?q=${encodeURIComponent(query)}&limit=20`)
      const data = await res.json()
      setResults(data.models || [])
    } catch (e) {
      console.error('Search failed:', e)
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = async (modelId) => {
    setDownloading(prev => ({ ...prev, [modelId]: true }))
    setProgress(prev => ({ ...prev, [modelId]: 0 }))

    try {
      const res = await fetch('/v1/models/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId }),
      })

      if (!res.ok) throw new Error('Download failed')

      const reader = res.body.getReader()
      const contentLength = +res.headers.get('Content-Length') || 0
      let received = 0

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        received += value.length
        setProgress(prev => ({
          ...prev,
          [modelId]: contentLength ? Math.round((received / contentLength) * 100) : 0
        }))
      }

      await fetchLocalModels()
      setResults(prev => prev.map(m => m.id === modelId ? { ...m, local: true } : m))
    } catch (e) {
      alert(`Download failed: ${e.message}`)
    } finally {
      setDownloading(prev => ({ ...prev, [modelId]: false }))
      setProgress(prev => ({ ...prev, [modelId]: 0 }))
    }
  }

  const handleDelete = async (modelId) => {
    if (!confirm(`Delete ${modelId}?`)) return

    try {
      await fetch(`/v1/models/${modelId}`, { method: 'DELETE' })
      await fetchLocalModels()
    } catch (e) {
      alert(`Delete failed: ${e.message}`)
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-dark-900 dark:text-white flex items-center gap-2">
          <DownloadCloud className="w-6 h-6 text-primary-600" />
          Download Models
        </h1>
        <p className="text-dark-500 dark:text-dark-400 text-sm mt-1">
          Search and download models from Hugging Face and other registries
        </p>
      </div>

      <div className="flex gap-2 border-b border-dark-200 dark:border-dark-700">
        {['search', 'local'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={clsx(
              'px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors',
              activeTab === tab
                ? 'border-primary-600 text-primary-600 dark:text-primary-400'
                : 'border-transparent text-dark-500 dark:text-dark-400 hover:text-dark-700 dark:hover:text-dark-200'
            )}
          >
            {tab === 'search' ? 'Search & Download' : 'Local Models'}
          </button>
        ))}
      </div>

      {activeTab === 'search' && (
        <SearchTab
          query={query}
          setQuery={setQuery}
          results={results}
          loading={loading}
          downloading={downloading}
          progress={progress}
          onSearch={handleSearch}
          onDownload={handleDownload}
        />
      )}

      {activeTab === 'local' && (
        <LocalTab
          models={localModels}
          downloading={downloading}
          progress={progress}
          onDownload={handleDownload}
          onDelete={handleDelete}
        />
      )}
    </div>
  )
}

function SearchTab({ query, setQuery, results, loading, downloading, progress, onSearch, onDownload }) {
  const popularModels = [
    { id: 'meta-llama/Llama-3.2-3B-Instruct-GGUF', description: 'Llama 3.2 3B Instruct (GGUF)', size: '2.0 GB' },
    { id: 'microsoft/Phi-3.5-mini-instruct-GGUF', description: 'Phi 3.5 Mini Instruct (GGUF)', size: '2.2 GB' },
    { id: 'Qwen/Qwen2.5-7B-Instruct-GGUF', description: 'Qwen 2.5 7B Instruct (GGUF)', size: '4.4 GB' },
    { id: 'google/gemma-2-2b-it-GGUF', description: 'Gemma 2 2B IT (GGUF)', size: '1.6 GB' },
    { id: 'microsoft/Phi-3-mini-4k-instruct-GGUF', description: 'Phi 3 Mini 4K Instruct (GGUF)', size: '2.3 GB' },
  ]

  return (
    <div className="space-y-6">
      <form onSubmit={onSearch} className="card p-4">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-400" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search Hugging Face models (e.g., llama-3.2, phi-3, qwen2.5)..."
              className="input pl-10"
            />
          </div>
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="btn-primary whitespace-nowrap"
          >
            {loading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              'Search'
            )}
          </button>
        </div>
      </form>

      {query.trim() === '' && results.length === 0 && (
        <div className="card p-6">
          <h3 className="font-medium text-dark-900 dark:text-white mb-4">Popular Models</h3>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {popularModels.map(model => (
              <PopularModelCard key={model.id} model={model} onDownload={onDownload} downloading={downloading} progress={progress} />
            ))}
          </div>
        </div>
      )}

      {results.length > 0 && (
        <div className="space-y-3">
          <h3 className="font-medium text-dark-900 dark:text-white">Search Results ({results.length})</h3>
          <div className="space-y-3">
            {results.map(model => (
              <SearchResultCard
                key={model.id}
                model={model}
                downloading={downloading[model.id]}
                progress={progress[model.id] || 0}
                onDownload={onDownload}
              />
            ))}
          </div>
        </div>
      )}

      {query.trim() !== '' && !loading && results.length === 0 && (
        <div className="card p-12 text-center">
          <Search className="w-16 h-16 mx-auto mb-4 text-dark-300 dark:text-dark-600" />
          <h3 className="text-lg font-medium text-dark-900 dark:text-white mb-2">No results found</h3>
          <p className="text-dark-500 dark:text-dark-400">Try a different search term</p>
        </div>
      )}
    </div>
  )
}

function PopularModelCard({ model, onDownload, downloading, progress }) {
  const isDownloading = downloading[model.id]
  const modelProgress = progress[model.id] || 0

  return (
    <div className="card p-4 hover:shadow-lg transition-shadow">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <h4 className="font-medium text-dark-900 dark:text-white truncate">{model.id.split('/').pop()}</h4>
          <p className="text-sm text-dark-500 dark:text-dark-400 mt-1">{model.description}</p>
          <span className="badge badge-info mt-2 inline-block">{model.size}</span>
        </div>
      </div>
      <div className="mt-3 flex gap-2">
        {isDownloading ? (
          <div className="flex-1 flex items-center gap-2">
            <div className="flex-1 h-2 bg-dark-200 dark:bg-dark-700 rounded-full overflow-hidden">
              <div className="h-full bg-primary-600 transition-all duration-300" style={{ width: `${modelProgress}%` }} />
            </div>
            <span className="text-sm text-dark-500 w-10 text-right">{modelProgress}%</span>
          </div>
        ) : (
          <button
            onClick={() => onDownload(model.id)}
            className="btn-primary flex-1"
          >
            <Download className="w-4 h-4 mr-1" /> Download
          </button>
        )}
      </div>
    </div>
  )
}

function SearchResultCard({ model, downloading, progress, onDownload }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="card p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h4 className="font-medium text-dark-900 dark:text-white">{model.id}</h4>
            {model.local && <CheckCircle className="w-4 h-4 text-green-500" title="Already downloaded" />}
          </div>
          <p className="text-sm text-dark-500 dark:text-dark-400 mt-1 line-clamp-2">{model.description || 'No description'}</p>

          {expanded && (
            <div className="mt-3 flex flex-wrap gap-2 text-xs">
              {model.tags?.map(tag => <span key={tag} className="badge">{tag}</span>)}
              {model.size && <span className="badge badge-info">{(model.size / (1024**3)).toFixed(1)} GB</span>}
              {model.architecture && <span className="badge">{model.architecture}</span>}
              {model.quantization && <span className="badge badge-warning">{model.quantization}</span>}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2 flex-shrink-0">
          {downloading ? (
            <div className="w-48 flex items-center gap-2">
              <div className="flex-1 h-2 bg-dark-200 dark:bg-dark-700 rounded-full overflow-hidden">
                <div className="h-full bg-primary-600 transition-all duration-300" style={{ width: `${progress}%` }} />
              </div>
              <span className="text-sm text-dark-500 w-10 text-right">{progress}%</span>
            </div>
          ) : model.local ? (
            <span className="badge badge-success">Downloaded</span>
          ) : (
            <button
              onClick={() => onDownload(model.id)}
              className="btn-primary whitespace-nowrap"
            >
              <Download className="w-4 h-4 mr-1" /> Download
            </button>
          )}
          <button
            onClick={() => setExpanded(!expanded)}
            className="btn-ghost p-2"
          >
            <ChevronDown className={clsx('w-4 h-4', expanded && 'rotate-180')} />
          </button>
        </div>
      </div>
    </div>
  )
}

function LocalTab({ models, downloading, progress, onDownload, onDelete }) {
  if (models.length === 0) {
    return (
      <div className="card p-12 text-center">
        <Database className="w-16 h-16 mx-auto mb-4 text-dark-300 dark:text-dark-600" />
        <h3 className="text-lg font-medium text-dark-900 dark:text-white mb-2">No local models</h3>
        <p className="text-dark-500 dark:text-dark-400 mb-4">Download models from the Search tab to get started</p>
        <button onClick={() => document.querySelector('[data-tab="search"]')?.click()} className="btn-primary">
          <Globe className="w-4 h-4 mr-1" /> Browse Models
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <h3 className="font-medium text-dark-900 dark:text-white">Local Models ({models.length})</h3>
      <div className="space-y-3">
        {models.map(model => (
          <LocalModelCard
            key={model.id}
            model={model}
            downloading={downloading[model.id]}
            progress={progress[model.id] || 0}
            onDelete={onDelete}
          />
        ))}
      </div>
    </div>
  )
}

function LocalModelCard({ model, downloading, progress, onDelete }) {
  const sizeStr = model.size ? (model.size / (1024**3)).toFixed(1) + ' GB' : 'Unknown size'

  return (
    <div className="card p-4 flex items-center justify-between gap-4">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <h4 className="font-medium text-dark-900 dark:text-white truncate">{model.id}</h4>
          <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
        </div>
        <div className="flex flex-wrap gap-2 mt-1 text-sm text-dark-500 dark:text-dark-400">
          <span className="badge">{sizeStr}</span>
          {model.architecture && <span className="badge">{model.architecture}</span>}
          {model.quantization && <span className="badge badge-warning">{model.quantization}</span>}
          {model.context_length && <span className="badge">{model.context_length.toLocaleString()} ctx</span>}
        </div>
      </div>

      <div className="flex items-center gap-2 flex-shrink-0">
        {downloading && (
          <div className="w-40 flex items-center gap-2">
            <div className="flex-1 h-2 bg-dark-200 dark:bg-dark-700 rounded-full overflow-hidden">
              <div className="h-full bg-primary-600 transition-all duration-300" style={{ width: `${progress}%` }} />
            </div>
            <span className="text-sm text-dark-500 w-10 text-right">{progress}%</span>
          </div>
        )}
        <button
          onClick={() => onDelete(model.id)}
          className="btn-ghost text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20"
          title="Delete model"
          disabled={downloading}
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}