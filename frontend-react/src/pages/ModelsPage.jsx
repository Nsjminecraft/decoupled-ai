import { useState, useEffect } from 'react'
import { Download, Trash2, Search, Box, CheckCircle, AlertCircle, Loader2, Cpu, Zap } from 'lucide-react'
import { clsx } from 'clsx'

export function ModelsPage() {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('all')
  const [downloading, setDownloading] = useState(null)
  const [downloadProgress, setDownloadProgress] = useState({})

  useEffect(() => {
    fetchModels()
  }, [])

  const fetchModels = async () => {
    try {
      const res = await fetch('/v1/models')
      const data = await res.json()
      setModels(data.data || [])
    } catch (e) {
      console.error('Failed to fetch models:', e)
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = async (modelId) => {
    setDownloading(modelId)
    setDownloadProgress(prev => ({ ...prev, [modelId]: 0 }))

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
        setDownloadProgress(prev => ({
          ...prev,
          [modelId]: contentLength ? Math.round((received / contentLength) * 100) : 0
        }))
      }

      await fetchModels()
    } catch (e) {
      alert(`Failed to download: ${e.message}`)
    } finally {
      setDownloading(null)
      setDownloadProgress(prev => ({ ...prev, [modelId]: 0 }))
    }
  }

  const handleDelete = async (modelId) => {
    if (!confirm(`Delete model ${modelId}?`)) return

    try {
      await fetch(`/v1/models/${modelId}`, { method: 'DELETE' })
      await fetchModels()
    } catch (e) {
      alert(`Failed to delete: ${e.message}`)
    }
  }

  const filteredModels = models.filter(m => {
    if (filter === 'local') return m.local
    if (filter === 'remote') return !m.local
    return true
  })

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-dark-900 dark:text-white flex items-center gap-2">
            <Box className="w-6 h-6 text-primary-600" />
            Models
          </h1>
          <p className="text-dark-500 dark:text-dark-400 text-sm mt-1">
            Manage your local and remote models
          </p>
        </div>
        <button className="btn-primary" onClick={() => fetchModels()} disabled={loading}>
          <Loader2 className={clsx('w-4 h-4', loading && 'animate-spin')} />
          Refresh
        </button>
      </div>

      <div className="flex gap-2 mb-4">
        {['all', 'local', 'remote'].map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={clsx(
              'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
              filter === f
                ? 'bg-primary-600 text-white'
                : 'bg-dark-100 dark:bg-dark-800 text-dark-600 dark:text-dark-400 hover:bg-dark-200 dark:hover:bg-dark-700'
            )}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
        </div>
      ) : filteredModels.length === 0 ? (
        <div className="card p-12 text-center">
          <Box className="w-16 h-16 mx-auto mb-4 text-dark-300 dark:text-dark-600" />
          <h3 className="text-lg font-medium text-dark-900 dark:text-white mb-2">
            No models found
          </h3>
          <p className="text-dark-500 dark:text-dark-400 mb-4">
            {filter === 'all' ? 'Download a model to get started' : `No ${filter} models available`}
          </p>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredModels.map(model => (
            <ModelCard
              key={model.id}
              model={model}
              downloading={downloading === model.id}
              progress={downloadProgress[model.id] || 0}
              onDownload={handleDownload}
              onDelete={handleDelete}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function ModelCard({ model, downloading, progress, onDownload, onDelete }) {
  const [expanded, setExpanded] = useState(false)

  const sizeStr = model.size
    ? (model.size / (1024 ** 3)).toFixed(1) + ' GB'
    : 'Unknown'

  return (
    <div className="card p-4 hover:shadow-lg transition-shadow">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2">
            <h3 className="font-medium text-dark-900 dark:text-white truncate">{model.id}</h3>
            {model.local && (
              <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" title="Downloaded locally" />
            )}
            {!model.local && (
              <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0" title="Not downloaded" />
            )}
          </div>
          <p className="text-sm text-dark-500 dark:text-dark-400 mb-2 line-clamp-2">
            {model.description || 'No description available'}
          </p>

          <div className="flex flex-wrap gap-2 text-xs">
            <span className="badge badge-info">{model.architecture || 'Unknown'}</span>
            <span className="badge">{sizeStr}</span>
            {model.context_length && (
              <span className="badge">{model.context_length.toLocaleString()} ctx</span>
            )}
            {model.quantization && (
              <span className="badge badge-warning">{model.quantization}</span>
            )}
          </div>
        </div>
      </div>

      {expanded && (
        <div className="mt-4 pt-4 border-t border-dark-200 dark:border-dark-700 space-y-3 animate-slide-up">
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div><span className="text-dark-500 dark:text-dark-400">Format:</span> <span className="text-dark-900 dark:text-white ml-1">{model.format || 'GGUF'}</span></div>
            <div><span className="text-dark-500 dark:text-dark-400">Backend:</span> <span className="text-dark-900 dark:text-white ml-1">{model.backend || 'auto'}</span></div>
            {model.license && <div className="col-span-2"><span className="text-dark-500 dark:text-dark-400">License:</span> <span className="text-dark-900 dark:text-white ml-1">{model.license}</span></div>}
            {model.tags && <div className="col-span-2"><span className="text-dark-500 dark:text-dark-400">Tags:</span> <span className="text-dark-900 dark:text-white ml-1">{model.tags.join(', ')}</span></div>}
          </div>

          <div className="flex gap-2">
            {!model.local && !downloading && (
              <button
                onClick={() => onDownload(model.id)}
                className="btn-primary flex-1"
              >
                <Download className="w-4 h-4 mr-1" /> Download
              </button>
            )}
            {downloading && (
              <div className="flex-1 flex items-center gap-2">
                <div className="flex-1 h-2 bg-dark-200 dark:bg-dark-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary-600 transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <span className="text-sm text-dark-500 w-10 text-right">{progress}%</span>
              </div>
            )}
            {model.local && (
              <button
                onClick={() => onDelete(model.id)}
                className="btn-danger"
                title="Delete model"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            )}
            <button
              onClick={() => setExpanded(!expanded)}
              className="btn-ghost"
              aria-label={expanded ? 'Collapse' : 'Expand'}
            >
              <Cpu className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full mt-3 btn-ghost justify-center text-xs"
      >
        {expanded ? 'Hide details' : 'Show details'}
      </button>
    </div>
  )
}