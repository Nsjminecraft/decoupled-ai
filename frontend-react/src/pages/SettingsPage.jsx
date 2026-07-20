import { useState, useEffect } from 'react'
import { Save, Loader2, AlertCircle, CheckCircle, Database, Cpu, Globe, HardDrive, Network, Settings } from 'lucide-react'
import { clsx } from 'clsx'

export function SettingsPage() {
  const [config, setConfig] = useState({
    host: '0.0.0.0',
    port: 8080,
    model_dir: './models',
    backend: 'auto',
    api_key: '',
    enable_cors: true,
    max_request_size: 104857600,
    gpu_index: 0,
    gpu_interactive: false,
    auto_update: true,
    auto_install_updates: false,
    update_check_interval: 86400,
    auto_load: true,
  })
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState(null)

  useEffect(() => {
    fetchConfig()
  }, [])

  const fetchConfig = async () => {
    try {
      const res = await fetch('/v1/system/config')
      if (res.ok) {
        const data = await res.json()
        setConfig(data.config || config)
      }
    } catch (e) {
      console.error('Failed to fetch config:', e)
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    setSaving(true)
    setMessage(null)
    try {
      const res = await fetch('/v1/system/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      if (res.ok) {
        setMessage({ type: 'success', text: 'Configuration saved successfully' })
        fetchConfig()
      } else {
        const err = await res.json()
        setMessage({ type: 'error', text: err.detail || 'Failed to save configuration' })
      }
    } catch (e) {
      setMessage({ type: 'error', text: e.message })
    } finally {
      setSaving(false)
    }
  }

  const handleChange = (field, value) => {
    setConfig(prev => ({ ...prev, [field]: value }))
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
      </div>
    )
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-dark-900 dark:text-white flex items-center gap-2">
          <Settings className="w-6 h-6 text-primary-600" />
          Settings
        </h1>
        <p className="text-dark-500 dark:text-dark-400 text-sm mt-1">
          Configure server settings, model storage, and API behavior
        </p>
      </div>

      {message && (
        <div className={clsx(
          'p-4 rounded-lg flex items-center gap-3 animate-slide-up',
          message.type === 'success'
            ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-800'
            : 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 border border-red-200 dark:border-red-800'
        )}>
          {message.type === 'success' ? (
            <CheckCircle className="w-5 h-5 flex-shrink-0" />
          ) : (
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
          )}
          <span>{message.text}</span>
        </div>
      )}

      <div className="card p-6 space-y-6">
        <h2 className="text-lg font-semibold text-dark-900 dark:text-white flex items-center gap-2">
          <Database className="w-5 h-5" />
          Server Configuration
        </h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <label className="block text-sm font-medium text-dark-700 dark:text-dark-300 mb-1">Host</label>
            <input
              type="text"
              value={config.host}
              onChange={(e) => handleChange('host', e.target.value)}
              className="input"
              placeholder="0.0.0.0"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-dark-700 dark:text-dark-300 mb-1">Port</label>
            <input
              type="number"
              value={config.port}
              onChange={(e) => handleChange('port', parseInt(e.target.value) || 8080)}
              className="input"
              min="1"
              max="65535"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-dark-700 dark:text-dark-300 mb-1">Model Directory</label>
            <input
              type="text"
              value={config.model_dir}
              onChange={(e) => handleChange('model_dir', e.target.value)}
              className="input"
              placeholder="./models"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-dark-700 dark:text-dark-300 mb-1">Backend</label>
            <select
              value={config.backend}
              onChange={(e) => handleChange('backend', e.target.value)}
              className="input"
            >
              <option value="auto">Auto-detect</option>
              <option value="cpu">CPU</option>
              <option value="cuda">CUDA</option>
              <option value="rocm">ROCm</option>
              <option value="metal">Metal</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-dark-700 dark:text-dark-300 mb-1">API Key</label>
            <input
              type="password"
              value={config.api_key}
              onChange={(e) => handleChange('api_key', e.target.value)}
              className="input"
              placeholder="Leave empty for no auth"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-dark-700 dark:text-dark-300 mb-1">Max Request Size (bytes)</label>
            <input
              type="number"
              value={config.max_request_size}
              onChange={(e) => handleChange('max_request_size', parseInt(e.target.value) || 104857600)}
              className="input"
              min="1048576"
            />
          </div>
        </div>
      </div>

      <div className="card p-6 space-y-6">
        <h2 className="text-lg font-semibold text-dark-900 dark:text-white flex items-center gap-2">
          <Network className="w-5 h-5" />
          CORS & Network
        </h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="enable_cors"
              checked={config.enable_cors}
              onChange={(e) => handleChange('enable_cors', e.target.checked)}
              className="w-4 h-4 rounded border-dark-300 text-primary-600 focus:ring-primary-500"
            />
            <label htmlFor="enable_cors" className="text-sm font-medium text-dark-700 dark:text-dark-300">
              Enable CORS
            </label>
          </div>
        </div>
      </div>

      <div className="card p-6 space-y-6">
        <h2 className="text-lg font-semibold text-dark-900 dark:text-white flex items-center gap-2">
          <Cpu className="w-5 h-5" />
          GPU Settings
        </h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <label className="block text-sm font-medium text-dark-700 dark:text-dark-300 mb-1">GPU Index</label>
            <input
              type="number"
              value={config.gpu_index}
              onChange={(e) => handleChange('gpu_index', parseInt(e.target.value) || 0)}
              className="input"
              min="0"
            />
          </div>
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="gpu_interactive"
              checked={config.gpu_interactive}
              onChange={(e) => handleChange('gpu_interactive', e.target.checked)}
              className="w-4 h-4 rounded border-dark-300 text-primary-600 focus:ring-primary-500"
            />
            <label htmlFor="gpu_interactive" className="text-sm font-medium text-dark-700 dark:text-dark-300">
              GPU Interactive Mode
            </label>
          </div>
        </div>
      </div>

      <div className="card p-6 space-y-6">
        <h2 className="text-lg font-semibold text-dark-900 dark:text-white flex items-center gap-2">
          <Globe className="w-5 h-5" />
          Auto Updates
        </h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="auto_update"
              checked={config.auto_update}
              onChange={(e) => handleChange('auto_update', e.target.checked)}
              className="w-4 h-4 rounded border-dark-300 text-primary-600 focus:ring-primary-500"
            />
            <label htmlFor="auto_update" className="text-sm font-medium text-dark-700 dark:text-dark-300">
              Auto Check for Updates
            </label>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="auto_install_updates"
              checked={config.auto_install_updates}
              onChange={(e) => handleChange('auto_install_updates', e.target.checked)}
              className="w-4 h-4 rounded border-dark-300 text-primary-600 focus:ring-primary-500"
            />
            <label htmlFor="auto_install_updates" className="text-sm font-medium text-dark-700 dark:text-dark-300">
              Auto Install Updates
            </label>
          </div>
          <div>
            <label className="block text-sm font-medium text-dark-700 dark:text-dark-300 mb-1">Check Interval (seconds)</label>
            <input
              type="number"
              value={config.update_check_interval}
              onChange={(e) => handleChange('update_check_interval', parseInt(e.target.value) || 86400)}
              className="input"
              min="3600"
            />
          </div>
        </div>
      </div>

      <div className="card p-6 space-y-6">
        <h2 className="text-lg font-semibold text-dark-900 dark:text-white flex items-center gap-2">
          <HardDrive className="w-5 h-5" />
          Model Loading
        </h2>
        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="auto_load"
            checked={config.auto_load}
            onChange={(e) => handleChange('auto_load', e.target.checked)}
            className="w-4 h-4 rounded border-dark-300 text-primary-600 focus:ring-primary-500"
          />
          <label htmlFor="auto_load" className="text-sm font-medium text-dark-700 dark:text-dark-300">
            Auto-load first available model on startup
          </label>
        </div>
      </div>

      <div className="flex justify-end gap-3">
        <button
          onClick={fetchConfig}
          disabled={saving}
          className="btn-ghost"
        >
          Reload from Server
        </button>
        <button
          onClick={handleSave}
          disabled={saving}
          className="btn-primary"
        >
          {saving ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin mr-2" />
              Saving...
            </>
          ) : (
            <>
              <Save className="w-4 h-4 mr-2" />
              Save Configuration
            </>
          )}
        </button>
      </div>
    </div>
  )
}