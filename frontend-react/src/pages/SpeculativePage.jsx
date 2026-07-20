import { useState, useEffect } from 'react'
import { Zap, Cpu, Gauge, Target, Loader2, AlertCircle, CheckCircle, Info } from 'lucide-react'
import { clsx } from 'clsx'

export function SpeculativePage() {
  const [config, setConfig] = useState({
    enabled: false,
    draft_model: '',
    target_model: '',
    num_draft_tokens: 5,
    temperature: 0.7,
    max_tokens: 2048,
  })
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null)
  const [stats, setStats] = useState(null)

  useEffect(() => {
    fetchModels()
    fetchStats()
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

  const fetchStats = async () => {
    try {
      const res = await fetch('/v1/speculative/stats')
      if (res.ok) {
        const data = await res.json()
        setStats(data)
      }
    } catch (e) {
      console.error('Failed to fetch stats:', e)
    }
  }

  const handleTest = async () => {
    if (!config.draft_model || !config.target_model) return

    setTesting(true)
    setTestResult(null)

    try {
      const res = await fetch('/v1/speculative/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          draft_model: config.draft_model,
          target_model: config.target_model,
          prompt: 'The future of AI is',
          max_tokens: 50,
        }),
      })

      const data = await res.json()
      setTestResult(data)
    } catch (e) {
      setTestResult({ error: e.message })
    } finally {
      setTesting(false)
    }
  }

  const handleSave = async () => {
    try {
      await fetch('/v1/speculative/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      alert('Configuration saved!')
      fetchStats()
    } catch (e) {
      alert(`Failed to save: ${e.message}`)
    }
  }

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-center h-64">
          <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
        </div>
      </div>
    )
  }

  const localModels = models.filter(m => m.local)

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-dark-900 dark:text-white flex items-center gap-2">
          <Zap className="w-6 h-6 text-yellow-500" />
          Speculative Decoding
        </h1>
        <p className="text-dark-500 dark:text-dark-400 text-sm mt-1">
          Accelerate generation using a small draft model to speculate tokens for a larger target model
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-dark-900 dark:text-white mb-4 flex items-center gap-2">
            <Cpu className="w-5 h-5" />
            Configuration
          </h2>

          <div className="space-y-4">
            <div>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={config.enabled}
                  onChange={(e) => setConfig({ ...config, enabled: e.target.checked })}
                  className="w-4 h-4 rounded border-dark-300 text-primary-600 focus:ring-primary-500"
                />
                <span className="font-medium">Enable Speculative Decoding</span>
              </label>
            </div>

            <div>
              <label className="block text-sm font-medium text-dark-700 dark:text-dark-300 mb-1">
                Draft Model (Small/Fast)
              </label>
              <select
                value={config.draft_model}
                onChange={(e) => setConfig({ ...config, draft_model: e.target.value })}
                disabled={!config.enabled}
                className="input"
              >
                <option value="">Select draft model...</option>
                {localModels.map(m => (
                  <option key={m.id} value={m.id}>{m.id}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-dark-700 dark:text-dark-300 mb-1">
                Target Model (Large/Accurate)
              </label>
              <select
                value={config.target_model}
                onChange={(e) => setConfig({ ...config, target_model: e.target.value })}
                disabled={!config.enabled}
                className="input"
              >
                <option value="">Select target model...</option>
                {localModels.map(m => (
                  <option key={m.id} value={m.id}>{m.id}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-dark-700 dark:text-dark-300 mb-1">
                Draft Tokens: {config.num_draft_tokens}
              </label>
              <input
                type="range"
                min="1"
                max="10"
                value={config.num_draft_tokens}
                onChange={(e) => setConfig({ ...config, num_draft_tokens: parseInt(e.target.value) })}
                disabled={!config.enabled}
                className="w-full h-2 bg-dark-200 dark:bg-dark-700 rounded-lg appearance-none cursor-pointer accent-primary-600"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-dark-700 dark:text-dark-300 mb-1">
                Temperature: {config.temperature.toFixed(1)}
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={config.temperature}
                onChange={(e) => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
                disabled={!config.enabled}
                className="w-full h-2 bg-dark-200 dark:bg-dark-700 rounded-lg appearance-none cursor-pointer accent-primary-600"
              />
            </div>

            <button
              onClick={handleSave}
              disabled={!config.enabled || !config.draft_model || !config.target_model}
              className="btn-primary w-full"
            >
              Save Configuration
            </button>
          </div>
        </div>

        <div className="card p-6">
          <h2 className="text-lg font-semibold text-dark-900 dark:text-white mb-4 flex items-center gap-2">
            <Gauge className="w-5 h-5" />
            Performance Stats
          </h2>

          {stats ? (
            <div className="space-y-4">
              <StatRow label="Acceptance Rate" value={`${(stats.acceptance_rate * 100).toFixed(1)}%`} icon={<Target className="w-5 h-5" />} />
              <StatRow label="Avg Tokens/Step" value={stats.avg_tokens_per_step?.toFixed(2) || 'N/A'} icon={<Zap className="w-5 h-5" />} />
              <StatRow label="Speedup" value={`${stats.speedup?.toFixed(2)}x` || 'N/A'} icon={<Gauge className="w-5 h-5" />} />
              <StatRow label="Total Requests" value={stats.total_requests?.toString() || '0'} icon={<Cpu className="w-5 h-5" />} />
            </div>
          ) : (
            <div className="text-center py-8 text-dark-500 dark:text-dark-400">
              <Info className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No stats available yet. Run a test to see performance.</p>
            </div>
          )}
        </div>
      </div>

      <div className="card p-6">
        <h2 className="text-lg font-semibold text-dark-900 dark:text-white mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5" />
          Test Configuration
        </h2>

        <div className="flex flex-wrap gap-4 items-end">
          <button
            onClick={handleTest}
            disabled={testing || !config.enabled || !config.draft_model || !config.target_model}
            className="btn-primary"
          >
            {testing ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin mr-2" />
                Testing...
              </>
            ) : (
              'Run Test Generation'
            )}
          </button>

          {testResult && (
            <div className={clsx(
              'flex-1 min-w-[300px] p-4 rounded-lg',
              testResult.error ? 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 border border-red-200 dark:border-red-800' :
              'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-800'
            )}>
              {testResult.error ? (
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-5 h-5 flex-shrink-0" />
                  <span>Error: {testResult.error}</span>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 flex-shrink-0" />
                  <span>
                    Generated {testResult.tokens_generated} tokens in {testResult.time_ms}ms
                    ({testResult.tokens_per_second?.toFixed(1)} tok/s)
                  </span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function StatRow({ label, value, icon }) {
  return (
    <div className="flex items-center gap-3 p-3 bg-dark-50 dark:bg-dark-800/50 rounded-lg">
      <div className="p-2 bg-white dark:bg-dark-900 rounded-lg border border-dark-200 dark:border-dark-700 text-primary-600">
        {icon}
      </div>
      <div className="flex-1">
        <p className="text-xs text-dark-500 dark:text-dark-400">{label}</p>
        <p className="text-lg font-semibold text-dark-900 dark:text-white">{value}</p>
      </div>
    </div>
  )
}