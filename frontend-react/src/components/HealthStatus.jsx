import { useEffect, useState } from 'react'
import { Wifi, WifiOff, AlertCircle, CheckCircle } from 'lucide-react'

export function HealthStatus() {
  const [health, setHealth] = useState({ status: 'checking', details: null })

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch('/health')
        if (response.ok) {
          const data = await response.json()
          setHealth({ status: 'healthy', details: data })
        } else {
          setHealth({ status: 'degraded', details: { message: `HTTP ${response.status}` } })
        }
      } catch (error) {
        setHealth({ status: 'error', details: { message: error.message } })
      }
    }

    checkHealth()
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  const icons = {
    healthy: <CheckCircle className="w-5 h-5 text-green-500" />,
    degraded: <AlertCircle className="w-5 h-5 text-yellow-500" />,
    error: <WifiOff className="w-5 h-5 text-red-500" />,
    checking: <Wifi className="w-5 h-5 text-dark-400 animate-pulse" />,
  }

  const labels = {
    healthy: 'Server Healthy',
    degraded: 'Server Degraded',
    error: 'Server Unreachable',
    checking: 'Checking...',
  }

  const colors = {
    healthy: 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 border-green-200 dark:border-green-800',
    degraded: 'bg-yellow-50 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-400 border-yellow-200 dark:border-yellow-800',
    error: 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 border-red-200 dark:border-red-800',
    checking: 'bg-dark-50 dark:bg-dark-800 text-dark-600 dark:text-dark-400 border-dark-200 dark:border-dark-700',
  }

  if (health.status === 'healthy' && !health.details) return null

  return (
    <div
      className={`flex items-center gap-3 p-3 rounded-lg border animate-slide-up ${colors[health.status]}`}
      role="status"
      aria-live="polite"
    >
      {icons[health.status]}
      <span className="text-sm font-medium">{labels[health.status]}</span>
      {health.details?.message && (
        <span className="text-sm opacity-80 ml-auto">{health.details.message}</span>
      )}
      {health.details?.model && (
        <span className="text-sm px-2 py-0.5 bg-white/50 dark:bg-dark-800/50 rounded">
          {health.details.model}
        </span>
      )}
    </div>
  )
}