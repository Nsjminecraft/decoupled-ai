import { Menu, X, ChevronLeft, Send, Box, Zap, DownloadCloud, Settings, Activity, Cpu, Server } from 'lucide-react'
import { NavLink } from 'react-router-dom'
import { clsx } from 'clsx'

export function Sidebar({ open, onClose }) {
  const navItems = [
    { path: '/chat', label: 'Chat', icon: Send },
    { path: '/models', label: 'Models', icon: Box },
    { path: '/speculative', label: 'Speculative Decoding', icon: Zap },
    { path: '/download', label: 'Download Models', icon: DownloadCloud },
    { path: '/settings', label: 'Settings', icon: Settings },
  ]

  return (
    <>
      <div
        className={clsx(
          'fixed inset-0 z-40 bg-black/50 transition-opacity lg:hidden',
          open ? 'opacity-100' : 'opacity-0 pointer-events-none'
        )}
        onClick={onClose}
        aria-hidden="true"
      />

      <aside
        className={clsx(
          'fixed lg:static z-50 w-64 h-full bg-white dark:bg-dark-900 border-r border-dark-200 dark:border-dark-700 flex flex-col transition-transform duration-300 ease-in-out lg:translate-x-0',
          open ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        )}
        role="navigation"
        aria-label="Main navigation"
      >
        <div className="flex items-center justify-between h-16 px-4 border-b border-dark-200 dark:border-dark-700">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center">
              <Cpu className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-lg text-dark-900 dark:text-white">DeCoupled-AI</span>
          </div>
          <button
            className="lg:hidden p-2 rounded-lg hover:bg-dark-100 dark:hover:bg-dark-800"
            onClick={onClose}
            aria-label="Close sidebar"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <nav className="flex-1 p-4 space-y-1 overflow-y-auto" aria-label="Sidebar navigation">
          {navItems.map(({ path, label, icon: Icon }) => (
            <NavLink
              key={path}
              to={path}
              className={({ isActive }) =>
                clsx(
                  'sidebar-link',
                  isActive && 'sidebar-link-active'
                )
              }
              onClick={onClose}
            >
              <Icon className="w-5 h-5 flex-shrink-0" aria-hidden="true" />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-dark-200 dark:border-dark-700">
          <div className="space-y-2 text-xs text-dark-500 dark:text-dark-400">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              <span>Server Status: Connected</span>
            </div>
            <div className="flex items-center gap-2">
              <Server className="w-4 h-4" />
              <span>Backend: Running</span>
            </div>
          </div>
        </div>
      </aside>
    </>
  )
}