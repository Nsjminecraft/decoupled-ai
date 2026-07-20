import { Menu, Sun, Moon, Cpu, Github, ExternalLink } from 'lucide-react'
import { clsx } from 'clsx'

export function Header({ onMenuClick }) {
  const [darkMode, setDarkMode] = React.useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('darkMode') === 'true' ||
        (!localStorage.getItem('darkMode') && window.matchMedia('(prefers-color-scheme: dark)').matches)
    }
    return false
  })

  React.useEffect(() => {
    const root = document.documentElement
    if (darkMode) {
      root.classList.add('dark')
      localStorage.setItem('darkMode', 'true')
    } else {
      root.classList.remove('dark')
      localStorage.setItem('darkMode', 'false')
    }
  }, [darkMode])

  return (
    <header className="sticky top-0 z-30 h-16 bg-white/80 dark:bg-dark-900/80 backdrop-blur-md border-b border-dark-200 dark:border-dark-700">
      <div className="h-full px-4 lg:px-6 flex items-center justify-between gap-4">
        <button
          className="lg:hidden p-2 rounded-lg hover:bg-dark-100 dark:hover:bg-dark-800"
          onClick={onMenuClick}
          aria-label="Open menu"
        >
          <Menu className="w-6 h-6 text-dark-600 dark:text-dark-400" />
        </button>

        <div className="flex-1 lg:flex-none" />

        <div className="flex items-center gap-3">
          <button
            className="p-2 rounded-lg hover:bg-dark-100 dark:hover:bg-dark-800 transition-colors"
            onClick={() => setDarkMode(!darkMode)}
            aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {darkMode ? <Sun className="w-5 h-5 text-dark-600 dark:text-dark-400" /> : <Moon className="w-5 h-5 text-dark-600 dark:text-dark-400" />}
          </button>

          <div className="flex items-center gap-2">
            <a
              href="https://github.com/nsjminecraft/DeCoupled-AI"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-lg hover:bg-dark-100 dark:hover:bg-dark-800 transition-colors"
              aria-label="GitHub repository"
            >
              <Github className="w-5 h-5 text-dark-600 dark:text-dark-400" />
            </a>
            <a
              href="http://localhost:8080/v1"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-lg hover:bg-dark-100 dark:hover:bg-dark-800 transition-colors"
              aria-label="Open API docs"
            >
              <ExternalLink className="w-5 h-5 text-dark-600 dark:text-dark-400" />
            </a>
          </div>
        </div>
      </div>
    </header>
  )
}