import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import { radar, type Config, type RunSummary } from './api'

interface AppState {
  config: Config | null
  runs: RunSummary[]
  refresh: () => Promise<void>
  toast: (message: string) => void
}

const Ctx = createContext<AppState | null>(null)

export function AppProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<Config | null>(null)
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [message, setMessage] = useState<string | null>(null)
  const timer = useRef<number | undefined>(undefined)

  const toast = useCallback((text: string) => {
    setMessage(text)
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setMessage(null), 6500)
  }, [])

  const refresh = useCallback(async () => {
    try {
      const [cfg, list] = await Promise.all([radar.config(), radar.runs()])
      setConfig(cfg)
      setRuns(list.runs)
    } catch (e) {
      toast((e as Error).message)
    }
  }, [toast])

  useEffect(() => {
    void refresh()
  }, [refresh])

  return (
    <Ctx.Provider value={{ config, runs, refresh, toast }}>
      {children}
      {message && (
        <div className="toast" role="alert">
          {message}
        </div>
      )}
    </Ctx.Provider>
  )
}

export function useApp() {
  const value = useContext(Ctx)
  if (!value) throw new Error('useApp вне AppProvider')
  return value
}

/** Загрузка с повтором, пока shouldPoll(data) истинно. */
export function useResource<T>(load: () => Promise<T>, deps: unknown[], shouldPoll?: (data: T) => boolean, interval = 2000) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const loadRef = useRef(load)
  const pollRef = useRef(shouldPoll)
  loadRef.current = load
  pollRef.current = shouldPoll

  useEffect(() => {
    let alive = true
    let handle: number | undefined
    setData(null)
    setError(null)
    const tick = async () => {
      try {
        const value = await loadRef.current()
        if (!alive) return
        setData(value)
        setError(null)
        if (pollRef.current?.(value)) handle = window.setTimeout(tick, interval)
      } catch (e) {
        if (!alive) return
        setError((e as Error).message)
        handle = window.setTimeout(tick, interval * 2.5)
      }
    }
    void tick()
    return () => {
      alive = false
      window.clearTimeout(handle)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return { data, error }
}
