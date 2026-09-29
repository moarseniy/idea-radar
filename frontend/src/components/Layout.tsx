import { useState } from 'react'
import { NavLink, Outlet, useLocation, useNavigate, useParams } from 'react-router-dom'
import { radar } from '../api'
import { dateText, STATUS } from '../labels'
import { useApp } from '../state'

const NAV = [
  { to: '/', icon: '⌕', label: 'Обзор сигналов', end: true },
  { to: '/stats', icon: '▤', label: 'Статистика' },
  { to: '/registry', icon: '◈', label: 'Реестр источников' },
  { to: '/catalog', icon: '▦', label: 'Каталог примеров', badge: '100' },
  { to: '/method', icon: '◎', label: 'Методология' },
]

const TITLES: [RegExp, string][] = [
  [/^\/runs\//, 'Исследование'],
  [/^\/signals\//, 'Инсайт'],
  [/^\/stats/, 'Статистика'],
  [/^\/registry/, 'Реестр источников'],
  [/^\/catalog/, 'Каталог примеров'],
  [/^\/method/, 'Методология'],
]

export function Layout() {
  const { config, runs, refresh, toast } = useApp()
  const { pathname } = useLocation()
  const navigate = useNavigate()
  const params = useParams()
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const title = TITLES.find(([re]) => re.test(pathname))?.[1] ?? 'Обзор сигналов'
  const activeRun = params.runId ?? params.signalId?.split('_cand_')[0]

  async function deleteRun(run: (typeof runs)[number]) {
    if (!window.confirm(`Удалить исследование «${run.query}»? Будут удалены его результаты и сохранённые источники.`)) return
    setDeletingId(run.id)
    try {
      await radar.deleteRun(run.id)
      if (activeRun === run.id) navigate('/')
      await refresh()
      toast('Исследование удалено из истории')
    } catch (e) {
      toast((e as Error).message)
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <>
      <aside className="sidebar">
        <NavLink className="brand" to="/" aria-label="IDEA — главная">
          <span className="brand-mark">
            i<span>✦</span>
          </span>
          <span>
            IDEA<span className="brand-dot">.</span>
            <small>Технологический радар</small>
          </span>
        </NavLink>
        <div className="nav-label">Рабочее пространство</div>
        <nav>
          {NAV.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.end} className={({ isActive }) => `nav-button${isActive ? ' active' : ''}`}>
              <span>{item.icon}</span> {item.label}
              {item.badge && <b>{item.badge}</b>}
            </NavLink>
          ))}
        </nav>
        <div className="history-heading">
          <span className="nav-label">История исследований</span>
          <button onClick={() => void refresh()} title="Обновить историю" aria-label="Обновить историю">
            ↻
          </button>
        </div>
        <div className="history">
          {runs.length === 0 && <p className="sidebar-note">Здесь появятся ваши исследования.</p>}
          {runs.map((r) => {
            const running = r.status === 'queued' || r.status === 'running'
            return (
              <div key={r.id} className="history-entry">
                <NavLink to={`/runs/${r.id}`} className={`history-item${r.id === activeRun ? ' selected' : ''}`}>
                  <strong>{r.query}</strong>
                  <small>
                    {dateText(r.created_at)} · {STATUS[r.status] ?? r.status} · {r.signal_count}
                  </small>
                </NavLink>
                <button
                  type="button"
                  className="history-delete"
                  aria-label={`Удалить исследование: ${r.query}`}
                  title={running ? 'Дождитесь завершения поиска' : 'Удалить из истории'}
                  disabled={running || deletingId === r.id}
                  onClick={() => void deleteRun(r)}
                >
                  {deletingId === r.id ? '…' : '×'}
                </button>
              </div>
            )
          })}
        </div>
        <div className="sidebar-bottom">
          <span className={`status-dot${config?.ready ? ' ready' : ''}`} />
          <span>{config ? (config.ready ? 'Сервис готов' : 'Поиск не настроен') : 'Подключение…'}</span>
        </div>
      </aside>
      <main>
        <header className="topline">
          <div>
            <span className="breadcrumb">Рабочее пространство</span>
            <span className="slash">/</span>
            <span>{title}</span>
          </div>
          <div className="top-actions">
            <span className="project-label">ЛЦТ 2026 · Команда JaJaBinx</span>
          </div>
        </header>
        <div className="content">
          <Outlet />
        </div>
        <footer>
          IDEA <span>Intelligent Discovery &amp; Evaluation Assistant</span>
          <span>Факты · Контекст · Решения</span>
        </footer>
      </main>
    </>
  )
}
