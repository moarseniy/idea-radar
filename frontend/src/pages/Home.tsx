import { useRef, useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { radar, type ScopeChatMessage, type ScopeSuggestion } from '../api'
import { useApp } from '../state'

const SUGGESTIONS = [
  ['Защита ИИ', 'Ранние технологии защиты ИИ-агентов'],
  ['Финтех', 'Слабые сигналы в финансовых технологиях'],
  ['Робототехника', 'Новые технологии промышленной робототехники'],
  ['Инфраструктура ИИ', 'Новые подходы к инфраструктуре для обучения и инференса ИИ'],
  ['Edge', 'Ранние технологии edge-вычислений и ИИ на устройствах'],
]

type ScopeStep = 'closed' | 'loading' | 'choices' | 'chat'
const MAX_CLARIFICATIONS = 2

function today() {
  const d = new Date()
  d.setMinutes(d.getMinutes() - d.getTimezoneOffset())
  return d.toISOString().slice(0, 10)
}

export function Home() {
  const { config, refresh, toast } = useApp()
  const navigate = useNavigate()
  const requestId = useRef(0)
  const [query, setQuery] = useState('')
  const [asOf, setAsOf] = useState(today())
  const [busy, setBusy] = useState(false)
  const [scopeStep, setScopeStep] = useState<ScopeStep>('closed')
  const [chatBusy, setChatBusy] = useState(false)
  const [chatDraft, setChatDraft] = useState('')
  const [chatMessages, setChatMessages] = useState<ScopeChatMessage[]>([])
  const [suggestions, setSuggestions] = useState<ScopeSuggestion[]>([])
  const [selectedDirections, setSelectedDirections] = useState<string[]>([])
  const [clarificationTurns, setClarificationTurns] = useState(0)
  const maxDirections = Math.max(0, (config?.search_limits.branches ?? 4) - 1)
  const clarificationLimitReached = clarificationTurns >= MAX_CLARIFICATIONS

  function closeScope() {
    requestId.current += 1
    setScopeStep('closed')
    setChatBusy(false)
  }

  function changeQuery(value: string) {
    requestId.current += 1
    setQuery(value)
    setScopeStep('closed')
    setChatBusy(false)
    setChatMessages([])
    setSuggestions([])
    setSelectedDirections([])
    setClarificationTurns(0)
  }

  async function startSearch(directions = selectedDirections) {
    closeScope()
    setBusy(true)
    try {
      const run = await radar.start(query.trim(), asOf, directions)
      await refresh()
      navigate(`/runs/${run.id}`)
    } catch (e) {
      toast((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function openScope(event?: FormEvent) {
    event?.preventDefault()
    if (!query.trim() || !config?.ready || chatBusy || scopeStep !== 'closed') return
    const currentRequest = ++requestId.current
    setScopeStep('loading')
    setChatBusy(true)
    setChatMessages([])
    setSuggestions([])
    setSelectedDirections([])
    setClarificationTurns(0)
    try {
      const result = await radar.scopeChat(query.trim(), '', [], [])
      if (requestId.current !== currentRequest) return
      setChatMessages([{ role: 'assistant', content: result.reply }])
      setSuggestions(result.suggestions)
      setScopeStep('choices')
    } catch (e) {
      if (requestId.current === currentRequest) {
        setScopeStep('closed')
        toast((e as Error).message)
      }
    } finally {
      if (requestId.current === currentRequest) setChatBusy(false)
    }
  }

  async function sendMessage(event: FormEvent) {
    event.preventDefault()
    const message = chatDraft.trim()
    if (!message || chatBusy || clarificationLimitReached || scopeStep !== 'chat') return
    const history = chatMessages
    const currentRequest = ++requestId.current
    setChatMessages([...history, { role: 'user', content: message }])
    setChatDraft('')
    setChatBusy(true)
    setScopeStep('loading')
    try {
      const result = await radar.scopeChat(query.trim(), message, history.slice(-10), selectedDirections)
      if (requestId.current !== currentRequest) return
      setChatMessages((current) => [...current, { role: 'assistant', content: result.reply }])
      setSuggestions(result.suggestions)
      setClarificationTurns((turns) => turns + 1)
      setScopeStep('choices')
    } catch (e) {
      if (requestId.current === currentRequest) {
        setChatMessages(history)
        setChatDraft(message)
        setScopeStep('chat')
        toast((e as Error).message)
      }
    } finally {
      if (requestId.current === currentRequest) setChatBusy(false)
    }
  }

  function toggleDirection(suggestion: ScopeSuggestion) {
    setSelectedDirections((current) => {
      if (current.includes(suggestion.title)) return current.filter((item) => item !== suggestion.title)
      if (current.length >= maxDirections) return current
      return [...current, suggestion.title]
    })
  }

  return (
    <section>
      <div className="intro">
        <div className="eyebrow">
          <span /> Технологическая разведка
        </div>
        <h1>
          Замечать изменения.
          <br />
          <span>Раньше рынка.</span>
        </h1>
        <p>Ранние технологии, конкретные применения и факты, на которых можно построить следующий шаг.</p>
      </div>
      <form className="search-box" onSubmit={openScope}>
        <label htmlFor="query">Какое направление исследуем?</label>
        <div className="search-input-row">
          <span aria-hidden="true">⌕</span>
          <input
            id="query"
            required
            minLength={3}
            maxLength={1200}
            value={query}
            onChange={(e) => changeQuery(e.target.value)}
            placeholder="Например, новые технологии в промышленной робототехнике"
            autoComplete="off"
          />
          <button type="submit" disabled={busy || chatBusy || scopeStep !== 'closed' || !config?.ready}>
            {chatBusy && scopeStep === 'loading' ? 'Готовлю…' : 'Уточнить тему'} <span>↗</span>
          </button>
        </div>
        <div className="search-bottom">
          <div className="suggestions">
            {SUGGESTIONS.map(([label, text]) => (
              <button type="button" key={label} onClick={() => changeQuery(text)}>
                {label}
              </button>
            ))}
          </div>
          <label className="date-label">
            <span className="date-label-title"><span aria-hidden="true">◷</span> Дата среза</span>
            <input aria-label="Дата среза поиска" type="date" required value={asOf} max={today()} onChange={(e) => setAsOf(e.target.value)} />
            <small>Более поздние публикации не учитываются</small>
          </label>
        </div>
        {query.trim().length >= 3 && (
          <button
            type="button"
            className="direct-search-link"
            disabled={busy || !config?.ready}
            onClick={() => void startSearch([])}
          >
            Начать поиск сразу по исходной теме
          </button>
        )}
      </form>

      {scopeStep !== 'closed' && (
        <section className="scope-chat" aria-label="Уточнение темы поиска" aria-busy={scopeStep === 'loading'}>
          <div className="scope-chat-head">
            <div>
              <div className="eyebrow"><span /> Уточнение области поиска</div>
              <h2>{query}</h2>
              <small className="scope-turn-count">
                Уточнения: {clarificationTurns} из {MAX_CLARIFICATIONS}
              </small>
            </div>
            <button type="button" className="scope-close" aria-label="Закрыть уточнение"
              onClick={closeScope}>×</button>
          </div>

          <div className="scope-chat-body">
            <div className={`scope-chat-underlay${scopeStep === 'choices' || scopeStep === 'loading' ? ' dimmed' : ''}`}>
              <div className="scope-messages" aria-live="polite">
                {chatMessages.map((message, index) => (
                  <div key={`${message.role}-${index}`} className={`scope-message ${message.role}`}>
                    {message.content}
                  </div>
                ))}
              </div>
              {selectedDirections.length > 0 && (
                <div className="scope-selected">
                  <strong>Добавим в поиск:</strong>
                  {selectedDirections.map((direction) => (
                    <button type="button" key={direction} onClick={() => setSelectedDirections((current) => current.filter((item) => item !== direction))}>
                      {direction} <span aria-hidden="true">×</span>
                    </button>
                  ))}
                </div>
              )}
              <form className="scope-chat-input" onSubmit={sendMessage}>
                <input
                  aria-label="Сообщение для уточнения темы"
                  value={chatDraft}
                  onChange={(e) => setChatDraft(e.target.value)}
                  placeholder={clarificationLimitReached ? 'Лимит уточнений исчерпан' : 'Например: интересуют платежи и решения на ранней стадии'}
                  maxLength={1200}
                  disabled={clarificationLimitReached || scopeStep !== 'chat' || chatBusy}
                />
                <button type="submit" disabled={chatBusy || clarificationLimitReached || !chatDraft.trim()}>Отправить</button>
                <button type="button" className="scope-back" onClick={() => setScopeStep('choices')}>К вариантам</button>
              </form>
              <div className="scope-chat-footer">
                <small>Исходная тема останется в поиске. Свободные ветки заполнятся близкими переформулировками.</small>
                <button type="button" className="scope-start" disabled={busy || !config?.ready}
                  onClick={() => void startSearch()}>
                  {busy ? 'Запускаю…' : 'Начать поиск'} <span>↗</span>
                </button>
              </div>
            </div>

            {scopeStep === 'loading' && (
              <div className="scope-overlay scope-loading-overlay" role="status" aria-live="polite">
                <span className="scope-spinner" aria-hidden="true" />
                <strong>{chatMessages.length === 0 ? 'Подбираю варианты темы' : 'Учитываю уточнение'}</strong>
                <small>Чат и выбор временно недоступны</small>
                <div className="scope-skeletons" aria-hidden="true">
                  <i /><i /><i /><i />
                </div>
              </div>
            )}

            {scopeStep === 'choices' && (
              <div className="scope-overlay scope-choice-overlay">
                <div className="scope-options-head">
                  <div>
                    <strong>Какие направления включить?</strong>
                    <small>{selectedDirections.length} из {maxDirections} выбрано</small>
                  </div>
                  <small>Исходная тема уже включена</small>
                </div>
                <div className="scope-option-grid">
                  {suggestions.slice(0, 4).map((suggestion) => {
                    const selected = selectedDirections.includes(suggestion.title)
                    const locked = !selected && selectedDirections.length >= maxDirections
                    return (
                      <button
                        type="button"
                        key={suggestion.title}
                        className={`scope-option${selected ? ' selected' : ''}`}
                        aria-pressed={selected}
                        disabled={locked}
                        onClick={() => toggleDirection(suggestion)}
                      >
                        <span className="scope-check">{selected ? '✓' : '+'}</span>
                        <strong>{suggestion.title}</strong>
                        <small>{suggestion.description}</small>
                      </button>
                    )
                  })}
                  {Array.from({ length: Math.max(0, 4 - suggestions.length) }, (_, index) => (
                    <div className="scope-option-placeholder" aria-hidden="true" key={`empty-${index}`} />
                  ))}
                </div>
                <div className="scope-choice-actions">
                  <button type="button" className="scope-own-option"
                    disabled={clarificationLimitReached || chatBusy}
                    onClick={() => setScopeStep('chat')}>
                    <span className="scope-check">✎</span>
                    <strong>Свой вариант</strong>
                    <small>{clarificationLimitReached ? 'Лимит уточнений использован' : 'Открыть чат и уточнить направление'}</small>
                  </button>
                  <button type="button" className="scope-start" disabled={busy || !config?.ready}
                    onClick={() => void startSearch()}>
                    {busy ? 'Запускаю…' : 'Начать поиск'} <span>↗</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </section>
      )}

      {config && !config.ready && <div className="notice">{config.message}</div>}
      <section className="welcome">
        <div className="section-head">
          <h2>От наблюдения к решению</h2>
          <span>Каждый вывод можно проверить</span>
        </div>
        <div className="principles">
          <article>
            <span className="principle-icon">01 /</span>
            <h3>Что изменилось</h3>
            <p>Новые результаты, первые пилоты и применения за пределами привычных сценариев.</p>
          </article>
          <article>
            <span className="principle-icon">02 /</span>
            <h3>Насколько модель уверена</h3>
            <p>Калиброванная вероятность обученной модели и признаки, которые сильнее всего повлияли на решение.</p>
          </article>
          <article>
            <span className="principle-icon">03 /</span>
            <h3>На чём основан вывод</h3>
            <p>Источники из реестра доверия, даты и дословные фрагменты рядом с каждым заключением.</p>
          </article>
        </div>
      </section>
    </section>
  )
}
