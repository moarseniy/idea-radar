import { Link } from 'react-router-dom'
import type { Assessment, Contribution, ModelResult, Source } from '../api'
import { dateText, percent, safeUrl, SOURCE_STATE, TRUST, VERDICT } from '../labels'

export function Empty({ title, text }: { title: string; text: string }) {
  return (
    <div className="empty">
      <strong>{title}</strong>
      {text}
    </div>
  )
}

/** Шкала уверенности модели с отметкой порога. */
export function Confidence({ model, compact }: { model: ModelResult; compact?: boolean }) {
  const tone = model.probability > 0.75 ? 'high' : model.is_signal ? 'mid' : 'low'
  return (
    <div className={`confidence ${tone}${compact ? ' compact' : ''}`} title={`Порог решения ${percent(model.threshold)}`}>
      <div className="confidence-track">
        <div className="confidence-fill" style={{ width: percent(model.probability) }} />
        <div className="confidence-threshold" style={{ left: percent(model.threshold) }} />
      </div>
      {!compact && (
        <div className="confidence-legend">
          <span>0%</span>
          <span>порог {percent(model.threshold)}</span>
          <span>100%</span>
        </div>
      )}
    </div>
  )
}

/** Вклады признаков: столбик вправо — за сигнал, влево — против. */
export function Contributions({ items, scale }: { items: Contribution[]; scale: number }) {
  return (
    <div className="contributions">
      {items.map((c) => (
        <div className="contribution" key={c.feature}>
          <div className="contribution-label">
            <strong>{c.label}</strong>
            <small>
              {c.display_value} · {c.relative}
            </small>
          </div>
          <div className="contribution-bar">
            <span
              className={c.contribution > 0 ? 'for' : 'against'}
              style={{ width: `${Math.min(50, (50 * Math.abs(c.contribution)) / scale)}%` }}
            />
          </div>
          <b className={c.contribution > 0 ? 'for' : 'against'}>
            {c.contribution > 0 ? '+' : '−'}
            {Math.abs(c.contribution).toFixed(2)}
          </b>
        </div>
      ))}
    </div>
  )
}

export function KeyPredictors({ model, limit = 3 }: { model: ModelResult; limit?: number }) {
  return (
    <div className="key-predictors">
      {model.top_for.slice(0, limit).map((c) => (
        <span key={c.feature} className="chip for" title={`${c.display_value} · вклад +${c.contribution.toFixed(2)}`}>
          ↑ {c.label}: {c.display_value}
        </span>
      ))}
      {model.top_against.slice(0, 1).map((c) => (
        <span key={c.feature} className="chip against" title={`${c.display_value} · вклад ${c.contribution.toFixed(2)}`}>
          ↓ {c.label}: {c.display_value}
        </span>
      ))}
    </div>
  )
}

export function SignalCard({ s }: { s: Assessment }) {
  return (
    <article className="signal-card">
      <Link className="signal-main" to={`/signals/${s.id}`}>
        <span className="rank">{String(s.rank ?? '').padStart(2, '0')}</span>
        <div>
          <span className="signal-domain">{s.domain}</span>
          <span className={`verdict-badge verdict-${s.verdict}`}>{VERDICT[s.verdict]}</span>
          <h3 className="signal-title">{s.title}</h3>
          <p className="signal-summary">
            {s.verdict === 'weak_signal' ? s.why_now?.text || s.summary : s.reason || s.summary}
          </p>
          {s.verdict === 'weak_signal' && s.model && <KeyPredictors model={s.model} />}
          {s.related && s.related.length > 0 && (
            <p className="related">Ещё применений этой технологии: {s.related.length}</p>
          )}
        </div>
        <div className="score-badge">
          {s.score}
          <small>{s.score_kind === 'model' ? '% уверенность' : 'из 100'}</small>
          {s.model && <Confidence model={s.model} compact />}
        </div>
      </Link>
      <div className="signal-foot">
        <span>{s.source_ids.length} источн.</span>
        <span>{s.stage}</span>
        {s.limitations.length > 0 && <span>Ограничений: {s.limitations.length}</span>}
        <Link to={`/signals/${s.id}`}>Открыть инсайт ↗</Link>
      </div>
    </article>
  )
}

export function RejectedCard({ s }: { s: Assessment }) {
  return (
    <article className="rejected-card">
      <span className="reason-badge">{VERDICT[s.verdict]}</span>
      {s.score_kind === 'model' && <span className="tag">Уверенность {s.score}%</span>}
      <h3>{s.title}</h3>
      <p>{s.reason}</p>
      {s.model && s.model.exclusion_reasons.length > 0 && (
        <ul className="reasons">
          {s.model.exclusion_reasons.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      )}
      <Link className="subtle" to={`/signals/${s.id}`}>
        Посмотреть основания ↗
      </Link>
    </article>
  )
}

export function TrustTag({ trust }: { trust: Source['trust'] }) {
  return <span className={`trust trust-${trust}`}>{TRUST[trust] ?? trust}</span>
}

export function SourceCard({ s, onOpen }: { s: Source; onOpen?: (s: Source) => void }) {
  return (
    <article className="source-card">
      <span className="tag">{s.type_label || s.type}</span>
      {s.media_kind === 'spoken' && <span className="tag">Доклад / аудио / видео</span>}
      {s.transcript_origin && <span className="tag">{s.transcript_origin.startsWith('asr_') ? 'Распознано из аудио' : 'Опубликованная расшифровка'}</span>}
      {s.transcript_format && <span className="tag">Текст: {s.transcript_format.toUpperCase()}</span>}
      {s.media_license && <span className="tag">Лицензия: {s.media_license}</span>}
      <TrustTag trust={s.trust} />
      {s.primary_only && <span className="tag warn">Только первичный индикатор</span>}
      <span className="tag">{SOURCE_STATE(s)}</span>
      {s.cached && <span className="tag">Из кэша</span>}
      <h3>
        <a href={safeUrl(s.url)} target="_blank" rel="noopener noreferrer">
          {s.title} ↗
        </a>
      </h3>
      <small>
        {s.publisher} · {dateText(s.published_at)} · {s.language}
      </small>
      <p>{s.trust_reason}</p>
      {s.transcript_url && <a className="transcript-download" href={safeUrl(s.transcript_url)} target="_blank" rel="noopener noreferrer">Исходный файл расшифровки ↗</a>}
      {s.audio_url && <a className="transcript-download" href={safeUrl(s.audio_url)} target="_blank" rel="noopener noreferrer">Аудиозапись эпизода ↗</a>}
      {s.error && <p className="muted">{s.error}</p>}
      <div className="source-actions">
        {s.status === 'read' && onOpen && (
          <button className="subtle" onClick={() => onOpen(s)}>
            Прочитанный текст
          </button>
        )}
        <span>Получен: {dateText(s.retrieved_at)}</span>
      </div>
    </article>
  )
}

export function SourceTranscriptText({ source }: { source: Source }) {
  const text = source.text || source.error || 'Текст отсутствует'
  const chunks = text.split(/(\[\d{2}:\d{2}(?::\d{2})?\])/g)
  return (
    <>
      {source.transcript_url && <p><a className="transcript-download" href={safeUrl(source.transcript_url)} target="_blank" rel="noopener noreferrer">Открыть исходный файл транскрипта ↗</a></p>}
      <div className="source-text">
        {chunks.map((chunk, index) => {
          const match = /^\[(\d{2}:\d{2}(?::\d{2})?)\]$/.exec(chunk)
          if (!match) return <span key={index}>{chunk}</span>
          const href = sourceTimestampUrl(source, match[1])
          return href ? <a key={index} href={href} target="_blank" rel="noopener noreferrer">{chunk}</a> : <strong key={index}>{chunk}</strong>
        })}
      </div>
    </>
  )
}

export function sourceTimestampUrl(source: Source, time: string): string | null {
  let url: URL
  try {
    url = new URL(source.audio_url || source.url)
  } catch {
    return null
  }
  const parts = time.split(':').map(Number)
  const seconds = parts.length === 3 ? parts[0] * 3600 + parts[1] * 60 + parts[2]
    : parts[0] * 60 + parts[1]
  if (/(^|\.)youtube\.com$/.test(url.hostname) || url.hostname === 'youtu.be') {
    url.searchParams.set('t', String(seconds))
    return safeUrl(url.toString()) ?? null
  }
  if (/\.(mp3|m4a|wav|ogg|opus|webm|mp4)$/i.test(url.pathname)) {
    url.hash = `t=${seconds}`
    return safeUrl(url.toString()) ?? null
  }
  return null
}

export function Modal({ children, onClose, wide }: { children: React.ReactNode; onClose: () => void; wide?: boolean }) {
  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <div className={`modal${wide ? ' wide' : ''}`} role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <button className="dialog-close" onClick={onClose} aria-label="Закрыть">
          ×
        </button>
        {children}
      </div>
    </div>
  )
}
