import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { radar, type Assessment, type Finding, type Source } from '../api'
import { Confidence, Contributions, Empty, Modal, SourceCard, SourceTranscriptText, sourceTimestampUrl, TrustTag } from '../components/parts'
import { dateText, LLM_AXES, LLM_VALUES, percent, safeUrl, VERDICT } from '../labels'
import { useApp, useResource } from '../state'

const BLOCKS: [keyof Assessment, string][] = [
  ['description', 'Технология'],
  ['why_now', 'Почему сейчас'],
  ['why_early', 'Почему это ранняя стадия'],
  ['advantage', 'Потенциальное преимущество'],
]

export function SignalPage() {
  const { signalId = '' } = useParams()
  const { toast } = useApp()
  const [opened, setOpened] = useState<Source | null>(null)
  const { data, error } = useResource(() => radar.signal(signalId), [signalId])
  if (error && !data) return <Empty title="Инсайт не загружен" text={error} />
  if (!data) return <div className="loading">Загрузка инсайта…</div>
  const s = data.signal
  const m = s.model
  const runId = signalId.split('_cand_')[0]
  const sources = new Map(data.sources.map((x) => [x.id, x]))
  const scale = m ? Math.max(0.5, ...[...m.top_for, ...m.top_against].map((c) => Math.abs(c.contribution))) : 1

  async function openSource(id: string) {
    try {
      setOpened(await radar.source(runId, id))
    } catch (e) {
      toast((e as Error).message)
    }
  }

  const links = (ids: string[] = []) =>
    ids.length > 0 && (
      <div className="source-links">
        {ids.map((id) => (
          <button key={id} className="evidence-link" onClick={() => void openSource(id)}>
            {sources.get(id)?.publisher ?? 'Источник'} ↗
          </button>
        ))}
      </div>
    )

  const finding = (key: keyof Assessment, title: string) => {
    const f = s[key] as Finding | undefined
    if (!f?.text) return null
    return (
      <section className="report-section" key={key}>
        <h3>{title}</h3>
        <p>{f.text}</p>
        {links(f.source_ids)}
        {!f.grounded && <div className="attribution">Прямое подтверждение этого вывода не установлено.</div>}
      </section>
    )
  }

  return (
    <article className="insight">
      <Link className="back" to={`/runs/${runId}`}>
        ← К исследованию «{data.query}»
      </Link>
      <div className="insight-head">
        <div>
          <div className="eyebrow">
            {s.domain} / {VERDICT[s.verdict]}
            {s.rank ? ` / №${s.rank} в ТОП` : ''}
          </div>
          <h1 className="insight-title">{s.title}</h1>
          <p className="muted">{s.application}</p>
          <div className="report-meta">
            <span>{s.stage || 'Стадия не установлена'}</span>
            <span>На дату {dateText(data.as_of)}</span>
            {m && <span>Поисковый термин: {m.search_term}</span>}
          </div>
        </div>
        {m ? (
          <div className="insight-score">
            <span>Уверенность модели</span>
            <strong>{percent(m.probability)}</strong>
            <Confidence model={m} />
            <small>{m.is_signal ? 'выше порога — слабый сигнал' : 'ниже порога — исключён'}</small>
          </div>
        ) : (
          <div className="insight-score">
            <span>Оценка по критериям</span>
            <strong>{s.score}</strong>
            <small>из 100, не вероятность</small>
          </div>
        )}
      </div>

      <p className="reason">{s.reason}</p>
      {s.related && s.related.length > 0 && (
        <section className="panel">
          <h3>Другие применения этой технологии</h3>
          <p className="muted small">Кандидаты с той же технологией не занимают отдельные места в ТОП.</p>
          <ul className="related-list">
            {s.related.map((r) => (
              <li key={r.id}>
                <Link to={`/signals/${r.id}`}>{r.title}</Link> <span className="muted">· {r.score}%</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="insight-grid">
        <div>
          {BLOCKS.map(([key, title]) => finding(key, title))}
          {finding('case', s.case_kind === 'observed' ? 'Кейс из источника' : s.case_kind === 'proposed' ? 'Предполагаемое применение' : 'Кейс не подтверждён')}
          {s.business_use && (
            <section className="report-section">
              <h3>Возможное применение для бизнеса</h3>
              <p>{s.business_use}</p>
              <div className="attribution">Аналитическое предположение системы.</div>
            </section>
          )}
          <section className="report-section">
            <h3>Доказательства и возражения</h3>
            {s.evidence.length === 0 && <p className="muted">Дословно подтверждённых фрагментов нет.</p>}
            {s.evidence.map((e) => {
              const src = sources.get(e.source_id)
              const time = /^\[(\d{2}:\d{2}(?::\d{2})?)\]/.exec(e.quote)?.[1]
              const citedUrl = src && time ? sourceTimestampUrl(src, time) : null
              return (
                <div className="evidence-item" key={e.id}>
                  <span className="tag">{e.relation === 'contradicts' ? 'Контрдоказательство' : e.relation === 'context' ? 'Контекст' : 'Основание'}</span>
                  {src && <TrustTag trust={src.trust} />}
                  <p>{e.claim}</p>
                  <blockquote>{e.quote}</blockquote>
                  <a href={citedUrl || safeUrl(src?.url)} target="_blank" rel="noopener noreferrer">
                    {src?.title} ↗
                  </a>
                  <div className="attribution">
                    {dateText(src?.published_at)} · {src?.language} · {src?.type_label || src?.type} · фрагмент найден в тексте
                  </div>
                </div>
              )
            })}
          </section>
        </div>

        <aside>
          {m && (
            <section className="panel">
              <h3>Почему модель так решила</h3>
              <p className="muted small">
                Вклад признака в логит: знак — направление, длина — сила. Вклады складываются в итоговую оценку точно.
              </p>
              {m.patent_source && (
                <p className="muted small">
                  Патентный канал: {m.patent_source}{m.patent_warning ? ` · ${m.patent_warning}` : ''}
                </p>
              )}
              <h4>За слабый сигнал</h4>
              {m.top_for.length ? <Contributions items={m.top_for} scale={scale} /> : <p className="muted small">Нет</p>}
              <h4>Против</h4>
              {m.top_against.length ? <Contributions items={m.top_against} scale={scale} /> : <p className="muted small">Нет</p>}
              {m.exclusion_reasons.length > 0 && (
                <>
                  <h4>Причины исключения</h4>
                  <ul className="reasons">
                    {m.exclusion_reasons.map((r) => (
                      <li key={r}>{r}</li>
                    ))}
                  </ul>
                </>
              )}
            </section>
          )}
          <section className="panel">
            <h3>Проверка источников (LLM)</h3>
            {s.llm_verdict && s.llm_verdict !== s.verdict && (
              <p className="muted small">Вердикт проверки: {VERDICT[s.llm_verdict]}.</p>
            )}
            {s.predictors.map((p) => (
              <div className="predictor" key={p.name}>
                <strong>
                  {LLM_AXES[p.name] ?? p.name}
                  <small>{LLM_VALUES[p.value]}</small>
                </strong>
                <div>{p.explanation}</div>
              </div>
            ))}
          </section>
          <section className="panel">
            <h3>Ограничения</h3>
            <ul className="report-limitations">
              {s.limitations.length ? s.limitations.map((x) => <li key={x}>{x}</li>) : <li>Дополнительные ограничения не указаны; вывод требует экспертной оценки.</li>}
            </ul>
          </section>
        </aside>
      </div>

      <section className="report-section">
        <h3>Источники ({data.sources.length})</h3>
        {data.sources.map((src) => (
          <SourceCard key={src.id} s={src} onOpen={(x) => void openSource(x.id)} />
        ))}
      </section>
      <p className="muted small">
        Русское резюме и интерпретация сформированы ИИ, оригинальные цитаты сохранены.{' '}
        {m ? 'Уверенность — калиброванная вероятность обученной модели.' : 'Оценка не является калиброванной вероятностью.'}
      </p>

      {opened && (
        <Modal onClose={() => setOpened(null)} wide>
          <div className="eyebrow">Прочитанный источник</div>
          <h2>{opened.title}</h2>
          <p className="muted small">
            {dateText(opened.published_at)} · {opened.language} · {opened.type_label || opened.type}
          </p>
          <p className="notice">{opened.trust_reason}</p>
          <SourceTranscriptText source={opened} />
        </Modal>
      )}
    </article>
  )
}
