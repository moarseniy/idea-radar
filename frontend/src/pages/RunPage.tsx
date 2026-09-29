import { useEffect, useMemo, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { radar, type Run, type Source } from '../api'
import { Empty, Modal, RejectedCard, SignalCard, SourceCard, SourceTranscriptText } from '../components/parts'
import { RunStats } from '../components/RunStats'
import { dateText, STAGES, STATUS, VERDICT } from '../labels'
import { useApp, useResource } from '../state'

const ACTIVE = new Set(['queued', 'running'])
type Tab = 'top' | 'signals' | 'candidates' | 'rejected' | 'sources' | 'stats'

function durationText(seconds: number) {
  const minutes = Math.floor(seconds / 60)
  if (minutes >= 60) return `${Math.floor(minutes / 60)} ч ${minutes % 60} мин`
  if (minutes > 0) return `${minutes} мин`
  return `${Math.floor(seconds)} сек.`
}

export function RunPage() {
  const { runId = '' } = useParams()
  const { toast, refresh } = useApp()
  const [params, setParams] = useSearchParams()
  const tab = (params.get('tab') as Tab) || 'top'
  const setTab = (t: Tab) => setParams({ tab: t }, { replace: true })
  const [opened, setOpened] = useState<Source | null>(null)
  const [clientNow, setClientNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setClientNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [])
  const { data: run, error } = useResource<Run>(
    () => radar.run(runId),
    [runId],
    (r) => {
      const active = ACTIVE.has(r.status)
      if (!active) void refresh()
      return active
    },
  )

  if (error && !run) return <Empty title="Исследование не загружено" text={error} />
  if (!run) return <div className="loading">Загрузка исследования…</div>
  const active = ACTIVE.has(run.status)
  const topCandidates = run.top_candidates ?? run.signals
  const savedAt = Date.parse(run.updated_at || run.created_at)
  const noProgressSeconds = active && Number.isFinite(savedAt)
    ? Math.max(0, (clientNow - savedAt) / 1000)
    : 0
  const elapsedSeconds = (run.elapsed_seconds ?? 0) + (run.status === 'running' ? noProgressSeconds : 0)
  const stalled = run.status === 'running' && noProgressSeconds > 120

  async function openSource(s: Source) {
    try {
      setOpened(await radar.source(run!.id, s.id))
    } catch (e) {
      toast((e as Error).message)
    }
  }

  return (
    <section>
      <div className="run-title-row">
        <div>
          <div className="eyebrow">Исследование</div>
          <h2>{run.query}</h2>
          <p className="muted">
            {STATUS[run.status] ?? run.status} · на дату {dateText(run.as_of)}
          </p>
        </div>
        <div className="run-actions">
          {active && (
            <button
              className="subtle"
              onClick={() =>
                radar
                  .cancel(run.id)
                  .then(() => toast('Остановка запрошена. Текущие операции завершатся в пределах таймаута.'))
                  .catch((e: Error) => toast(e.message))
              }
            >
              Остановить
            </button>
          )}
          <a className="subtle" href={radar.exportUrl(run.id, 'md')} download>
            ↓ Отчёт .md
          </a>
          <a className="subtle" href={radar.exportUrl(run.id, 'json')} download>
            ↓ JSON
          </a>
        </div>
      </div>

      <div className="progress-panel">
        <div className="progress-heading">
          {active && !stalled && <span className="spinner" />}
          <strong>{run.stage}</strong>
          <span>{elapsedSeconds ? `${Math.round(elapsedSeconds)} сек.` : ''}</span>
        </div>
        <div className="stage-steps">
          {STAGES.map((x, i) => (
            <span
              key={x}
              title={x}
              className={`stage-step${
                i < run.stage_index || (run.status === 'completed' && i === run.stage_index)
                  ? ' done'
                  : i === run.stage_index
                    ? ' active'
                    : ''
              }`}
            />
          ))}
        </div>
        <div className="muted small">{run.events.at(-1)?.message ?? 'Подготовка исследования'}</div>
        {stalled && (
          <div className="progress-stale" role="status">
            Сервер не сообщал о прогрессе {durationText(noProgressSeconds)}. Показаны последние
            сохранённые данные; итоговый статус появится после завершения текущей операции.
          </div>
        )}
      </div>

      <div className="stats">
        <div>
          <span>Найдено источников</span>
          <strong>{run.counters.found}</strong>
          <small>URL из поисковых ответов</small>
        </div>
        <button className="stat-link" onClick={() => setTab('sources')}>
          <span>Обработано документов</span>
          <strong>{run.counters.read}</strong>
          <small>Повторов: {run.counters.duplicates} · всего загружено {run.counters.fetched}</small>
        </button>
        <button className="stat-link" onClick={() => setTab('candidates')}>
          <span>Технологий-кандидатов</span>
          <strong>{run.counters.candidates}</strong>
          <small>Проверено: {run.counters.assessed} · открыть список →</small>
        </button>
        <div className="stat-signal">
          <span>Выбрано сигналов</span>
          <strong>
            {run.signals.length}
            <span>/ {run.limit}</span>
          </strong>
          <small>Уверенность &gt; 75%: {run.signals.filter((x) => x.score_kind === 'model' && x.score > 75).length}</small>
        </div>
      </div>

      {run.warnings.length > 0 && (
        <div className="warnings">
          {[...new Set(run.warnings)].map((w) => (
            <p key={w}>{w}</p>
          ))}
        </div>
      )}

      <div className="results-toolbar">
        <div className="tabs" role="tablist">
          {(
            [
              ['top', 'ТОП-15 кандидатов', topCandidates.length],
              ['signals', 'ТОП сигналов', run.signals.length],
              ['candidates', 'Все кандидаты', run.candidates.length],
              ['rejected', 'Исключённые', run.rejected.length],
              ['sources', 'Источники', run.sources.length],
              ['stats', 'Статистика', null],
            ] as [Tab, string, number | null][]
          ).map(([key, label, count]) => (
            <button key={key} role="tab" aria-selected={tab === key} className={tab === key ? 'active' : ''} onClick={() => setTab(key)}>
              {label} {count !== null && <b>{count}</b>}
            </button>
          ))}
        </div>
        <span className="score-help" title={run.score_note}>
          Скоринг — уверенность модели ⓘ
        </span>
      </div>

      {tab === 'top' &&
        (topCandidates.length ? (
          topCandidates.map((s) => <SignalCard key={s.id} s={s} />)
        ) : (
          <Empty
            title="Кандидаты ещё проверяются"
            text={active ? 'Финальный список появится после завершения проверки.' : 'В этом запуске не удалось подтвердить кандидатов.'}
          />
        ))}
      {tab === 'signals' &&
        (run.signals.length ? (
          run.signals.map((s) => <SignalCard key={s.id} s={s} />)
        ) : (
          <Empty
            title="Подтверждённых сигналов пока нет"
            text={active ? 'Результаты появятся после чтения источников и проверки кандидатов.' : 'Проверьте замечания к запуску или повторите поиск с более широким направлением.'}
          />
        ))}
      {tab === 'candidates' && <CandidatesTable run={run} />}
      {tab === 'rejected' &&
        (run.rejected.length ? (
          run.rejected.map((s) => <RejectedCard key={s.id} s={s} />)
        ) : (
          <Empty title="Исключений пока нет" text="Здесь появятся кандидаты, не прошедшие проверку, и причины решения." />
        ))}
      {tab === 'sources' &&
        (run.sources.length ? (
          run.sources.map((s) => <SourceCard key={s.id} s={s} onOpen={openSource} />)
        ) : (
          <Empty title="Источники ещё не загружены" text="Поиск и чтение документов выполняются последовательно." />
        ))}
      {tab === 'stats' && <RunStats run={run} />}

      <details className="audit">
        <summary>Как получен результат</summary>
        <p>{run.plan?.interpretation}</p>
        {run.plan?.query_language && <p>Язык исходного запроса: {run.plan.query_language.toUpperCase()}</p>}
        <p>{run.score_note}</p>
        {run.plan && (
          <ul>
            {run.plan.branches.map((b) => (
              <li key={b.topic}>
                {b.topic}: {b.query_ru} / {b.query_en}
                {b.localized_queries && b.localized_queries.length > 2 && (
                  <details className="locale-audit">
                    <summary>Международные формулировки запроса</summary>
                    <ul>
                      {b.localized_queries.filter((q) => !['ru', 'en'].includes(q.language)).map((q) => (
                        <li key={q.language}><strong>{q.language.toUpperCase()}</strong>: {q.query}</li>
                      ))}
                    </ul>
                  </details>
                )}
              </li>
            ))}
          </ul>
        )}
        <h4>Каналы поиска</h4>
        <table>
          <thead>
            <tr>
              <th>Канал</th>
              <th>Запросов</th>
              <th>Ссылок</th>
              <th>Ошибок</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(
              (run.searches ?? []).reduce<Record<string, { n: number; refs: number; errors: string[] }>>((acc, s) => {
                const key = s.connector ?? 'Веб-поиск'
                acc[key] ??= { n: 0, refs: 0, errors: [] }
                acc[key].n += s.query_count ?? 1
                acc[key].refs += s.references.length
                if (s.error) acc[key].errors.push(s.error)
                return acc
              }, {}),
            ).map(([name, v]) => (
              <tr key={name}>
                <td>{name}</td>
                <td>{v.n}</td>
                <td>{v.refs}</td>
                <td title={[...new Set(v.errors)].join('; ')}>
                  {v.errors.length ? <>{v.errors.length}<small className="block muted">{[...new Set(v.errors)].join(', ')}</small></> : 0}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <h4>Автоматические этапы</h4>
        <table>
          <thead>
            <tr>
              <th>Этап</th>
              <th>Время</th>
              <th>Результат</th>
            </tr>
          </thead>
          <tbody>
            {run.calls.map((c, i) => (
              <tr key={i}>
                <td>{c.role}</td>
                <td>{c.seconds} с</td>
                <td>{c.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>

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
    </section>
  )
}

function CandidatesTable({ run }: { run: Run }) {
  const byCandidate = useMemo(() => {
    const map = new Map<string, Run['assessments'][number]>()
    for (const a of run.assessments) map.set(a.id.split('_').slice(1).join('_'), a)
    return map
  }, [run.assessments])
  const variantOf = new Map<string, number>()
  for (const s of run.signals) for (const r of s.related ?? []) variantOf.set(r.id, s.rank ?? 0)
  if (!run.candidates.length) return <Empty title="Кандидаты ещё не выделены" text="Они появятся после чтения документов." />
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Технология и применение</th>
            <th>Область</th>
            <th>Решение</th>
            <th className="num">Уверенность</th>
          </tr>
        </thead>
        <tbody>
          {run.candidates.map((c, i) => {
            const a = byCandidate.get(c.id)
            return (
              <tr key={c.id}>
                <td className="muted">{i + 1}</td>
                <td>
                  {a ? <Link to={`/signals/${a.id}`}>{c.title}</Link> : c.title}
                  <small className="block muted">{c.application}</small>
                </td>
                <td>{c.domain}</td>
                <td>
                  {a ? <span className={`verdict v-${a.verdict}`}>{VERDICT[a.verdict]}</span> : <span className="muted">Проверяется…</span>}
                  {a && variantOf.has(a.id) && <small className="block muted">применение сигнала №{variantOf.get(a.id)}</small>}
                </td>
                <td className="num">{a?.score_kind === 'model' ? `${a.score}%` : '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
