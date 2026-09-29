import type { Run } from '../api'
import { TRUST, VERDICT } from '../labels'

interface Bar {
  label: string
  value: number
  hint?: string
  shade?: number // 0..1 — насыщенность одного оттенка (порядковые шкалы)
}

/** Горизонтальные столбики одной серии: подписи в цвете текста, значение справа. */
function Bars({ title, note, bars, total }: { title: string; note?: string; bars: Bar[]; total?: number }) {
  const max = Math.max(1, ...bars.map((b) => b.value))
  return (
    <section className="chart">
      <h3>{title}</h3>
      {note && <p className="muted small">{note}</p>}
      <div className="bars">
        {bars.map((b) => (
          <div className="bar-row" key={b.label} title={`${b.label}: ${b.value}${total ? ` из ${total}` : ''}${b.hint ? ` — ${b.hint}` : ''}`}>
            <span className="bar-label">{b.label}</span>
            <span className="bar-track">
              <span className="bar-fill" style={{ width: `${(100 * b.value) / max}%`, opacity: b.shade ?? 1 }} />
            </span>
            <b className="bar-value">{b.value}</b>
          </div>
        ))}
      </div>
    </section>
  )
}

export function RunStats({ run }: { run: Run }) {
  const topCandidates = run.top_candidates ?? run.signals
  const scored = run.assessments.filter((a) => a.model)
  const confident = run.signals.filter((s) => s.score_kind === 'model' && s.score > 75).length
  const threshold = scored[0]?.model?.threshold ?? 0.5

  const bins = Array.from({ length: 10 }, (_, i) => ({
    label: `${i * 10}–${i * 10 + 10}%`,
    value: scored.filter((a) => Math.min(9, Math.floor((a.model!.probability * 100) / 10)) === i).length,
    hint: i * 10 >= threshold * 100 ? 'выше порога' : 'ниже порога',
  }))
  const verdicts = Object.entries(VERDICT).map(([k, label]) => ({
    label,
    value: run.assessments.filter((a) => a.verdict === k).length,
  }))
  const read = run.sources.filter((s) => s.status === 'read' && !s.duplicate_of)
  const trustOrder = ['high', 'medium', 'low', 'unverified'] as const
  const trust = trustOrder.map((t, i) => ({ label: TRUST[t], value: read.filter((s) => s.trust === t).length, shade: 1 - i * 0.22 }))
  const types = Object.entries(
    read.reduce<Record<string, number>>((acc, s) => {
      const key = s.type_label || s.type
      acc[key] = (acc[key] ?? 0) + 1
      return acc
    }, {}),
  )
    .sort((a, b) => b[1] - a[1])
    .map(([label, value]) => ({ label, value }))
  const domains = Object.entries(
    run.signals.reduce<Record<string, number>>((acc, s) => {
      acc[s.domain] = (acc[s.domain] ?? 0) + 1
      return acc
    }, {}),
  )
    .sort((a, b) => b[1] - a[1])
    .map(([label, value]) => ({ label, value }))
  const reasons = Object.entries(
    run.rejected
      .flatMap((a) => a.model?.exclusion_reasons ?? [])
      .reduce<Record<string, number>>((acc, r) => {
        acc[r] = (acc[r] ?? 0) + 1
        return acc
      }, {}),
  )
    .sort((a, b) => b[1] - a[1])
    .map(([label, value]) => ({ label, value }))

  return (
    <div className="run-stats">
      <div className="hero-numbers">
        <div>
          <strong>{confident}</strong>
          <span>сигналов с уверенностью &gt; 75%</span>
        </div>
        <div>
          <strong>{topCandidates.length}</strong>
          <span>кандидатов в финальном ТОП из {run.counters.assessed} проверенных</span>
        </div>
        <div>
          <strong>{read.length}</strong>
          <span>уникальных документов прочитано</span>
        </div>
        <div>
          <strong>{Math.round((100 * trust[0].value + 100 * trust[1].value) / Math.max(1, read.length))}%</strong>
          <span>документов высокого и среднего доверия</span>
        </div>
      </div>
      <div className="chart-grid">
        <Bars
          title="Уверенность модели по кандидатам"
          note={`Порог решения ${Math.round(threshold * 100)}%. Оценено моделью: ${scored.length} из ${run.assessments.length}.`}
          bars={bins}
          total={scored.length}
        />
        <Bars title="Решения по кандидатам" bars={verdicts} total={run.assessments.length} />
        <Bars title="Доверие к прочитанным документам" note="По реестру источников." bars={trust} total={read.length} />
        <Bars title="Типы источников" bars={types} total={read.length} />
        {domains.some((d) => d.value > 1) && <Bars title="Области сигналов в ТОП" bars={domains} total={run.signals.length} />}
        {reasons.length > 0 && <Bars title="Причины исключения (по модели)" bars={reasons} />}
      </div>
    </div>
  )
}
