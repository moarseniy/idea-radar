import type { Assessment, EvidenceTimelineData, ModelResult } from '../api'
import { VERDICT } from '../labels'

function publicationData(model?: ModelResult | null) {
  const visual = model?.visualization
  if (!visual || (visual.science_status === 'unavailable' && !visual.evidence_timeline)) return null
  return visual
}

export function EvidenceSparkline({ model }: { model?: ModelResult | null }) {
  const visual = publicationData(model)
  if (!visual) return null
  const timeline = visual.evidence_timeline
  const years = timeline?.years ?? Object.keys(visual.publication_year_counts).sort((a, b) => Number(a) - Number(b))
  const channels = timeline
    ? Object.entries(timeline.channels).map(([key, channel]) => ({ key, ...channel }))
    : [{ key: 'science', label: 'Научные работы', counts: visual.publication_year_counts }]
  if (!years.length || !years.some((year) => channels.some((channel) => (channel.counts[year] ?? 0) > 0))) return null
  const totals = years.map((year) => channels.reduce((sum, channel) => sum + (channel.counts[year] ?? 0), 0))
  const max = Math.max(1, ...totals)
  const width = 88
  const height = 24
  const slot = width / years.length
  const colors = Object.fromEntries(TIMELINE_CHANNELS.map((channel) => [channel.key, channel.color]))
  const summary = channels.map((channel) =>
    `${channel.label}: ${years.map((year) => `${year} — ${channel.counts[year] ?? 'нет данных'}`).join('; ')}`,
  ).join('\n')
  const title = timeline
    ? `Найденные датированные результаты по каналам:\n${summary}`
    : `Для этой сохранённой оценки доступны только научные работы:\n${summary}`
  return (
    <span className="evidence-sparkline" title={title} role="img" aria-label={title}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Мини-график датированных результатов по доступным каналам">
        {years.map((year, index) => {
          let accumulated = 0
          const bars = channels.map((channel) => {
            const value = channel.counts[year] ?? 0
            const barHeight = height * value / max
            const y = height - accumulated - barHeight
            accumulated += barHeight
            return value > 0 ? <rect key={`${channel.key}:${year}`} x={index * slot + 1} y={y}
              width={Math.max(2, slot - 2)} height={barHeight} rx="1"
              fill={colors[channel.key] ?? '#a3a39a'} /> : null
          })
          return <g key={year}>{bars}</g>
        })}
      </svg>
      <small>по каналам</small>
    </span>
  )
}

const TIMELINE_CHANNELS = [
  { key: 'science', label: 'Научные работы', color: '#4e806d' },
  { key: 'patents', label: 'Патенты', color: '#748ad1' },
  { key: 'news', label: 'Новости и медиа', color: '#c1745d' },
  { key: 'funding', label: 'Финансирование и гранты', color: '#c69652' },
  { key: 'adoption', label: 'Пилоты, внедрения и закупки', color: '#8a6bb5' },
  { key: 'community', label: 'Технические сообщества', color: '#849277' },
  { key: 'other', label: 'Другие источники', color: '#a3a39a' },
]

export function EvidenceTimeline({ data, legacyScienceCounts }: {
  data?: EvidenceTimelineData | null
  legacyScienceCounts?: Record<string, number | null>
}) {
  const timeline: EvidenceTimelineData | undefined = data ?? (legacyScienceCounts ? {
    years: Object.keys(legacyScienceCounts).sort((a, b) => Number(a) - Number(b)),
    channels: { science: { label: 'Научные работы', counts: legacyScienceCounts } },
    note: 'Для этой сохранённой оценки доступны годовые данные только научного поиска. Другие каналы появятся в новых исследованиях.',
  } : undefined)
  const years = timeline?.years ?? []
  const channels = TIMELINE_CHANNELS.filter((channel) => timeline?.channels[channel.key])
  const totals = years.map((year) => ({
    year,
    total: channels.some((channel) => timeline?.channels[channel.key]?.counts[year] !== null
      && timeline?.channels[channel.key]?.counts[year] !== undefined)
      ? channels.reduce((sum, channel) => sum + (timeline?.channels[channel.key]?.counts[year] ?? 0), 0)
      : null,
  }))
  const max = Math.max(1, ...totals.map((item) => item.total ?? 0))
  const found = totals.reduce((sum, item) => sum + (item.total ?? 0), 0)
  const hasChannelData = (channel: typeof TIMELINE_CHANNELS[number]) =>
    years.some((year) => timeline?.channels[channel.key]?.counts[year] !== null
      && timeline?.channels[channel.key]?.counts[year] !== undefined)
  const hasMeasurements = channels.some(hasChannelData)

  return (
    <section className="signal-visual-card evidence-timeline">
      <div className="signal-visual-heading">
        <div><h3>Найденные упоминания по годам</h3>
          <p className="muted small">Динамика по всем каналам с датированными результатами: наука, патенты, медиа, финансирование, внедрения и другие источники.</p></div>
        <span className="publication-summary">В доступных каналах: {found}</span>
      </div>
      {hasMeasurements ? <>
        <div className="evidence-timeline-scroll">
          <div className="evidence-timeline-bars" role="img" aria-label={`Найденные упоминания по годам: ${totals.map((item) => `${item.year}: ${item.total}`).join(', ')}`}
            style={{ gridTemplateColumns: `repeat(${years.length}, minmax(34px, 1fr))`, minWidth: `${Math.max(years.length * 42, 480)}px` }}>
            {totals.map((item) => <div className="evidence-year" key={item.year} title={`${item.year}: ${item.total === null ? 'данных нет' : `${item.total} найденных результатов`} `}>
              <b>{item.total === null ? '—' : item.total || ''}</b>
              <span className="evidence-year-track">
                {channels.map((channel) => {
                  const value = timeline?.channels[channel.key]?.counts[item.year]
                  if (value === null || value === undefined || !value) return null
                  return <i key={channel.key} title={`${timeline?.channels[channel.key]?.label ?? channel.label}: ${value}`} style={{ height: `${100 * value / max}%`, background: channel.color }} />
                })}
              </span>
              <small>{item.year}</small>
            </div>)}
          </div>
        </div>
      </> : <p className="muted small">Агрегированные данные поисковых каналов недоступны для этого кандидата.</p>}
      <div className="evidence-timeline-legend">
        {channels.map((channel) => {
          const available = hasChannelData(channel)
          return <span key={channel.key} className={available ? undefined : 'unavailable'}
            title={!available ? 'В этом исследовании нет датированных результатов этого канала' : undefined}>
            <i style={{ background: channel.color }} />{timeline?.channels[channel.key]?.label ?? channel.label}
            {available ? '' : ' — нет данных'}
          </span>
        })}
      </div>
    </section>
  )
}

export function PublicationMap({ candidates }: { candidates: Assessment[] }) {
  const points = candidates.flatMap((candidate, index) => {
    const visual = publicationData(candidate.model)
    if (!visual) return []
    const count = visual.publication_total_found
    const share = visual.publication_recent_share_found
    if (count === null || share === null) return []
    return [{ candidate, index, count, share, visual }]
  })
  const unavailable = candidates.length - points.length
  const chart = { left: 63, top: 20, width: 535, height: 205 }
  const maxCount = Math.max(1, ...points.map((p) => p.count))
  const x = (count: number) => chart.left + Math.log10(count + 1) / Math.log10(maxCount + 1) * chart.width
  const y = (share: number) => chart.top + (1 - share) * chart.height
  const supportedCounts = points.filter((point) => point.count >= 3 && point.visual.publication_total_complete &&
    !point.visual.publication_total_capped && !point.visual.publication_years_capped)
    .map((point) => point.count).sort((a, b) => a - b)
  const lowCountLimit = supportedCounts.length
    ? supportedCounts[Math.floor((supportedCounts.length - 1) * 0.3)]
    : 0
  const highlightX = lowCountLimit ? x(lowCountLimit) : 0
  const freshY = y(0.75)
  const ticks = [0, 0.25, 0.5, 0.75, 1]
  return (
    <section className="chart publication-map">
      <h3>Карта публикационного следа</h3>
      <p className="muted small">Каждая точка — кандидат из финального ТОПа. Выделенная область: нижняя треть ТОПа по числу найденных публикаций и не менее 75% за 3 года; нужны минимум 3 работы и полный запрос без срабатывания лимита выдачи.</p>
      {points.length ? <>
        <svg viewBox="0 0 620 280" role="img" aria-label="Карта числа найденных публикаций и их свежести">
          {highlightX > 0 && <rect className="publication-map-highlight" x={chart.left} y={chart.top}
            width={highlightX - chart.left} height={freshY - chart.top} rx="7" />}
          {ticks.map((tick) => <g key={tick}>
            <line className="publication-map-grid" x1={chart.left} x2={chart.left + chart.width} y1={y(tick)} y2={y(tick)} />
            <text className="publication-map-tick" x={chart.left - 9} y={y(tick) + 3} textAnchor="end">{Math.round(tick * 100)}%</text>
          </g>)}
          {highlightX > 0 && <line className="publication-map-cut" x1={highlightX} x2={highlightX} y1={chart.top} y2={chart.top + chart.height} />}
          {points.map(({ candidate, index, count, share, visual }) => {
            const lowSample = count < 3 || !visual.publication_total_complete || visual.publication_total_capped || visual.publication_years_capped
            const highlighted = !lowSample && count <= lowCountLimit && share >= 0.75
            const cx = x(count)
            const cy = y(share)
            return <g key={candidate.id} className={`publication-point ${candidate.verdict === 'weak_signal' ? 'weak' : 'other'}${lowSample ? ' low-sample' : ''}`}>
              <title>{`${candidate.title}: ${count} найденных публикаций, ${Math.round(100 * share)}% за 3 года${visual.publication_total_capped || visual.publication_years_capped ? '; выдача могла достичь лимита' : ''}; ${VERDICT[candidate.verdict]}`}</title>
              <circle cx={cx} cy={cy} r={highlighted ? 7 : 5} />
              <text x={cx + 8} y={cy + 3}>{candidate.rank ?? index + 1}</text>
            </g>
          })}
          <line className="publication-map-axis" x1={chart.left} x2={chart.left + chart.width} y1={chart.top + chart.height} y2={chart.top + chart.height} />
          <text className="publication-map-label" x={chart.left + chart.width / 2} y="270" textAnchor="middle">Найдено публикаций в поиске (логарифмическая шкала)</text>
          <text className="publication-map-label" transform={`translate(15 ${chart.top + chart.height / 2}) rotate(-90)`} textAnchor="middle">Доля за последние 3 года</text>
          {[1, 10, 100, 1000].filter((n) => n <= maxCount || n === 1).map((n) => (
            <text key={n} className="publication-map-tick" x={x(n)} y={chart.top + chart.height + 16} textAnchor="middle">{n}</text>
          ))}
        </svg>
        <div className="publication-map-legend"><span><i className="weak" />Слабый сигнал</span><span><i className="other" />Остальные кандидаты</span><span>Обводка — мало данных или сработал лимит</span></div>
        {unavailable > 0 && <p className="publication-caption">Не показано кандидатов без сопоставимой доли свежих публикаций: {unavailable}.</p>}
      </> : <p className="muted small">Для кандидатов в этом запуске нет сопоставимых данных о научных публикациях.</p>}
    </section>
  )
}

export function ModelImpactChart({ model }: { model?: ModelResult | null }) {
  if (!model?.waterfall) return null
  if (model.fallback) return <section className="signal-visual-card"><h3>Что повлияло на оценку</h3><p className="muted small">Для этой предварительной оценки недостаточно данных, чтобы показать влияние отдельных факторов.</p></section>
  const waterfall = model.waterfall
  const stepByFeature = new Map<string, typeof waterfall.steps[number]>()
  for (const step of waterfall.steps) {
    if (step.feature !== 'other') stepByFeature.set(step.feature, step)
  }
  for (const step of [...model.top_for, ...model.top_against]) {
    if (!stepByFeature.has(step.feature)) stepByFeature.set(step.feature, step)
  }
  const allSteps = [...stepByFeature.values()]
  const positiveDrivers = allSteps.filter((step) => step.contribution > 0).sort((a, b) => b.contribution - a.contribution).slice(0, 8)
  const negativeDrivers = allSteps.filter((step) => step.contribution < 0).sort((a, b) => a.contribution - b.contribution).slice(0, 3)
  const steps = [...positiveDrivers, ...negativeDrivers].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
  const maxEffect = Math.max(0.01, ...steps.map((step) => Math.abs(step.contribution)))
  return (
    <section className="signal-visual-card model-impact-card">
      <div className="signal-visual-heading"><div>
        <h3>Что повлияло на оценку</h3>
        <p className="muted small">Показаны до 8 признаков, повышающих оценку, и до 3, снижающих её. Модель учитывает и другие признаки. Полоса вправо повышает оценку, влево снижает.</p>
      </div></div>
      <div className="model-impact-axis" aria-hidden="true">
        <span>Признак и значение</span><span><i>Снижает</i><b>Влияние на оценку</b><i>Повышает</i></span><span>Вклад</span>
      </div>
      <div className="model-impact-list">
        {steps.map((step) => {
          const magnitude = Math.min(48, 48 * Math.abs(step.contribution) / maxEffect)
          const left = step.contribution >= 0 ? 50 : 50 - magnitude
          const valueText = step.contribution > 0 ? `+${step.contribution.toFixed(3)}` : step.contribution.toFixed(3).replace('-', '−')
          const currentValue = step.imputed
            ? 'Нет данных; использовано типичное значение'
            : [step.display_value, step.relative].filter(Boolean).join(' · ')
          return <div className="model-impact-row" key={step.feature}>
            <div className="model-impact-feature"><strong>{step.label}</strong><small>{currentValue}</small></div>
            <div className="model-impact-track" role="img" aria-label={`${step.label}: ${step.contribution > 0 ? 'повышает' : step.contribution < 0 ? 'снижает' : 'не меняет'} оценку на ${valueText}`}>
              {step.contribution !== 0 && <i className={step.contribution > 0 ? 'positive' : 'negative'} style={{ left: `${left}%`, width: `${magnitude}%` }} />}
            </div>
            <b className={`model-impact-value ${step.contribution > 0 ? 'positive' : step.contribution < 0 ? 'negative' : ''}`}>{valueText}</b>
          </div>
        })}
      </div>
    </section>
  )
}
