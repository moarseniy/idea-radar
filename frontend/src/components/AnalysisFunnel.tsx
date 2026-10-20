import { Link } from 'react-router-dom'
import type { Assessment, CandidateAuditItem, Run, Source } from '../api'
import { dateText, safeUrl, SOURCE_STATE, TRUST, VERDICT } from '../labels'

type FunnelState = 'done' | 'active' | 'pending' | 'partial'

function stageState(run: Run, index: number): FunnelState {
  if (index < run.stage_index) return 'done'
  if (index > run.stage_index) return 'pending'
  if (run.status === 'completed') return 'done'
  if (['partial', 'failed', 'cancelled', 'interrupted'].includes(run.status)) return 'partial'
  return 'active'
}

function stateLabel(state: FunnelState) {
  return state === 'done' ? 'Завершён' : state === 'active' ? 'В работе' : state === 'partial' ? 'Не полностью' : 'Ожидает'
}

function FunnelStage({ number, title, count, state, children }: {
  number: number
  title: string
  count: string
  state: FunnelState
  children: React.ReactNode
}) {
  return (
    <details className={`funnel-stage ${state}`} open={state === 'active'}>
      <summary>
        <span className="funnel-index">{number}</span>
        <span className="funnel-stage-title"><strong>{title}</strong><small>{count}</small></span>
        <span className={`funnel-state ${state}`}>{stateLabel(state)}</span>
      </summary>
      <div className="funnel-stage-body">{children}</div>
    </details>
  )
}

function CandidateLine({ title, application, detail, href }: {
  title: string
  application?: string
  detail?: string
  href?: string
}) {
  return (
    <li className="funnel-item">
      <div>{href ? <Link to={href}>{title}</Link> : <strong>{title}</strong>}{application && <small>{application}</small>}</div>
      {detail && <span className="funnel-item-detail">{detail}</span>}
    </li>
  )
}

function FunnelSourceLine({ source, onOpen }: { source: Source; onOpen: (source: Source) => void }) {
  const url = safeUrl(source.url)
  return (
    <li className="funnel-item funnel-source-item">
      <div>
        <button type="button" className="funnel-source-open" onClick={() => onOpen(source)}>
          {source.title || source.publisher || 'Источник'}
        </button>
        <small>{source.publisher} · {dateText(source.published_at)} · {source.type_label || source.type}</small>
      </div>
      <span className="funnel-item-detail">{SOURCE_STATE(source)} · {TRUST[source.trust] ?? source.trust}</span>
      {url && <a className="funnel-source-url" href={url} target="_blank" rel="noopener noreferrer">Открыть сайт ↗</a>}
    </li>
  )
}

function assessmentHref(assessment?: Assessment) {
  return assessment ? `/signals/${assessment.id}` : undefined
}

function auditLabel(item: CandidateAuditItem) {
  if (item.action === 'merged_duplicate') return 'Объединён с похожим кандидатом'
  if (item.action === 'limit_excluded') return 'Не прошёл лимит числа кандидатов'
  return 'Не сохранён как отдельная тема'
}

function sourceCounts(sources: Source[]) {
  const duplicate = sources.filter((source) => Boolean(source.duplicate_of)).length
  const read = sources.filter((source) => source.status === 'read' && !source.duplicate_of).length
  const failed = sources.filter((source) => source.status === 'failed' && !source.duplicate_of).length
  const excluded = sources.filter((source) => source.status === 'excluded_date').length
  return { duplicate, read, failed, excluded }
}

export function AnalysisFunnel({ run, onOpenSource }: { run: Run; onOpenSource: (source: Source) => void }) {
  const searches = run.searches ?? []
  const verificationSearches = searches.filter((search) => search.branch.toLocaleLowerCase().includes('проверка стадии и внедрений'))
  const bingVerificationSearches = verificationSearches.filter((search) => (search.connector ?? '').toLocaleLowerCase().includes('bing news'))
  const webVerificationSearches = verificationSearches.filter((search) => !(search.connector ?? '').toLocaleLowerCase().includes('bing news'))
  const referencesFound = (items: typeof searches) => items.reduce((sum, item) => sum + (item.references?.length ?? 0), 0)
  const webReferences = referencesFound(webVerificationSearches)
  const bingNewsReferences = referencesFound(bingVerificationSearches)
  const webSearchUnavailable = webVerificationSearches.some((search) => Boolean(search.error))
  const sources = run.sources ?? []
  const sourceTotals = sourceCounts(sources)
  const audit = run.candidate_audit
  const assessments = run.assessments ?? []
  const weakSignals = assessments.filter((item) => item.verdict === 'weak_signal')
  const filteredAssessments = assessments.filter((item) => item.verdict !== 'weak_signal')
  const topCandidates = run.top_candidates ?? run.signals ?? []
  const topIds = new Set(topCandidates.map((candidate) => candidate.id))
  const outsideTop = assessments.filter((item) => !topIds.has(item.id))
  const assessedCandidateIds = new Set(assessments.map((item) => item.id.slice(run.id.length + 1)))
  const unassessedCandidates = run.candidates.filter((candidate) => !assessedCandidateIds.has(candidate.id))
  const finalSources = sources.filter((source) => (source.discovered_by ?? '').toLocaleLowerCase().includes('проверка стадии и внедрений'))
  const finalSourceTotals = sourceCounts(finalSources)

  return (
    <section className="analysis-funnel">
      <div className="funnel-heading">
        <div><h3>Воронка анализа</h3><p>Откройте этап, чтобы посмотреть его результаты и отсеянные записи.</p></div>
        <span>{topCandidates.length} / {run.limit} в финальном списке</span>
      </div>
      <div className="funnel-stages">
        <FunnelStage number={1} title="Обработка источников" count={`${sourceTotals.read} прочитано · ${sourceTotals.failed} не прочитано`} state={stageState(run, 2)}>
          <p className="funnel-note">Всего найдено {sources.length}: успешно прочитано {sourceTotals.read}; не прочитано {sourceTotals.failed}; дублей {sourceTotals.duplicate}; исключено по дате {sourceTotals.excluded}.</p>
          {sources.length > 0 && <>
            <h4 className="funnel-subhead">Все источники ({sources.length})</h4>
            <ul className="funnel-list">
              {sources.map((source) => <FunnelSourceLine key={source.id} source={source} onOpen={onOpenSource} />)}
            </ul>
          </>}
        </FunnelStage>

        <FunnelStage number={2} title="Выделение и объединение кандидатов" count={`${run.candidates.length} передано на проверку`} state={stageState(run, 3)}>
          <p className="funnel-note">Извлечено из документов: {audit?.extracted_count ?? 'нет данных'}. В итоговый список кандидатов вошли: {run.candidates.length}.</p>
          {run.candidates.length ? <ul className="funnel-list">
            {run.candidates.map((candidate) => <CandidateLine key={candidate.id} title={candidate.title} application={candidate.application}
              detail={`${candidate.domain} · ${candidate.source_ids.length} источн. · передан на проверку`} />)}
          </ul> : <p className="muted small">Подходящие технологии пока не выделены.</p>}
          {!!audit?.filtered.length && <>
            <h4 className="funnel-subhead">Объединены или не прошли этот этап ({audit.filtered.length})</h4>
            <ul className="funnel-list excluded">
              {audit.filtered.map((item, index) => <CandidateLine key={`${item.stage}-${item.action}-${item.candidate.id ?? index}`}
                title={item.candidate.title} application={item.candidate.application}
                detail={`${auditLabel(item)} · ${item.reason}${item.merged_into?.title ? ` · объединён с «${item.merged_into.title}»` : ''}`} />)}
            </ul>
          </>}
        </FunnelStage>

        <FunnelStage number={3} title="Проверка зрелости" count={`Веб-поиск: ${webReferences} · Bing News: ${bingNewsReferences}`} state={stageState(run, 4)}>
          <p className="funnel-note">Дополнительный проход ищет публикации о пилотах и внедрениях. В нём используется общий веб-поиск и Bing News; научные и патентные каналы в рамках этой проверки повторно не запускаются.</p>
          <ul className="funnel-list">
            <CandidateLine title="Общий веб-поиск" detail={webVerificationSearches.length
              ? `${webVerificationSearches.length} задач · ${webReferences} материалов${webSearchUnavailable ? ' · сервис не ответил в этом запуске' : ''}`
              : 'Не запускался'} />
            <CandidateLine title="Bing News" detail={bingVerificationSearches.length
              ? `${bingVerificationSearches.length} задач · ${bingNewsReferences} материалов`
              : 'Не запускался'} />
            <CandidateLine title="Загрузка найденных материалов" detail={`${finalSources.length} найдено · ${finalSourceTotals.read} прочитано · ${finalSourceTotals.failed} не удалось прочитать`} />
          </ul>
        </FunnelStage>

        <FunnelStage number={4} title="Проверка доказательств и оценка" count={`${assessments.length} из ${run.candidates.length} оценено`} state={stageState(run, 5)}>
          <p className="funnel-note">Слабые сигналы проходят дальше. Остальные сохраняются с вердиктом и причиной, а не исчезают из истории проверки.</p>
          {unassessedCandidates.length > 0 && <>
            <h4 className="funnel-subhead">Ожидают оценки ({unassessedCandidates.length})</h4>
            <ul className="funnel-list">{unassessedCandidates.map((candidate) => <CandidateLine key={candidate.id}
              title={candidate.title} application={candidate.application} detail="Проверка не завершена" />)}</ul>
          </>}
          {weakSignals.length > 0 && <>
            <h4 className="funnel-subhead">Прошли как слабые сигналы ({weakSignals.length})</h4>
            <ul className="funnel-list">{weakSignals.map((item) => <CandidateLine key={item.id} title={item.title}
              application={item.application} detail={`${item.score_kind === 'model' ? `${item.score}%` : `${item.score} из 100`} · ${item.reason}`}
              href={assessmentHref(item)} />)}</ul>
          </>}
          {filteredAssessments.length > 0 && <>
            <h4 className="funnel-subhead">Не подтверждены как слабые сигналы ({filteredAssessments.length})</h4>
            <ul className="funnel-list excluded">{filteredAssessments.map((item) => <CandidateLine key={item.id} title={item.title}
              application={item.application} detail={`${VERDICT[item.verdict]} · ${item.reason}`}
              href={assessmentHref(item)} />)}</ul>
          </>}
        </FunnelStage>

        <FunnelStage number={5} title={`Финальный ТОП-${run.limit}`} count={`${topCandidates.length} кандидатов`} state={stageState(run, 6)}>
          <p className="funnel-note">Сначала идут подтверждённые слабые сигналы по убыванию оценки, затем остальные кандидаты по убыванию оценки. Поэтому слабых сигналов может быть меньше {run.limit}, но сам список заполняется лучшими проверенными кандидатами.</p>
          {topCandidates.length > 0 ? <ul className="funnel-list">
            {topCandidates.map((item, index) => <CandidateLine key={item.id} title={item.title} application={item.application}
              detail={`№${item.rank ?? index + 1} · ${VERDICT[item.verdict]} · ${item.score_kind === 'model' ? `${item.score}%` : `${item.score} из 100`}`}
              href={assessmentHref(item)} />)}
          </ul> : <p className="muted small">Финальный список появится после проверки кандидатов.</p>}
          {outsideTop.length > 0 && <>
            <h4 className="funnel-subhead">Не вошли в финальный список ({outsideTop.length})</h4>
            <ul className="funnel-list excluded">{outsideTop.map((item) => <CandidateLine key={item.id} title={item.title}
              application={item.application}
              detail={`${VERDICT[item.verdict]} · ${item.verdict === 'weak_signal' ? 'ниже порога ТОП-15 по оценке' : 'не хватило места после ранжирования'}`}
              href={assessmentHref(item)} />)}</ul>
          </>}
        </FunnelStage>
      </div>
    </section>
  )
}
