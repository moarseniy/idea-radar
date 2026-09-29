import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { Link } from 'react-router-dom'
import { radar, type Registry, type RegistryDomainInput } from '../api'
import { Empty, TrustTag } from '../components/parts'
import { dateText, safeUrl, STATUS } from '../labels'
import { useApp, useResource } from '../state'

export function RegistryPage() {
  const { data, error } = useResource(() => radar.registry(), [])
  const { toast } = useApp()
  const [registry, setRegistry] = useState<Registry | null>(null)
  const [q, setQ] = useState('')
  const [type, setType] = useState('')
  const [editing, setEditing] = useState(false)
  const [form, setForm] = useState<RegistryDomainInput | null>(null)
  const [originalDomain, setOriginalDomain] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  useEffect(() => {
    if (data) setRegistry(data)
  }, [data])
  const current = registry ?? data
  const domains = useMemo(
    () =>
      (current?.domains ?? []).filter(
        (d) => (!type || d.type === type) && `${d.domain} ${d.name} ${d.type_label}`.toLowerCase().includes(q.toLowerCase()),
      ),
    [current, q, type],
  )
  if (error && !current) return <Empty title="Реестр не загружен" text={error} />
  if (!current) return <div className="loading">Загрузка реестра…</div>

  function addDomain() {
    setOriginalDomain(null)
    setForm({ domain: '', name: '', type: current?.types[0]?.type ?? 'website', lang: 'en' })
  }

  function editDomain(domain: RegistryDomainInput) {
    setOriginalDomain(domain.domain)
    setForm({ ...domain })
  }

  async function saveDomain(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!form) return
    setSaving(true)
    try {
      const updated = await radar.saveRegistryDomain(form)
      setRegistry(updated)
      setForm(null)
      toast(`Сохранено: ${form.domain}`)
    } catch (e) {
      toast((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  async function removeDomain() {
    if (!originalDomain || !window.confirm(`Убрать ${originalDomain} из реестра?`)) return
    setSaving(true)
    try {
      const updated = await radar.deleteRegistryDomain(originalDomain)
      setRegistry(updated)
      setForm(null)
      toast(`Домен ${originalDomain} убран из реестра`)
    } catch (e) {
      toast((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <section>
      <div className="eyebrow">Доверие к источникам</div>
      <h1 className="compact-title">Реестр источников</h1>
      <p className="muted">
        Единый реестр для радара и модели: тип источника задаёт уровень доверия и роль. Соцсети, блоги и агрегаторы по служат
        только первичным индикатором: сигнал на них не может держаться без независимого подтверждения.
      </p>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Тип</th>
              <th>Доверие</th>
              <th>Роль</th>
              <th>Основание</th>
              <th className="num">Доменов</th>
            </tr>
          </thead>
          <tbody>
              {current.types.map((t) => (
              <tr key={t.type}>
                <td>
                  <button className="link" onClick={() => setType(type === t.type ? '' : t.type)}>
                    {t.label}
                  </button>
                </td>
                <td>
                  <TrustTag trust={t.trust} />
                </td>
                <td>{t.role === 'Подтверждающий' ? t.role : <span className="tag warn">{t.role}</span>}</td>
                <td className="muted">{t.reason}</td>
                <td className="num">{t.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="catalog-toolbar">
        <input type="search" placeholder="Поиск по домену или названию" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Поиск по реестру" />
        <select value={type} onChange={(e) => setType(e.target.value)} aria-label="Тип источника">
          <option value="">Все типы</option>
          {current.types.map((t) => (
            <option key={t.type} value={t.type}>
              {t.label}
            </option>
          ))}
        </select>
        <span>{domains.length} из {current.domains.length}</span>
        <button className="subtle" type="button" onClick={() => setEditing((value) => !value)}>
          {editing ? 'Готово' : 'Редактировать'}
        </button>
        {editing && <button className="subtle registry-add" type="button" onClick={addDomain}>+ Добавить домен</button>}
      </div>
      {editing && <p className="registry-hint muted small">Изменения сохраняются отдельно от встроенного списка и применяются к новым исследованиям.</p>}
      {form && (
        <form className="registry-editor" onSubmit={saveDomain}>
          <div className="registry-editor-heading">
            <strong>{originalDomain ? 'Настройки домена' : 'Новый домен'}</strong>
            <button className="link" type="button" onClick={() => setForm(null)}>Отмена</button>
          </div>
          <div className="registry-editor-fields">
            <label>Домен
              <input required value={form.domain} readOnly={Boolean(originalDomain)} placeholder="example.org" onChange={(e) => setForm({ ...form, domain: e.target.value })} />
            </label>
            <label>Название
              <input required maxLength={120} value={form.name} placeholder="Название издания" onChange={(e) => setForm({ ...form, name: e.target.value })} />
            </label>
            <label>Тип источника
              <select value={form.type} onChange={(e) => setForm({ ...form, type: e.target.value })}>
                {current.types.map((t) => <option key={t.type} value={t.type}>{t.label}</option>)}
              </select>
            </label>
            <label>Язык
              <select value={form.lang} onChange={(e) => setForm({ ...form, lang: e.target.value })}>
                <option value="ru">Русский</option>
                <option value="en">Английский</option>
                <option value="multi">Несколько языков</option>
              </select>
            </label>
          </div>
          <div className="registry-editor-actions">
            {originalDomain && <button className="registry-delete" type="button" disabled={saving} onClick={() => void removeDomain()}>Убрать из реестра</button>}
            <button className="search-submit" type="submit" disabled={saving}>{saving ? 'Сохраняю…' : 'Сохранить'}</button>
          </div>
        </form>
      )}
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Домен</th>
              <th>Название</th>
              <th>Тип</th>
              <th>Доверие</th>
              <th>Язык</th>
              {editing && <th>Управление</th>}
            </tr>
          </thead>
          <tbody>
            {domains.map((d) => (
              <tr key={d.domain}>
                <td>
                  <code>{d.domain}</code>{d.customized && <small className="block muted">Изменён вручную</small>}
                </td>
                <td>{d.name}</td>
                <td>
                  {d.type_label}
                  {d.primary_only && <small className="block muted">только первичный индикатор</small>}
                </td>
                <td>
                  <TrustTag trust={d.trust} />
                </td>
                <td>{d.lang.toUpperCase()}</td>
                {editing && <td><button className="link" type="button" onClick={() => editDomain(d)}>Изменить</button></td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

export function StatsPage() {
  const { runs } = useApp()
  const done = runs.filter((r) => !['queued', 'running'].includes(r.status))
  const confident = done.reduce((sum, r) => sum + (r.counters.confident ?? 0), 0)
  return (
    <section>
      <div className="eyebrow">Статистика</div>
      <h1 className="compact-title">Модель и исследования</h1>
      <div className="hero-numbers">
        <div>
          <strong>{done.length}</strong>
          <span>завершённых исследований</span>
        </div>
        <div>
          <strong>{done.reduce((s, r) => s + r.signal_count, 0)}</strong>
          <span>сигналов в ТОП</span>
        </div>
        <div>
          <strong>{confident}</strong>
          <span>сигналов с уверенностью &gt; 75%</span>
        </div>
        <div>
          <strong>{done.reduce((s, r) => s + r.counters.read, 0)}</strong>
          <span>документов обработано</span>
        </div>
      </div>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Запрос</th>
              <th>Дата</th>
              <th>Статус</th>
              <th className="num">Источников</th>
              <th className="num">Кандидатов</th>
              <th className="num">Сигналов</th>
              <th className="num">&gt; 75%</th>
              <th className="num">Время</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id}>
                <td>
                  <Link to={`/runs/${r.id}`}>{r.query}</Link>
                </td>
                <td>{dateText(r.created_at)}</td>
                <td>{STATUS[r.status] ?? r.status}</td>
                <td className="num">{r.counters.read}</td>
                <td className="num">
                  <Link to={`/runs/${r.id}?tab=candidates`}>{r.counters.candidates}</Link>
                </td>
                <td className="num">{r.signal_count}</td>
                <td className="num">{r.counters.confident ?? '—'}</td>
                <td className="num">{r.elapsed_seconds ? `${Math.round(r.elapsed_seconds / 60)} мин` : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

export function CatalogPage() {
  const { data, error } = useResource(() => radar.catalog(), [])
  const [q, setQ] = useState('')
  const [domain, setDomain] = useState('')
  if (error && !data) return <Empty title="Каталог не загружен" text={error} />
  if (!data) return <div className="loading">Загрузка каталога…</div>
  const records = data.records.filter(
    (r) => (!domain || r.domain === domain) && `${r.title} ${r.companies} ${r.domain}`.toLowerCase().includes(q.toLowerCase()),
  )
  return (
    <section>
      <div className="eyebrow">Материалы организаторов</div>
      <h1 className="compact-title">Каталог примеров</h1>
      <p className="muted">100 технологических сигналов из исходной таблицы. Они используются только для валидации модели, не как готовые ответы поиска.</p>
      <div className="catalog-toolbar">
        <input type="search" placeholder="Поиск по технологии, компании или области" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Поиск по каталогу" />
        <select value={domain} onChange={(e) => setDomain(e.target.value)} aria-label="Область">
          <option value="">Все области</option>
          {Object.keys(data.domains).map((d) => (
            <option key={d}>{d}</option>
          ))}
        </select>
        <span>
          {records.length} из {data.count}
        </span>
      </div>
      <div className="notice">
        Показаны показатели, которые учитывает модель. Пустое значение означает «нет данных», а не ноль. Наведите курсор на показатель, чтобы увидеть его определение.
      </div>
      {records.map((r) => (
        <article className="catalog-card" key={r.id}>
          <span className="signal-domain">{r.domain}</span>
          <h3>
            {r.id}. {r.title}
          </h3>
          <p>{r.companies}</p>
          <span className="tag">{r.stage}</span>
          <span className="tag">Авторский балл: {r.reference_score}</span>
          <details>
            <summary>Объяснение и источники из Excel</summary>
            <p>{r.reference_explanation}</p>
            <p>{r.mention_trend}</p>
            {r.sources.map((s) => (
              <a key={s.url} href={safeUrl(s.url)} target="_blank" rel="noopener noreferrer">
                {s.title} ↗
              </a>
            ))}
          </details>
          <details className="catalog-features">
            <summary>
              Признаки модели: {r.measured_feature_count} из {r.features.length} заполнены
            </summary>
            {r.features.length ? Object.entries(r.features.reduce<Record<string, typeof r.features>>((groups, feature) => {
              const groupFeatures = groups[feature.group_label] ?? (groups[feature.group_label] = [])
              groupFeatures.push(feature)
              return groups
            }, {})).map(([group, features]) => (
              <section className="catalog-feature-group" key={group}>
                <h4>{group}</h4>
                <div className="catalog-feature-grid">
                  {features?.map((feature) => (
                    <span
                      className={`catalog-feature${feature.value === null ? ' missing' : ''}`}
                      key={feature.name}
                      tabIndex={0}
                      title={`${feature.description}\nЗначение: ${feature.display_value}`}
                      aria-label={`${feature.label}: ${feature.display_value}. ${feature.description}`}
                    >
                      <span>{feature.label}</span>
                      <strong>{feature.display_value}</strong>
                    </span>
                  ))}
                </div>
              </section>
            )) : <p>Для этого примера значения признаков не найдены.</p>}
          </details>
        </article>
      ))}
    </section>
  )
}

export function MethodPage() {
  return (
    <section className="method">
      <div className="eyebrow">Методология IDEA</div>
      <h1 className="compact-title">Как радар находит слабые сигналы</h1>
      <p className="muted">
        Слабый сигнал — технология на ранней стадии внедрения с растущей активностью. Радар ищет кандидатов в открытых источниках, модель оценивает каждого, а проверка
        источников отсеивает нерелевантное и неподтверждённое.
      </p>
      {[
        ['01', 'Поиск и чтение', 'LLM строит план мультязычного поиска, веб-поиск возвращает источники, радар читает тексты и отбрасывает документы позже даты оценки и повторы.'],
        ['02', 'Кандидаты', 'Из прочитанных документов выделяются конкретные пары «технология + применение»; дубли объединяются.'],
        ['03', 'Признаки', 'Для каждого кандидата собираются научные, отраслевые и финансовые свидетельства; модель использует их для оценки ранней стадии, динамики и подтверждённости технологии.'],
        ['04', 'Модель', `Логистическая регрессия с калибровкой выдаёт вероятность слабого сигнала. Вклад каждого признака показан в ответе — решение интерпретируемо.`],
        ['05', 'Проверка источников', 'LLM проверяет, относится ли кандидат к запросу, и подкрепляет описание дословными цитатами; цитаты сверяются с текстом документа. Сигнал без независимого источника из реестра не проходит.'],
        ['06', 'Реестр доверия', 'Тип и доверие каждого источника задаёт единый реестр; соцсети, блоги и агрегаторы — только первичный индикатор.'],
      ].map(([n, title, text]) => (
        <div className="method-row" key={n}>
          <b>{n}</b>
          <div>
            <h3>{title}</h3>
            <p>{text}</p>
          </div>
        </div>
      ))}
    </section>
  )
}
