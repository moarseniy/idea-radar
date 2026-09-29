import type { RunStatus, Trust, Verdict } from './api'

export const STATUS: Record<RunStatus, string> = {
  queued: 'В очереди',
  running: 'В работе',
  completed: 'Завершено',
  partial: 'Частичная выдача',
  failed: 'Ошибка',
  cancelled: 'Остановлено',
  interrupted: 'Прервано',
}

export const VERDICT: Record<Verdict, string> = {
  weak_signal: 'Слабый сигнал',
  mature: 'Зрелая технология',
  hype_or_noise: 'Шум или не технология',
  irrelevant: 'Не соответствует запросу',
  insufficient_evidence: 'Недостаточно доказательств',
}

export const TRUST: Record<Trust, string> = {
  high: 'Высокая',
  medium: 'Средняя',
  low: 'Пониженная',
  unverified: 'Не установлена',
}

export const STAGES = ['План', 'Поиск', 'Документы', 'Кандидаты', 'Проверка', 'Анализ', 'Результат']

export const LLM_AXES: Record<string, string> = {
  early_stage: 'Ранняя стадия',
  novelty: 'Конкретная новизна',
  momentum: 'Значимое изменение',
  evidence: 'Доказательства',
  maturity: 'Зрелость',
}

export const LLM_VALUES: Record<string, string> = {
  strong: 'Сильное подтверждение',
  partial: 'Частичное подтверждение',
  unknown: 'Нет данных',
  contradictory: 'Противоречие',
}

export const SOURCE_STATE = (s: { status: string; duplicate_of: string | null }) =>
  s.duplicate_of ? 'Повтор документа' : s.status === 'read' ? 'Текст получен' : s.status === 'excluded_date' ? 'Исключён по дате' : 'Не удалось прочитать'

export function dateText(value?: string | null) {
  if (!value) return 'Дата не установлена'
  const d = new Date(value)
  return Number.isNaN(d.getTime()) ? value : d.toLocaleDateString('ru-RU', { day: 'numeric', month: 'short', year: 'numeric' })
}

export function safeUrl(value?: string) {
  try {
    const u = new URL(value ?? '')
    return ['http:', 'https:'].includes(u.protocol) ? u.href : undefined
  } catch {
    return undefined
  }
}

export function percent(p: number) {
  return `${Math.round(p * 100)}%`
}
