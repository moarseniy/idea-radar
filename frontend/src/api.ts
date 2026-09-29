// Типы и вызовы API радара (app/radar/api.py).

export type Trust = 'high' | 'medium' | 'low' | 'unverified'
export type Verdict = 'weak_signal' | 'mature' | 'hype_or_noise' | 'irrelevant' | 'insufficient_evidence'
export type RunStatus = 'queued' | 'running' | 'completed' | 'partial' | 'failed' | 'cancelled' | 'interrupted'

export interface Counters {
  found: number
  fetched: number
  read: number
  duplicates: number
  candidates: number
  assessed: number
  confident?: number
}

export interface RunSummary {
  id: string
  query: string
  created_at: string
  status: RunStatus
  stage: string
  as_of: string
  elapsed_seconds?: number
  counters: Counters
  signal_count: number
}

export interface Source {
  id: string
  url: string
  title: string
  publisher: string
  type: string
  type_label?: string
  trust: Trust
  trust_reason: string
  primary_only?: boolean
  registry_match?: string | null
  language: string
  published_at: string | null
  retrieved_at: string
  status: 'read' | 'failed' | 'excluded_date'
  error: string | null
  cached: boolean
  duplicate_of: string | null
  discovered_by?: string
  text?: string
  truncated?: boolean
  media_kind?: string
  extraction?: string
  transcript_url?: string
  transcript_origin?: string
  transcript_format?: string
  transcript_timestamped?: boolean
  media_license?: string
  audio_url?: string
}

export interface Candidate {
  id: string
  title: string
  application: string
  domain: string
  summary: string
  source_ids: string[]
}

export interface Finding {
  text: string
  source_ids: string[]
  grounded?: boolean
}

export interface Evidence {
  id: string
  source_id: string
  quote: string
  claim: string
  relation: 'supports' | 'contradicts' | 'context'
  event_date: string | null
}

export interface Contribution {
  feature: string
  label: string
  display_value: string
  relative: string
  contribution: number
}

export interface ModelResult {
  probability: number
  is_signal: boolean
  threshold: number
  confident: boolean
  top_for: Contribution[]
  top_against: Contribution[]
  exclusion_reasons: string[]
  warnings: string[]
  search_term: string
  extraction_status: string
  patent_source?: string
  patent_warning?: string
  snapshot_date: string
  model: string
}

export interface LlmPredictor {
  name: string
  value: 'strong' | 'partial' | 'unknown' | 'contradictory'
  explanation: string
  source_ids: string[]
}

export interface Assessment extends Candidate {
  candidate_id?: string
  verdict: Verdict
  llm_verdict?: Verdict
  reason: string
  stage: string
  description?: Finding
  why_now?: Finding
  why_early?: Finding
  advantage?: Finding
  case?: Finding
  case_kind?: 'observed' | 'proposed' | 'unknown'
  business_use?: string
  limitations: string[]
  predictors: LlmPredictor[]
  evidence: Evidence[]
  score: number
  score_kind: 'model' | 'rubric'
  rank: number | null
  related?: { id: string; title: string; score: number; application: string }[]
  model?: ModelResult | null
}

export interface Run {
  id: string
  query: string
  directions?: string[]
  as_of: string
  limit: number
  created_at: string
  updated_at: string
  finished_at: string | null
  status: RunStatus
  stage: string
  stage_index: number
  plan: { interpretation: string; query_language?: string; branches: {
    topic: string
    query_ru: string
    query_en: string
    localized_queries?: { language: string; query: string }[]
  }[] } | null
  sources: Source[]
  candidates: Candidate[]
  signals: Assessment[]
  top_candidates?: Assessment[]
  rejected: Assessment[]
  assessments: Assessment[]
  searches?: { branch: string; connector?: string; references: unknown[]; error?: string | null;
    query_count?: number; query_languages?: string[] }[]
  calls: { role: string; model: string; seconds: number; status: string; usage?: { input_tokens?: number; output_tokens?: number } }[]
  warnings: string[]
  events: { at: string; stage: string; message: string }[]
  counters: Counters
  config: { model: string; provider: string; max_sources: number; max_candidates: number }
  score_note: string
  elapsed_seconds?: number
}

export interface Config {
  ready: boolean
  message: string | null
  search_limits: { branches: number; sources: number; candidates: number }
  model: string
  provider: string
  storage: string
  score_note: string
  scoring: {
    kind: 'model' | 'rubric'
    name?: string
    features?: number
    threshold?: number
    validation?: Record<string, number>
  }
}

export interface ScopeChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ScopeSuggestion {
  title: string
  description: string
}

export interface RegistryType {
  type: string
  label: string
  trust: Trust
  trust_label: string
  primary_only: boolean
  role: string
  reason: string
  count: number
}

export interface RegistryDomain {
  domain: string
  name: string
  type: string
  type_label: string
  lang: string
  trust: Trust
  trust_label: string
  primary_only: boolean
  customized?: boolean
}

export interface RegistryDomainInput {
  domain: string
  name: string
  type: string
  lang: string
}

export interface Registry {
  version: string
  updated: string
  types: RegistryType[]
  patterns: { suffix?: string; url_regex?: string; type: string }[]
  domains: RegistryDomain[]
}

export interface CatalogRecord {
  id: number
  title: string
  domain: string
  companies: string
  stage: string
  reference_score: number
  reference_explanation: string
  mention_trend: string
  sources: { title: string; url: string }[]
  features: {
    name: string
    label: string
    description: string
    group: string
    group_label: string
    value: number | null
    display_value: string
  }[]
  measured_feature_count: number
}

export async function api<T>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const headers: Record<string, string> = { ...(init.headers as Record<string, string>) }
  let body = init.body
  if (init.json !== undefined) {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify(init.json)
  }
  const response = await fetch(path, { ...init, headers, body })
  if (!response.ok) {
    let message = `Ошибка запроса (${response.status})`
    try {
      const data = await response.json()
      if (typeof data.detail === 'string') message = data.detail
      else if (Array.isArray(data.detail)) message = data.detail.map((x: { msg: string }) => x.msg).join('; ')
    } catch {
      /* тело без JSON */
    }
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

export const radar = {
  config: () => api<Config>('/api/radar/config'),
  runs: () => api<{ runs: RunSummary[] }>('/api/searches'),
  run: (id: string) => api<Run>(`/api/searches/${encodeURIComponent(id)}`),
  start: (query: string, asOf: string, directions: string[] = []) =>
    api<{ id: string; status: string }>('/api/searches', {
      method: 'POST', json: { query, as_of: asOf, limit: 15, directions },
    }),
  scopeChat: (query: string, message: string, history: ScopeChatMessage[], selectedDirections: string[]) =>
    api<{ reply: string; suggestions: ScopeSuggestion[] }>('/api/search-scope/chat', {
      method: 'POST', json: { query, message, history, selected_directions: selectedDirections },
    }),
  deleteRun: (id: string) =>
    api<{ deleted: boolean }>(`/api/searches/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  cancel: (id: string) => api(`/api/searches/${encodeURIComponent(id)}/cancel`, { method: 'POST' }),
  signal: (id: string) =>
    api<{ signal: Assessment; sources: Source[]; query: string; as_of: string }>(`/api/signals/${encodeURIComponent(id)}`),
  source: (runId: string, sourceId: string) =>
    api<Source>(`/api/searches/${encodeURIComponent(runId)}/sources/${encodeURIComponent(sourceId)}`),
  registry: () => api<Registry>('/api/sources/registry'),
  saveRegistryDomain: (entry: RegistryDomainInput) =>
    api<Registry>('/api/sources/registry/domains', { method: 'PUT', json: entry }),
  deleteRegistryDomain: (domain: string) =>
    api<Registry>(`/api/sources/registry/domains/${encodeURIComponent(domain)}`, { method: 'DELETE' }),
  catalog: () => api<{
    count: number
    domains: Record<string, number>
    records: CatalogRecord[]
    feature_groups: Record<string, string>
  }>('/api/catalog'),
  exportUrl: (id: string, format: 'md' | 'json') => `/api/searches/${encodeURIComponent(id)}/export.${format}`,
}
