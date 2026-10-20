import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import type { Assessment, Evidence, Source } from '../api'
import { dateText } from '../labels'

const WIDTH = 1360
const HEIGHT = 760
const CENTER = { x: WIDTH / 2, y: HEIGHT / 2 }
const ROOT = { x: 140, y: CENTER.y }
const LAYOUT_COLUMN_GAP = 215
const LAYOUT_ROW_LIMIT = 6

type NodeKind = 'technology' | 'claim' | 'event' | 'organization' | 'source'
type Relation = Evidence['relation']

interface GraphNode {
  id: string
  kind: NodeKind
  label: string
  subtitle: string
  detail: string
  x: number
  y: number
  sourceId?: string
  relation?: Relation
  quote?: string
  claim?: string
  source?: Source
  eventType?: string
  eventDate?: string | null
  company?: string | null
  customer?: string | null
  organizationRole?: string
}

interface GraphEdge {
  id: string
  from: string
  to: string
  label?: string
}

interface GraphModel {
  nodes: GraphNode[]
  edges: GraphEdge[]
  width: number
}

interface ViewTransform {
  x: number
  y: number
  scale: number
}

const RELATION_LABEL: Record<Relation, string> = {
  supports: 'поддерживает',
  contradicts: 'оспаривает',
  context: 'контекст',
}

const EVENT_LABEL: Record<string, string> = {
  funding: 'Финансирование',
  grant: 'Грант',
  pilot: 'Пилот',
  deployment: 'Внедрение',
  product: 'Продукт',
  procurement: 'Закупка',
  trial: 'Испытание',
  regulation: 'Регуляторное решение',
  standard_final: 'Стандарт',
}

function shortText(value: string, limit: number) {
  const normalized = value.replace(/\s+/g, ' ').trim()
  return normalized.length > limit ? `${normalized.slice(0, limit - 1).trimEnd()}…` : normalized
}

function russianCount(count: number, one: string, few: string, many: string) {
  const lastTwo = count % 100
  if (lastTwo >= 11 && lastTwo <= 14) return many
  const last = count % 10
  return last === 1 ? one : last >= 2 && last <= 4 ? few : many
}

function relationColor(relation?: Relation) {
  if (relation === 'supports') return 'supports'
  if (relation === 'contradicts') return 'contradicts'
  return 'context'
}

function placeColumns(nodes: GraphNode[], startX: number, singleOffset = 0) {
  for (let index = 0; index < nodes.length; index += 1) {
    const column = Math.floor(index / LAYOUT_ROW_LIMIT)
    const columnStart = column * LAYOUT_ROW_LIMIT
    const rowsInColumn = Math.min(LAYOUT_ROW_LIMIT, nodes.length - columnStart)
    const gap = rowsInColumn <= 1 ? 0 : Math.min(128, (HEIGHT - 180) / (rowsInColumn - 1))
    const yStart = (HEIGHT - gap * (rowsInColumn - 1)) / 2
    nodes[index].x = startX + column * LAYOUT_COLUMN_GAP
    nodes[index].y = rowsInColumn === 1 ? CENTER.y + singleOffset : yStart + (index - columnStart) * gap
  }
  return Math.ceil(nodes.length / LAYOUT_ROW_LIMIT)
}

function connectedY(node: GraphNode, edges: GraphEdge[], nodesById: Map<string, GraphNode>) {
  const parents = edges
    .filter((edge) => edge.to === node.id)
    .map((edge) => nodesById.get(edge.from))
    .filter((parent): parent is GraphNode => Boolean(parent))
  return parents.length ? parents.reduce((sum, parent) => sum + parent.y, 0) / parents.length : CENTER.y
}

function placeByParentColumns(
  nodes: GraphNode[], startX: number, anchors: GraphNode[], edges: GraphEdge[],
  nodesById: Map<string, GraphNode>,
) {
  const anchorIds = new Set(anchors.map((node) => node.id))
  const groups = new Map<number, GraphNode[]>()
  const anchorColumnCount = Math.max(1, ...anchors.map((node) => Math.round((node.x - anchors[0].x) / LAYOUT_COLUMN_GAP) + 1))

  for (const node of nodes) {
    const parents = edges.filter((edge) => edge.to === node.id && anchorIds.has(edge.from))
      .map((edge) => nodesById.get(edge.from))
      .filter((parent): parent is GraphNode => Boolean(parent))
    const parentX = parents.length ? parents.reduce((sum, parent) => sum + parent.x, 0) / parents.length : anchors[0]?.x ?? 0
    const groupIndex = Math.max(0, Math.min(anchorColumnCount - 1,
      Math.round((parentX - (anchors[0]?.x ?? parentX)) / LAYOUT_COLUMN_GAP)))
    groups.set(groupIndex, [...(groups.get(groupIndex) ?? []), node])
  }

  let columnOffset = 0
  for (let groupIndex = 0; groupIndex < anchorColumnCount; groupIndex += 1) {
    const group = groups.get(groupIndex) ?? []
    if (!group.length) {
      columnOffset += 1
      continue
    }
    group.sort((a, b) => connectedY(a, edges, nodesById) - connectedY(b, edges, nodesById)
      || a.id.localeCompare(b.id))
    const gap = Math.min(128, (HEIGHT - 180) / (LAYOUT_ROW_LIMIT - 1))
    const rows = Array.from({ length: LAYOUT_ROW_LIMIT }, (_, index) => (HEIGHT - gap * (LAYOUT_ROW_LIMIT - 1)) / 2 + index * gap)
    for (let start = 0; start < group.length; start += LAYOUT_ROW_LIMIT) {
      const chunk = group.slice(start, start + LAYOUT_ROW_LIMIT)
      const availableRows = [...rows]
      const x = startX + columnOffset * LAYOUT_COLUMN_GAP
      for (const node of chunk) {
        const targetY = connectedY(node, edges, nodesById)
        let closestIndex = 0
        for (let index = 1; index < availableRows.length; index += 1) {
          if (Math.abs(availableRows[index] - targetY) < Math.abs(availableRows[closestIndex] - targetY)) closestIndex = index
        }
        node.x = x
        node.y = availableRows.splice(closestIndex, 1)[0]
      }
      columnOffset += 1
    }
  }
  return columnOffset
}

function layoutGraph(nodes: GraphNode[], edges: GraphEdge[]): GraphModel {
  const root = nodes.find((node) => node.kind === 'technology')
  if (root) Object.assign(root, ROOT)

  const middle = nodes.filter((node) => node.kind === 'claim' || node.kind === 'event')
    .sort((a, b) => (a.kind === b.kind ? a.id.localeCompare(b.id) : a.kind === 'claim' ? -1 : 1))
  const middleStartX = 430
  const middleColumns = placeColumns(middle, middleStartX, -90)
  const sourceNodes = nodes.filter((node) => node.kind === 'source')
  const nodesById = new Map(nodes.map((node) => [node.id, node]))
  const sourceStartX = middleStartX + middleColumns * LAYOUT_COLUMN_GAP + 70
  const sourceColumns = placeByParentColumns(sourceNodes, sourceStartX, middle, edges, nodesById)

  const organizations = nodes.filter((node) => node.kind === 'organization')
  const organizationStartX = sourceStartX + sourceColumns * LAYOUT_COLUMN_GAP + 70
  const eventNodes = middle.filter((node) => node.kind === 'event')
  const organizationColumns = placeByParentColumns(organizations, organizationStartX,
    eventNodes.length ? eventNodes : middle, edges, nodesById)

  const lastColumn = organizations.length ? organizationStartX + (organizationColumns - 1) * LAYOUT_COLUMN_GAP
    : sourceNodes.length ? sourceStartX + (sourceColumns - 1) * LAYOUT_COLUMN_GAP
      : middleStartX + Math.max(0, middleColumns - 1) * LAYOUT_COLUMN_GAP
  const graphWidth = Math.max(WIDTH, lastColumn + 150)
  return { nodes, edges, width: graphWidth }
}

function buildGraph(assessment: Assessment, sources: Source[]): GraphModel {
  const evidenceSourceIds = new Set((assessment.evidence ?? []).map((evidence) => evidence.source_id))
  const graphSources = sources.filter((source) => source.status === 'read' && !source.duplicate_of &&
    (evidenceSourceIds.has(source.id) || Boolean(source.structured_event &&
      (source.structured_event.type || source.structured_event.date || source.structured_event.company || source.structured_event.customer))))
  const sourceById = new Map(graphSources.map((source) => [source.id, source]))
  const nodes: GraphNode[] = [{
    id: 'technology', kind: 'technology', label: assessment.title,
    subtitle: assessment.domain || 'Технология', detail: assessment.application || assessment.summary,
    x: ROOT.x, y: ROOT.y,
  }]
  const edges: GraphEdge[] = []
  const sourceNodes = new Map<string, GraphNode>()
  const addSource = (sourceId: string) => {
    if (sourceNodes.has(sourceId)) return sourceNodes.get(sourceId)!
    const source = sourceById.get(sourceId)
    if (!source) return undefined
    const node: GraphNode = {
      id: `source:${sourceId}`, kind: 'source', label: source.title || source.publisher || 'Источник',
      subtitle: source.type_label || source.type, detail: source.trust_reason,
      x: ROOT.x, y: ROOT.y, sourceId, source,
    }
    sourceNodes.set(sourceId, node)
    nodes.push(node)
    return node
  }

  for (const evidence of assessment.evidence ?? []) {
    const sourceNode = addSource(evidence.source_id)
    if (!sourceNode) continue
    const claimNode: GraphNode = {
      id: `claim:${evidence.id}`, kind: 'claim', label: evidence.claim || 'Подтверждение',
      subtitle: RELATION_LABEL[evidence.relation], detail: evidence.claim,
      x: ROOT.x, y: ROOT.y, sourceId: evidence.source_id,
      relation: evidence.relation, quote: evidence.quote, claim: evidence.claim,
    }
    nodes.push(claimNode)
    edges.push({ id: `edge:${claimNode.id}:root`, from: 'technology', to: claimNode.id, label: RELATION_LABEL[evidence.relation] })
    edges.push({ id: `edge:${claimNode.id}:${sourceNode.id}`, from: claimNode.id, to: sourceNode.id, label: 'цитата из' })
  }

  for (const source of graphSources) {
    const sourceNode = addSource(source.id)
    if (!sourceNode) continue
    const event = source.structured_event
    if (!evidenceSourceIds.has(source.id) && !event) {
      edges.push({ id: `edge:technology:${sourceNode.id}`, from: 'technology', to: sourceNode.id, label: 'источник' })
    }
    if (!event || !(event.type || event.date || event.company || event.customer)) continue
    const eventId = `event:${source.id}`
    const eventType = event.type || 'event'
    const eventNode: GraphNode = {
      id: eventId, kind: 'event', label: EVENT_LABEL[eventType] || 'Событие',
      subtitle: event.date ? String(event.date).slice(0, 10) : 'дата не установлена',
      detail: 'Событие извлечено из структурированной записи источника.',
      x: ROOT.x, y: ROOT.y, sourceId: source.id,
      eventType, eventDate: event.date, company: event.company, customer: event.customer,
    }
    nodes.push(eventNode)
    edges.push({ id: `edge:technology:${eventId}`, from: 'technology', to: eventId, label: 'событие' })
    edges.push({ id: `edge:${eventId}:${sourceNode.id}`, from: eventId, to: sourceNode.id, label: 'подтверждено записью' })

    for (const [role, name] of [['компания', event.company], ['заказчик', event.customer]] as const) {
      const organization = name?.trim()
      if (!organization) continue
      const orgId = `organization:${organization.toLowerCase()}`
      let orgNode = nodes.find((node) => node.id === orgId)
      if (!orgNode) {
        orgNode = { id: orgId, kind: 'organization', label: organization,
          subtitle: role === 'компания' ? 'Участник события' : 'Заказчик',
          detail: `Организация указана как ${role} в структурированном событии.`,
          x: ROOT.x, y: ROOT.y, organizationRole: role }
        nodes.push(orgNode)
      }
      edges.push({ id: `edge:${eventId}:${orgId}:${role}`, from: eventId, to: orgId, label: role })
    }
  }

  return layoutGraph(nodes, edges)
}

function initialPositions(nodes: GraphNode[]) {
  return Object.fromEntries(nodes.map(({ id, x, y }) => [id, { x, y }]))
}

function fitView(nodes: GraphNode[], width = WIDTH): ViewTransform {
  const bounds = nodes.map((node) => {
    const halfWidth = node.kind === 'technology' ? 108 : node.kind === 'source' ? 88 : 94
    const halfHeight = node.kind === 'technology' ? 50 : node.kind === 'source' ? 30 : 36
    return { left: node.x - halfWidth, right: node.x + halfWidth,
      top: node.y - halfHeight, bottom: node.y + halfHeight }
  })
  const minX = Math.min(...bounds.map((box) => box.left))
  const maxX = Math.max(...bounds.map((box) => box.right))
  const minY = Math.min(...bounds.map((box) => box.top))
  const maxY = Math.max(...bounds.map((box) => box.bottom))
  const contentWidth = Math.max(1, maxX - minX)
  const contentHeight = Math.max(1, maxY - minY)
  const scale = Math.max(0.42, Math.min(1.7, (width - 90) / contentWidth, (HEIGHT - 90) / contentHeight))
  return { scale, x: width / 2 - (minX + maxX) / 2 * scale, y: CENTER.y - (minY + maxY) / 2 * scale }
}

function relationText(relation?: Relation) {
  if (relation === 'supports') return 'Поддерживает сигнал'
  if (relation === 'contradicts') return 'Оспаривает сигнал'
  return 'Контекст'
}

export function SignalRelationshipGraph({ assessment, sources, onOpenSource }: {
  assessment: Assessment
  sources: Source[]
  onOpenSource: (id: string) => void
}) {
  const graph = useMemo(() => buildGraph(assessment, sources), [assessment, sources])
  const [positions, setPositions] = useState(() => initialPositions(graph.nodes))
  const [selectedId, setSelectedId] = useState('technology')
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [view, setView] = useState<ViewTransform>(() => fitView(graph.nodes, graph.width))
  const svgRef = useRef<SVGSVGElement | null>(null)
  const gesture = useRef<{
    kind: 'node' | 'pan'; id?: string; startX: number; startY: number;
    startView: ViewTransform
  } | null>(null)

  useEffect(() => {
    setPositions(initialPositions(graph.nodes))
    setView(fitView(graph.nodes, graph.width))
    setSelectedId('technology')
    setHoveredId(null)
  }, [graph])

  const positionMap = useMemo(() => new Map(graph.nodes.map((node) => [node.id, positions[node.id] ?? node])), [graph.nodes, positions])
  const nodeMap = useMemo(() => new Map(graph.nodes.map((node) => [node.id, node])), [graph.nodes])
  const selected = nodeMap.get(selectedId) ?? graph.nodes[0]
  const focusId = hoveredId || (selectedId === 'technology' ? null : selectedId)
  const neighborIds = focusId
    ? new Set(graph.edges.filter((edge) => edge.from === focusId || edge.to === focusId)
      .flatMap((edge) => [edge.from, edge.to]))
    : new Set(graph.nodes.map((node) => node.id))
  const showEmpty = graph.nodes.length <= 1
  const legendItems = ([
    { kind: 'technology', label: 'Сигнал' },
    { kind: 'claim', label: 'Утверждение' },
    { kind: 'event', label: 'Событие' },
    { kind: 'organization', label: 'Организация' },
    { kind: 'source', label: 'Источник' },
  ] satisfies { kind: NodeKind; label: string }[]).filter((item) => graph.nodes.some((node) => node.kind === item.kind))

  function svgPoint(clientX: number, clientY: number) {
    const svg = svgRef.current
    if (!svg) return { x: 0, y: 0 }
    const point = svg.createSVGPoint()
    point.x = clientX
    point.y = clientY
    const matrix = svg.getScreenCTM()
    return matrix ? point.matrixTransform(matrix.inverse()) : { x: 0, y: 0 }
  }

  function beginNodeDrag(event: PointerEvent<SVGGElement>, node: GraphNode) {
    event.stopPropagation()
    event.preventDefault()
    svgRef.current?.setPointerCapture(event.pointerId)
    gesture.current = { kind: 'node', id: node.id, startX: event.clientX, startY: event.clientY, startView: view }
  }

  function beginPan(event: PointerEvent<SVGSVGElement>) {
    if ((event.target as Element).closest('[data-graph-node]')) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    gesture.current = { kind: 'pan', startX: event.clientX, startY: event.clientY, startView: view }
  }

  function movePointer(event: PointerEvent<SVGSVGElement>) {
    const active = gesture.current
    if (!active) return
    const dx = event.clientX - active.startX
    const dy = event.clientY - active.startY
    if (active.kind === 'node' && active.id) {
      const point = svgPoint(event.clientX, event.clientY)
      setPositions((current) => ({ ...current,
        [active.id!]: { x: (point.x - active.startView.x) / active.startView.scale,
          y: (point.y - active.startView.y) / active.startView.scale } }))
      return
    }
    const svg = svgRef.current
    if (!svg) return
    const rect = svg.getBoundingClientRect()
    setView({ ...active.startView,
      x: active.startView.x + dx * graph.width / rect.width,
      y: active.startView.y + dy * HEIGHT / rect.height })
  }

  function endPointer(event: PointerEvent<SVGSVGElement>) {
    const active = gesture.current
    if (active?.kind === 'node' && active.id) setSelectedId(active.id)
    gesture.current = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
  }

  function zoom(factor: number) {
    setView((current) => {
      const scale = Math.max(0.65, Math.min(1.8, current.scale * factor))
      const centerX = graph.width / 2
      const x = centerX - (centerX - current.x) * scale / current.scale
      const y = CENTER.y - (CENTER.y - current.y) * scale / current.scale
      return { x, y, scale }
    })
  }

  function selectWithKeyboard(event: KeyboardEvent<SVGGElement>, node: GraphNode) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      setSelectedId(node.id)
    }
  }

  const detailSource = selected?.sourceId ? sources.find((source) => source.id === selected.sourceId) : undefined
  const edgePaths = graph.edges.map((edge) => {
    const from = positionMap.get(edge.from)
    const to = positionMap.get(edge.to)
    if (!from || !to) return null
    const middleX = (from.x + to.x) / 2
    const middleY = (from.y + to.y) / 2
    const bend = (edge.from.length + edge.to.length) % 2 ? 22 : -22
    const dx = to.x - from.x
    const dy = to.y - from.y
    const length = Math.max(1, Math.hypot(dx, dy))
    const controlX = middleX - dy / length * bend
    const controlY = middleY + dx / length * bend
    const connected = focusId === edge.from || focusId === edge.to
    return { edge, from, to, controlX, controlY, connected }
  })

  return (
    <section className="signal-graph-card" aria-label="Интерактивный граф связей сигнала">
      <div className="signal-graph-heading">
        <div>
          <span className="signal-graph-kicker">Карта доказательств</span>
          <h3>Связи сигнала</h3>
          <p>Показаны проверенные цитаты и подтверждённые события с их источниками. Перемещайте узлы, чтобы проследить связи.</p>
        </div>
        <div className="signal-graph-counts">
          <span><b>{graph.nodes.length - 1}</b> {russianCount(graph.nodes.length - 1, 'связанный элемент', 'связанных элемента', 'связанных элементов')}</span>
          <span><b>{graph.edges.length}</b> {russianCount(graph.edges.length, 'связь', 'связи', 'связей')}</span>
        </div>
      </div>

      <div className="signal-graph-toolbar" aria-label="Управление графом">
        <div className="signal-graph-legend">
          {legendItems.map((item) => <span key={item.kind}><i className={item.kind} />{item.label}</span>)}
        </div>
        <div className="signal-graph-controls">
          <button type="button" onClick={() => zoom(1.18)} aria-label="Увеличить граф">＋</button>
          <button type="button" onClick={() => zoom(1 / 1.18)} aria-label="Уменьшить граф">−</button>
          <button type="button" onClick={() => { setView(fitView(graph.nodes, graph.width)); setPositions(initialPositions(graph.nodes)) }}>Сбросить</button>
        </div>
      </div>

      <div className="signal-graph-layout">
        <div className="signal-graph-canvas-wrap">
          <svg ref={svgRef} className="signal-graph-svg" style={{ minWidth: `${graph.width}px` }}
            viewBox={`0 0 ${graph.width} ${HEIGHT}`} role="application"
            aria-label="Граф: перетаскивайте узлы, нажимайте для просмотра связей"
            onPointerDown={beginPan} onPointerMove={movePointer} onPointerUp={endPointer} onPointerCancel={endPointer}>
            <defs>
              <filter id="signal-graph-shadow" x="-30%" y="-30%" width="160%" height="170%">
                <feDropShadow dx="0" dy="5" stdDeviation="6" floodColor="#27342b" floodOpacity=".10" />
              </filter>
            </defs>
            <rect className="signal-graph-background" width={graph.width} height={HEIGHT} rx="16" />
            <g className="signal-graph-world" transform={`translate(${view.x} ${view.y}) scale(${view.scale})`}>
              <g className="signal-graph-edges">
                {edgePaths.map((item) => item && (
                  <g key={item.edge.id} className={`${item.connected ? 'connected' : ''}${focusId && !item.connected ? ' dimmed' : ''}`}>
                    <path d={`M ${item.from.x} ${item.from.y} Q ${item.controlX} ${item.controlY} ${item.to.x} ${item.to.y}`} />
                    {item.edge.label && item.connected && <text x={item.controlX} y={item.controlY - 7} textAnchor="middle">{item.edge.label}</text>}
                  </g>
                ))}
              </g>
              {graph.nodes.map((node) => {
                const pos = positionMap.get(node.id) ?? node
                const isSelected = selectedId === node.id
                const isNeighbor = neighborIds.has(node.id)
                const dimmed = Boolean(focusId && node.id !== focusId && !isNeighbor)
                return (
                  <g key={node.id} data-graph-node="true" className={`signal-graph-node ${node.kind} ${node.relation ? relationColor(node.relation) : ''}${isSelected ? ' selected' : ''}${dimmed ? ' dimmed' : ''}`}
                    transform={`translate(${pos.x} ${pos.y})`} role="button" tabIndex={0}
                    aria-label={`${node.label}. ${node.subtitle}`} aria-pressed={isSelected}
                    onPointerDown={(event) => beginNodeDrag(event, node)}
                    onClick={() => setSelectedId(node.id)}
                    onKeyDown={(event) => selectWithKeyboard(event, node)}
                    onMouseEnter={() => setHoveredId(node.id)} onMouseLeave={() => setHoveredId(null)}>
                    {node.kind === 'technology' ? (
                      <>
                        <rect className="signal-graph-node-halo" x="-102" y="-52" width="204" height="104" rx="24" />
                        <rect className="signal-graph-node-shape" x="-92" y="-42" width="184" height="84" rx="19" />
                        <text className="signal-graph-node-tag" y="-18" textAnchor="middle">ТЕХНОЛОГИЯ</text>
                        <text className="signal-graph-node-title" y="4" textAnchor="middle">{shortText(node.label, 20)}</text>
                        <text className="signal-graph-node-subtitle" y="22" textAnchor="middle">{shortText(node.subtitle, 25)}</text>
                      </>
                    ) : (
                      <>
                        <rect className="signal-graph-node-halo" x={node.kind === 'source' ? -87 : -92}
                          y={node.kind === 'organization' ? -28 : -34}
                          width={node.kind === 'source' ? 174 : 184} height={node.kind === 'organization' ? 56 : 68} rx="16" />
                        <rect className="signal-graph-node-shape" x={node.kind === 'source' ? -81 : -86}
                          y={node.kind === 'organization' ? -23 : -29}
                          width={node.kind === 'source' ? 162 : 172} height={node.kind === 'organization' ? 46 : 58} rx="13" />
                        <circle className="signal-graph-node-dot" cx={node.kind === 'source' ? -65 : -69} cy="0" r="5" />
                        <text className="signal-graph-node-title" x={node.kind === 'source' ? -52 : -56} y="-3">
                          {shortText(node.label, node.kind === 'source' ? 20 : 22)}
                        </text>
                        <text className="signal-graph-node-subtitle" x={node.kind === 'source' ? -52 : -56} y="15">
                          {shortText(node.subtitle, 23)}
                        </text>
                      </>
                    )}
                  </g>
                )
              })}
            </g>
            {showEmpty && <g className="signal-graph-empty" pointerEvents="none">
              <rect x={graph.width / 2 - 300} y="560" width="600" height="72" rx="14" />
              <text x={graph.width / 2} y="590" textAnchor="middle">Связанные доказательства ещё не собраны</text>
              <text x={graph.width / 2} y="614" textAnchor="middle">Когда появятся цитаты и события, они будут показаны здесь</text>
            </g>}
          </svg>
          <div className="signal-graph-pan-hint">Перетаскивайте фон или прокручивайте карту по горизонтали</div>
        </div>

        <aside className="signal-graph-detail" aria-live="polite">
          {selected?.kind === 'technology' && <>
            <span className="signal-graph-detail-type">Главный узел</span>
            <h4>{selected.label}</h4>
            <p>{selected.detail || 'Тема, по которой собраны доказательства.'}</p>
            <div className="signal-graph-detail-stats">
              <span>Доказательств: <b>{assessment.evidence?.length ?? 0}</b></span>
              <span>Источников: <b>{sources.length}</b></span>
            </div>
          </>}
          {selected?.kind === 'claim' && <>
            <span className={`signal-graph-detail-type ${relationColor(selected.relation)}`}>{relationText(selected.relation)}</span>
            <h4>{selected.claim}</h4>
            {selected.quote && <blockquote>{selected.quote}</blockquote>}
            {detailSource && <>
              <p className="signal-graph-detail-source">{detailSource.title}</p>
              <button type="button" className="signal-graph-open-source" onClick={() => onOpenSource(detailSource.id)}>Открыть источник ↗</button>
            </>}
          </>}
          {selected?.kind === 'source' && selected.source && <>
            <span className="signal-graph-detail-type">{selected.source.type_label || selected.source.type}</span>
            <h4>{selected.source.title}</h4>
            <p>{selected.source.publisher} · {dateText(selected.source.published_at)}</p>
            <p>{selected.source.trust_reason}</p>
            <button type="button" className="signal-graph-open-source" onClick={() => onOpenSource(selected.source!.id)}>Открыть документ ↗</button>
          </>}
          {selected?.kind === 'event' && <>
            <span className="signal-graph-detail-type">Событие</span>
            <h4>{EVENT_LABEL[selected.eventType || ''] || 'Событие'}</h4>
            <p>{selected.eventDate ? dateText(selected.eventDate) : 'Дата не установлена'}</p>
            {selected.company && <p>Компания: {selected.company}</p>}
            {selected.customer && <p>Заказчик: {selected.customer}</p>}
            {detailSource && <button type="button" className="signal-graph-open-source" onClick={() => onOpenSource(detailSource.id)}>Источник события ↗</button>}
          </>}
          {selected?.kind === 'organization' && <>
            <span className="signal-graph-detail-type">Организация</span>
            <h4>{selected.label}</h4>
            <p>Указана как {selected.organizationRole} в связанном событии.</p>
          </>}
          <div className="signal-graph-detail-help">Выберите узел на карте — связанные элементы подсветятся.</div>
        </aside>
      </div>
    </section>
  )
}
