/* <knowledge-graph> — canvas-граф знаний Hypothesis Lab.
   Режимы 2D (force-layout, пан/зум, миникарта) и 3D (орбита, глубина).
   Фокус на вершине + соседях, фильтр по типам, поиск, путь к KPI. */
(function () {
  if (customElements.get("knowledge-graph")) return;

  const TYPE = {
    hypothesis: { c: "#E09830", label: "Гипотезы" },
    property:   { c: "#F5F0E6", label: "Свойства / KPI" },
    material:   { c: "#C9AC80", label: "Материалы" },
    process:    { c: "#A08B6F", label: "Процессы" },
    reagent:    { c: "#B4915C", label: "Реагенты" },
    parameter:  { c: "#8CAF82", label: "Параметры" },
    metric:     { c: "#7E7668", label: "Метрики" },
    equipment:  { c: "#6C665C", label: "Оборудование" },
    risk:       { c: "#C27566", label: "Риски" },
    source:     { c: "#57524A", label: "Источники" },
    observation: { c: "#8B7F70", label: "Наблюдения" },
    concept:    { c: "#8B7F70", label: "Концепты" },
  };
  const GOLD = "#E09830", FG = "#F5F0E6";

  const CSS = `
    :host { display:block; position:relative; overflow:hidden; width:100%; height:100%; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    canvas.main { position:absolute; inset:0; width:100%; height:100%; cursor:grab; }
    canvas.main.grabbing { cursor:grabbing; }
    .bar { position:absolute; top:14px; left:14px; right:14px; display:flex; gap:8px; align-items:center; pointer-events:none; }
    .bar > * { pointer-events:auto; }
    .seg { display:flex; background:rgba(23,20,15,0.92); border:1px solid rgba(245,240,230,0.08); border-radius:999px; padding:3px; }
    .seg button { border:0; cursor:pointer; font:600 12px/1 'Onest',sans-serif; color:rgba(245,240,230,0.55); background:transparent; padding:7px 16px; border-radius:999px; transition:all 180ms cubic-bezier(0.2,0,0,1); }
    .seg button.on { color:#0C0B09; background:#E09830; }
    .seg button:not(.on):hover { color:#F5F0E6; }
    .search { display:flex; align-items:center; gap:7px; background:rgba(23,20,15,0.92); border:1px solid rgba(245,240,230,0.08); border-radius:999px; padding:0 14px; height:32px; width:210px; transition:border-color 180ms; }
    .search:focus-within { border-color:rgba(224,152,48,0.42); }
    .search svg { flex:none; opacity:0.4; }
    .search input { border:0; outline:0; background:transparent; color:#F5F0E6; font:500 12px 'Onest',sans-serif; width:100%; }
    .search input::placeholder { color:rgba(245,240,230,0.32); }
    .tool { width:32px; height:32px; flex:none; display:grid; place-items:center; background:rgba(23,20,15,0.92); border:1px solid rgba(245,240,230,0.08); border-radius:999px; color:rgba(245,240,230,0.6); cursor:pointer; transition:all 180ms cubic-bezier(0.2,0,0,1); }
    .tool:hover { color:#F5F0E6; border-color:rgba(245,240,230,0.16); }
    .tool:active { transform:scale(0.94); }
    .spacer { flex:1; pointer-events:none; }
    .hint { font:500 11px 'Onest',sans-serif; color:rgba(245,240,230,0.3); background:rgba(23,20,15,0.85); border-radius:999px; padding:7px 12px; white-space:nowrap; pointer-events:none !important; }
    .legend-toggle { position:absolute; top:58px; right:14px; z-index:4; height:32px; display:flex; align-items:center; gap:7px; padding:0 12px; border:1px solid rgba(245,240,230,0.08); border-radius:999px; background:rgba(23,20,15,0.92); color:rgba(245,240,230,0.68); font:700 11px/1 'Onest',sans-serif; cursor:pointer; }
    .legend-toggle i { position:relative; width:27px; height:7px; flex:none; border-radius:99px; background:transparent; }
    .legend-toggle i::before { content:""; position:absolute; inset:0 auto auto 0; width:7px; height:7px; border-radius:99px; background:#E09830; box-shadow:10px 0 0 #F5F0E6,20px 0 0 #C9AC80; }
    .legend-toggle.on { color:#F5F0E6; border-color:rgba(224,152,48,0.34); }
    .chips { position:absolute; top:96px; right:14px; z-index:3; width:220px; display:grid; gap:6px; max-height:210px; overflow:auto; padding:10px; border:1px solid rgba(245,240,230,0.08); border-radius:14px; background:rgba(14,13,11,0.88); backdrop-filter:blur(8px); transition:opacity 160ms, transform 160ms cubic-bezier(0.16,1,0.3,1); }
    .chips.hide { opacity:0; transform:translateY(-4px); pointer-events:none; }
    .legend-title { color:rgba(245,240,230,0.34); font:700 10px/1 'Onest',sans-serif; text-transform:uppercase; }
    .chip { display:flex; align-items:center; gap:6px; background:rgba(23,20,15,0.92); border:1px solid rgba(245,240,230,0.08); border-radius:999px; padding:5px 11px 5px 8px; font:500 11px 'Onest',sans-serif; color:rgba(245,240,230,0.68); cursor:pointer; transition:all 180ms cubic-bezier(0.2,0,0,1); }
    .chip i { width:7px; height:7px; border-radius:99px; flex:none; }
    .chip.off { opacity:0.35; }
    .chip:active { transform:scale(0.96); }
    .details { position:absolute; top:96px; right:14px; width:250px; background:rgba(14,13,11,0.94); border:1px solid rgba(245,240,230,0.1); border-radius:16px; padding:14px; display:grid; gap:8px; transition:opacity 180ms, transform 180ms cubic-bezier(0.16,1,0.3,1); }
    .details.hide { opacity:0; transform:translateY(-6px); pointer-events:none; }
    .details .ey { font:600 10px 'Onest',sans-serif; letter-spacing:0.22em; text-transform:uppercase; }
    .details h4 { margin:0; font:600 15px/1.25 'Onest',sans-serif; color:#F5F0E6; overflow-wrap:anywhere; }
    .details p { margin:0; font:500 12px/1.45 'Onest',sans-serif; color:rgba(245,240,230,0.55); }
    .details .deg { font:500 11px 'Space Grotesk',sans-serif; color:rgba(245,240,230,0.38); font-variant-numeric:tabular-nums; }
    .details .row { display:flex; gap:6px; margin-top:2px; }
    .details button { flex:1; border:0; cursor:pointer; border-radius:999px; padding:8px 0; font:600 11px 'Onest',sans-serif; transition:transform 180ms; }
    .details button:active { transform:scale(0.97); }
    .details .gold { background:#E09830; color:#0C0B09; }
    .details .ghost { background:transparent; border:1px solid rgba(245,240,230,0.14); color:rgba(245,240,230,0.75); }
    .details .x { position:absolute; top:8px; right:8px; width:24px; height:24px; flex:none; display:grid; place-items:center; background:transparent; border:0; color:rgba(245,240,230,0.4); cursor:pointer; border-radius:99px; padding:0; }
    .details .x:hover { color:#F5F0E6; }
    canvas.mini { position:absolute; right:14px; bottom:14px; width:148px; height:100px; border-radius:12px; border:1px solid rgba(245,240,230,0.08); background:rgba(14,13,11,0.88); cursor:pointer; transition:opacity 260ms; }
    canvas.mini.hide { opacity:0; pointer-events:none; }
    @media (max-width: 760px) {
      .legend-toggle { top:104px; }
      .chips { left:14px; right:14px; top:142px; width:auto; display:flex; flex-wrap:wrap; max-height:74px; }
      .legend-title { width:100%; }
      .details { top:190px; right:14px; width:min(250px, calc(100% - 28px)); }
      .hint { display:none; }
    }
  `;

  const SEARCH_SVG = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#F5F0E6" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"></circle><path d="M21 21l-4.2-4.2"></path></svg>`;
  const FIT_SVG = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9V5.5A1.5 1.5 0 0 1 5.5 4H9"></path><path d="M15 4h3.5A1.5 1.5 0 0 1 20 5.5V9"></path><path d="M20 15v3.5a1.5 1.5 0 0 1-1.5 1.5H15"></path><path d="M9 20H5.5A1.5 1.5 0 0 1 4 18.5V15"></path></svg>`;
  const X_SVG = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"></path></svg>`;

  class KnowledgeGraph extends HTMLElement {
    connectedCallback() {
      if (this._init) return;
      this._init = true;
      this.opts = { autoRotate: true, dimOpacity: 0.14, minimap: true };
      this.mode = "2d";
      this.transform = { x: 0, y: 0, k: 1 };
      this.rot = { x: -0.35, y: 0.6 };
      this.zoom3 = 1;
      this.focusSet = null; this.pathEdges = null;
      this.hidden_ = new Set();
      this.hoverNode = null; this.selNode = null;
      this.lastInteract = 0; this.dash = 0;

      this.kpiId = null;
      this.nodes = [];
      this.edges = [];
      this.byId = new Map();
      this.adj = new Map();

      this.attachShadow({ mode: "open" });
      this.shadowRoot.innerHTML = `
        <style>${CSS}</style>
        <canvas class="main"></canvas>
        <div class="bar">
          <div class="seg"><button data-m="2d" class="on">2D</button><button data-m="3d">3D</button></div>
          <div class="search">${SEARCH_SVG}<input placeholder="Поиск по графу" /></div>
          <button class="tool" title="Выравнивание графа">${FIT_SVG}</button>
          <div class="spacer"></div>
          <div class="hint"></div>
        </div>
        <button class="legend-toggle" type="button"><i></i>Цвета</button>
        <div class="chips hide"></div>
        <div class="details hide"></div>
        <canvas class="mini"></canvas>`;

      this.cv = this.shadowRoot.querySelector("canvas.main");
      this.ctx = this.cv.getContext("2d");
      this.mini = this.shadowRoot.querySelector("canvas.mini");
      this.mctx = this.mini.getContext("2d");
      this.detailsEl = this.shadowRoot.querySelector(".details");
      this.hintEl = this.shadowRoot.querySelector(".hint");
      this.legendEl = this.shadowRoot.querySelector(".chips");
      this.legendButton = this.shadowRoot.querySelector(".legend-toggle");

      this.shadowRoot.querySelectorAll(".seg button").forEach((b) =>
        b.addEventListener("click", () => this.setMode(b.dataset.m)));
      this.shadowRoot.querySelector(".tool").addEventListener("click", () => this.fit(true));
      this.shadowRoot.querySelector(".search input").addEventListener("input", (e) => this.search(e.target.value));
      this.legendButton.addEventListener("click", () => this.toggleLegend());

      this.setData((window.HL_DATA && window.HL_DATA.graph) || { nodes: [], edges: [] }, { fit: false });

      this.cv.addEventListener("pointerdown", (e) => this.onDown(e));
      this.cv.addEventListener("pointermove", (e) => this.onMove(e));
      this.cv.addEventListener("pointerup", (e) => this.onUp(e));
      this.cv.addEventListener("pointercancel", (e) => this.onUp(e));
      this.cv.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
      this.mini.addEventListener("pointerdown", (e) => this.onMini(e));

      if (!this.clientHeight) { this.style.width = "100%"; this.style.height = "100%"; }
      this.ro = new ResizeObserver(() => { this.resize(); });
      this.ro.observe(this);
      this.resize();
      this.alpha = 1;
      this.updateHint();
      const loop = () => { this.tick(); this.raf = requestAnimationFrame(loop); };
      this.raf = requestAnimationFrame(loop);
      setTimeout(() => this.fit(false), 60);
    }

    disconnectedCallback() { cancelAnimationFrame(this.raf); this.ro && this.ro.disconnect(); }

    /* ─── публичное API ─── */
    setOptions(o) {
      Object.assign(this.opts, o || {});
      this.mini?.classList.toggle("hide", !(this.opts.minimap && this.mode === "2d"));
      this.applyAlphaTargets();
    }
    setData(data, options = {}) {
      const graph = data || { nodes: [], edges: [] };
      this.kpiId = graph.kpi || null;
      this.focusSet = null;
      this.pathEdges = null;
      this.hoverNode = null;
      this.selNode = null;
      this.hidden_ = new Set([...this.hidden_].filter((type) => (graph.nodes || []).some((node) => node.type === type)));
      this.nodes = (graph.nodes || []).map((n, index) => ({
        ...n,
        id: String(n.id || n.label || `node:${index}`),
        label: String(n.label || n.id || "Узел"),
        type: String(n.type || "concept"),
        weight: Number(n.weight || 2),
        x: 0,
        y: 0,
        vx: 0,
        vy: 0,
        X: 0,
        Y: 0,
        Z: 0,
        a: 1,
        ta: 1,
        r: 7 + Number(n.weight || 2) * 2.1,
        px: 0,
        py: 0,
        ps: 1,
      }));
      this.byId = new Map(this.nodes.map((n) => [n.id, n]));
      this.edges = (graph.edges || [])
        .filter((e) => this.byId.has(e.source) && this.byId.has(e.target))
        .map((e) => ({ ...e, id: String(e.id || `${e.source}:${e.target}`), s: this.byId.get(e.source), t: this.byId.get(e.target), a: 1, ta: 1 }));
      this.adj = new Map();
      for (const n of this.nodes) this.adj.set(n.id, []);
      for (const e of this.edges) {
        this.adj.get(e.s.id)?.push(e);
        this.adj.get(e.t.id)?.push(e);
      }
      this._laid3d = false;
      this.rot = { x: -0.35, y: 0.6 };
      this.zoom3 = 1;
      this.transform = { x: 0, y: 0, k: 1 };
      this.renderChips();
      this.initLayout();
      this.applyAlphaTargets();
      this.hideDetails();
      this.alpha = 1;
      if (options.fit !== false) setTimeout(() => this.fit(false), 30);
    }
    renderChips() {
      const chips = this.shadowRoot?.querySelector(".chips");
      if (!chips) return;
      chips.innerHTML = `<div class="legend-title">Цвета графа</div>`;
      const present = new Set(this.nodes.map((n) => n.type));
      for (const t of Object.keys(TYPE)) {
        const c = document.createElement("button");
        c.className = "chip";
        c.classList.toggle("off", this.hidden_.has(t) || !present.has(t));
        c.innerHTML = `<i style="background:${TYPE[t].c}"></i>${TYPE[t].label}`;
        c.addEventListener("click", () => {
          if (!present.has(t)) return;
          if (this.hidden_.has(t)) this.hidden_.delete(t);
          else this.hidden_.add(t);
          c.classList.toggle("off", this.hidden_.has(t));
          this.applyAlphaTargets();
        });
        chips.appendChild(c);
      }
    }
    setBarLeft(px) {
      const bar = this.shadowRoot.querySelector(".bar"), chips = this.shadowRoot.querySelector(".chips");
      bar.style.left = 14 + px + "px";
      chips.style.right = 14 + "px";
    }
    toggleLegend(force) {
      const open = force === undefined ? this.legendEl.classList.contains("hide") : Boolean(force);
      this.legendEl.classList.toggle("hide", !open);
      this.legendButton.classList.toggle("on", open);
      if (open) this.hideDetails();
    }
    setMode(m) {
      if (m === this.mode) return;
      this.mode = m;
      this.shadowRoot.querySelectorAll(".seg button").forEach((b) => b.classList.toggle("on", b.dataset.m === m));
      this.mini?.classList.toggle("hide", !(this.opts.minimap && m === "2d"));
      if (m === "3d" && !this._laid3d) { this.layout3d(); this._laid3d = true; }
      if (m === "2d") this.fit(false);
      this.updateHint();
    }
    focusNode(id, opts) {
      const n = this.byId.get(id);
      if (!n) return;
      const set = new Set([id]);
      const related = this.adj.get(id) || [];
      for (const e of related) { set.add(e.s.id); set.add(e.t.id); }
      this.focusSet = set;
      this.pathEdges = new Set(related.map((e) => e.id));
      this.selNode = n;
      this.applyAlphaTargets();
      if (!(opts && opts.quiet)) this.showDetails(n);
      this.center(n);
    }
    highlightPath(fromId) {
      if (!this.kpiId) { this.focusNode(fromId); return; }
      const path = this.bfs(fromId, this.kpiId);
      if (!path) { this.focusNode(fromId); return; }
      const set = new Set(); const eset = new Set();
      for (let i = 0; i < path.length; i++) {
        set.add(path[i]);
        if (i) { const e = this.edgeBetween(path[i - 1], path[i]); if (e) eset.add(e.id); }
      }
      this.focusSet = set; this.pathEdges = eset;
      this.selNode = this.byId.get(fromId);
      this.applyAlphaTargets();
      if (this.selNode) {
        this.showDetails(this.selNode, "Путь к KPI");
        this.center(this.selNode);
      }
    }
    clearFocus() {
      this.focusSet = null; this.pathEdges = null; this.selNode = null;
      this.applyAlphaTargets(); this.hideDetails();
    }
    search(q) {
      q = (q || "").trim().toLowerCase();
      if (!q) { this.focusSet = null; this.pathEdges = null; this.applyAlphaTargets(); return; }
      const hits = this.nodes.filter((n) => String(n.label || "").toLowerCase().includes(q));
      this.focusSet = new Set(hits.map((n) => n.id));
      this.pathEdges = new Set();
      this.applyAlphaTargets();
      if (hits.length) this.center(hits[0]);
    }

    /* ─── графовые утилиты ─── */
    edgeBetween(a, b) { return (this.adj.get(a) || []).find((e) => (e.s.id === a && e.t.id === b) || (e.s.id === b && e.t.id === a)); }
    bfs(from, to) {
      if (from === to) return [from];
      const prev = new Map([[from, null]]);
      const q = [from];
      while (q.length) {
        const cur = q.shift();
        for (const e of this.adj.get(cur) || []) {
          const nxt = e.s.id === cur ? e.t.id : e.s.id;
          if (prev.has(nxt)) continue;
          prev.set(nxt, cur);
          if (nxt === to) { const path = [to]; let p = cur; while (p !== null) { path.unshift(p); p = prev.get(p); } return path; }
          q.push(nxt);
        }
      }
      return null;
    }
    applyAlphaTargets() {
      const dim = this.opts.dimOpacity;
      for (const n of this.nodes) {
        if (this.hidden_.has(n.type)) n.ta = 0.04;
        else if (!this.focusSet) n.ta = 1;
        else n.ta = this.focusSet.has(n.id) ? 1 : dim;
      }
      for (const e of this.edges) {
        const hidden = this.hidden_.has(e.s.type) || this.hidden_.has(e.t.type);
        if (hidden) e.ta = 0.02;
        else if (!this.focusSet) e.ta = 0.5;
        else e.ta = this.pathEdges && this.pathEdges.has(e.id) ? 1 : dim * 0.4;
      }
    }

    /* ─── раскладка ─── */
    initLayout() {
      const N = this.nodes.length;
      if (!N) return;
      this.nodes.forEach((n, i) => {
        const ang = (i / N) * Math.PI * 2;
        const rad = 160 + (i % 5) * 40;
        n.x = Math.cos(ang) * rad; n.y = Math.sin(ang) * rad;
      });
      for (let i = 0; i < 260; i++) this.step2d(1 - i / 260);
    }
    step2d(alpha) {
      const ns = this.nodes;
      for (let i = 0; i < ns.length; i++) for (let j = i + 1; j < ns.length; j++) {
        const a = ns[i], b = ns[j];
        let dx = b.x - a.x, dy = b.y - a.y;
        let d2 = dx * dx + dy * dy || 1;
        if (d2 < 90000) {
          const d = Math.sqrt(d2);
          const f = (2600 * alpha) / d2;
          dx /= d; dy /= d;
          a.x -= dx * f * 9; a.y -= dy * f * 9;
          b.x += dx * f * 9; b.y += dy * f * 9;
        }
      }
      for (const e of this.edges) {
        const dx = e.t.x - e.s.x, dy = e.t.y - e.s.y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1;
        const target = 120 + (e.s.r + e.t.r);
        const f = ((d - target) / d) * 0.05 * alpha * 9;
        if (e.s !== this.dragNode) { e.s.x += dx * f; e.s.y += dy * f; }
        if (e.t !== this.dragNode) { e.t.x -= dx * f; e.t.y -= dy * f; }
      }
      for (const n of ns) { if (n === this.dragNode) continue; n.x -= n.x * 0.012 * alpha * 9; n.y -= n.y * 0.012 * alpha * 9; }
    }
    layout3d() {
      const N = this.nodes.length;
      if (!N) return;
      this.nodes.forEach((n, i) => {
        const phi = Math.acos(1 - 2 * (i + 0.5) / N);
        const th = Math.PI * (1 + Math.sqrt(5)) * i;
        const R = 240;
        n.X = R * Math.sin(phi) * Math.cos(th); n.Y = R * Math.sin(phi) * Math.sin(th); n.Z = R * Math.cos(phi);
      });
      for (let it = 0; it < 200; it++) {
        const al = 1 - it / 200;
        const ns = this.nodes;
        for (let i = 0; i < ns.length; i++) for (let j = i + 1; j < ns.length; j++) {
          const a = ns[i], b = ns[j];
          let dx = b.X - a.X, dy = b.Y - a.Y, dz = b.Z - a.Z;
          let d2 = dx * dx + dy * dy + dz * dz || 1;
          if (d2 < 160000) {
            const d = Math.sqrt(d2), f = (5200 * al) / d2;
            dx /= d; dy /= d; dz /= d;
            a.X -= dx * f; a.Y -= dy * f; a.Z -= dz * f;
            b.X += dx * f; b.Y += dy * f; b.Z += dz * f;
          }
        }
        for (const e of this.edges) {
          const dx = e.t.X - e.s.X, dy = e.t.Y - e.s.Y, dz = e.t.Z - e.s.Z;
          const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
          const f = ((d - 150) / d) * 0.06 * al;
          e.s.X += dx * f; e.s.Y += dy * f; e.s.Z += dz * f;
          e.t.X -= dx * f; e.t.Y -= dy * f; e.t.Z -= dz * f;
        }
        for (const n of ns) { n.X *= 1 - 0.004 * al; n.Y *= 1 - 0.004 * al; n.Z *= 1 - 0.004 * al; }
      }
    }

    /* ─── взаимодействие ─── */
    pt(e) {
      const r = this.cv.getBoundingClientRect();
      const sx = r.width ? this.cv.clientWidth / r.width : 1;
      const sy = r.height ? this.cv.clientHeight / r.height : 1;
      return { x: (e.clientX - r.left) * sx, y: (e.clientY - r.top) * sy };
    }
    nodeAt(p) {
      let best = null, bd = 1e9;
      for (const n of this.nodes) {
        if (n.a < 0.06) continue;
        const dx = p.x - n.px, dy = p.y - n.py;
        const rr = Math.max(10, n.r * n.ps) + 4;
        const d = dx * dx + dy * dy;
        if (d < rr * rr && d < bd) { bd = d; best = n; }
      }
      return best;
    }
    onDown(e) {
      this.lastInteract = performance.now();
      this.cv.setPointerCapture(e.pointerId);
      const p = this.pt(e);
      this.downP = p; this.moved = false;
      const n = this.nodeAt(p);
      this.pendingNode = this.mode === "2d" ? n : null;
      if (!this.pendingNode) this.panStart = { ...p, tx: this.transform.x, ty: this.transform.y, rx: this.rot.x, ry: this.rot.y };
      this.cv.classList.add("grabbing");
    }
    onMove(e) {
      const p = this.pt(e);
      const movedDistance = this.downP ? Math.abs(p.x - this.downP.x) + Math.abs(p.y - this.downP.y) : 0;
      if (movedDistance > 5) this.moved = true;
      if (this.pendingNode && this.moved && !this.dragNode) {
        this.dragNode = this.pendingNode;
        this.alpha = Math.max(this.alpha, 0.35);
      }
      if (this.dragNode) {
        const k = this.transform.k;
        this.dragNode.x = (p.x - this.transform.x) / k;
        this.dragNode.y = (p.y - this.transform.y) / k;
        this.alpha = Math.max(this.alpha, 0.3);
        this.lastInteract = performance.now();
        return;
      }
      if (this.panStart) {
        this.lastInteract = performance.now();
        if (this.mode === "2d") {
          this.transform.x = this.panStart.tx + (p.x - this.panStart.x);
          this.transform.y = this.panStart.ty + (p.y - this.panStart.y);
        } else {
          this.rot.y = this.panStart.ry + (p.x - this.panStart.x) * 0.006;
          this.rot.x = Math.max(-1.4, Math.min(1.4, this.panStart.rx + (p.y - this.panStart.y) * 0.006));
        }
        return;
      }
      const n = this.nodeAt(p);
      if (n !== this.hoverNode) { this.hoverNode = n; this.cv.style.cursor = n ? "pointer" : "grab"; }
    }
    onUp(e) {
      this.cv.classList.remove("grabbing");
      const wasDrag = this.dragNode, p = this.pt(e);
      this.dragNode = null; this.panStart = null;
      if (!this.moved && this.downP) {
        const n = this.pendingNode || this.nodeAt(p);
        if (n) { this.focusNode(n.id); this.dispatchEvent(new CustomEvent("nodeselect", { detail: { id: n.id }, bubbles: true, composed: true })); }
        else if (!wasDrag) this.clearFocus();
      }
      this.pendingNode = null;
      this.downP = null;
    }
    onWheel(e) {
      e.preventDefault();
      this.lastInteract = performance.now();
      const f = Math.exp(-e.deltaY * 0.0016);
      if (this.mode === "2d") {
        const p = this.pt(e);
        const k2 = Math.max(0.25, Math.min(3.2, this.transform.k * f));
        const ratio = k2 / this.transform.k;
        this.transform.x = p.x - (p.x - this.transform.x) * ratio;
        this.transform.y = p.y - (p.y - this.transform.y) * ratio;
        this.transform.k = k2;
      } else {
        this.zoom3 = Math.max(0.4, Math.min(2.6, this.zoom3 * f));
      }
    }
    onMini(e) {
      const r = this.mini.getBoundingClientRect();
      const mx = r.width ? (e.clientX - r.left) / r.width : 0.5, my = r.height ? (e.clientY - r.top) / r.height : 0.5;
      const b = this.bounds();
      const wx = b.x0 + mx * (b.x1 - b.x0), wy = b.y0 + my * (b.y1 - b.y0);
      const area = this.visibleArea();
      const w = area.width, h = area.height;
      this.transform.x = area.x + w / 2 - wx * this.transform.k;
      this.transform.y = area.y + h / 2 - wy * this.transform.k;
    }
    center(n) {
      if (this.mode !== "2d") return;
      const area = this.visibleArea();
      this.centerAnim = {
        n,
        sx: this.transform.x,
        sy: this.transform.y,
        tx: area.x + area.width / 2 - n.x * this.transform.k,
        ty: area.y + area.height / 2 - n.y * this.transform.k,
        t: 0,
      };
    }
    bounds() {
      if (!this.nodes.length) return { x0: -160, y0: -110, x1: 160, y1: 110 };
      let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
      for (const n of this.nodes) { x0 = Math.min(x0, n.x); y0 = Math.min(y0, n.y); x1 = Math.max(x1, n.x); y1 = Math.max(y1, n.y); }
      return { x0: x0 - 60, y0: y0 - 60, x1: x1 + 60, y1: y1 + 60 };
    }
    fit(anim) {
      const area = this.visibleArea();
      const w = area.width, h = area.height;
      if (!w || !h) return;
      if (!this.nodes.length) {
        this.transform = { x: area.x + w / 2, y: area.y + h / 2, k: 1 };
        return;
      }
      const b = this.bounds();
      const k = Math.min(1.6, Math.min(w / (b.x1 - b.x0), h / (b.y1 - b.y0)) * 0.94);
      const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
      this.transform = { x: area.x + w / 2 - cx * k, y: area.y + h / 2 - cy * k, k };
      if (this.mode === "3d") { this.zoom3 = 1; }
    }
    visibleArea() {
      const w = this.cv.clientWidth || this.clientWidth || 1;
      const h = this.cv.clientHeight || this.clientHeight || 1;
      const top = this.mode === "3d" ? 56 : 116;
      const bottom = this.mode === "3d" ? 220 : 160;
      const right = w > 760 ? 250 : 0;
      const usableH = Math.max(220, h - top - bottom);
      return { x: 0, y: top, width: Math.max(320, w - right), height: usableH, cx: (w - right) / 2, cy: top + usableH / 2 };
    }

    /* ─── детали ─── */
    showDetails(n, note) {
      if (!n) return;
      this.toggleLegend(false);
      const t = TYPE[n.type] || { c: FG, label: n.type };
      const deg = (this.adj.get(n.id) || []).length;
      const isKpi = n.id === this.kpiId;
      this.detailsEl.innerHTML = `
        <button class="x">${X_SVG}</button>
        <span class="ey" style="color:${t.c}">${this.esc(t.label)}${isKpi ? " · KPI" : ""}</span>
        <h4>${this.esc(n.label)}</h4>
        <p>${this.esc(n.summary || "")}</p>
        <span class="deg">${deg} ${deg === 1 ? "связь" : deg < 5 ? "связи" : "связей"}${note ? " · " + this.esc(note) : ""}</span>
        ${isKpi ? "" : `<div class="row"><button class="gold">Путь к KPI</button></div>`}`;
      this.detailsEl.classList.remove("hide");
      this.detailsEl.querySelector(".x").addEventListener("click", () => this.clearFocus());
      const g = this.detailsEl.querySelector(".gold");
      if (g) g.addEventListener("click", () => this.highlightPath(n.id));
    }
    hideDetails() { this.detailsEl.classList.add("hide"); }
    updateHint() {
      this.hintEl.textContent = this.mode === "2d"
        ? "Клик по вершине — фокус на связях · колесо — зум"
        : "Тяните, чтобы вращать · колесо — приближение";
    }

    /* ─── рендер ─── */
    resize() {
      const dpr = window.devicePixelRatio || 1;
      const w = this.clientWidth, h = this.clientHeight;
      if (!w || !h) return;
      this.cv.width = w * dpr; this.cv.height = h * dpr;
      this.mini.width = 148 * dpr; this.mini.height = 100 * dpr;
      this.dpr = dpr;
    }
    tick() {
      if (this.alpha > 0.02 && this.mode === "2d") { this.step2d(this.alpha); this.alpha *= 0.97; }
      if (this.centerAnim) {
        const c = this.centerAnim; c.t += 0.07;
        const e = 1 - Math.pow(1 - Math.min(1, c.t), 3);
        this.transform.x = c.sx + (c.tx - c.sx) * e;
        this.transform.y = c.sy + (c.ty - c.sy) * e;
        if (c.t >= 1) this.centerAnim = null;
      }
      if (this.mode === "3d" && this.opts.autoRotate && performance.now() - this.lastInteract > 2600 && !this.focusSet) this.rot.y += 0.0016;
      for (const n of this.nodes) n.a += (n.ta - n.a) * 0.14;
      for (const e of this.edges) e.a += (e.ta - e.a) * 0.14;
      this.dash = (this.dash + 0.55) % 24;
      this.draw();
    }
    project() {
      const area = this.visibleArea();
      const w = area.width;
      const cy = Math.cos(this.rot.y), sy = Math.sin(this.rot.y);
      const cx = Math.cos(this.rot.x), sx = Math.sin(this.rot.x);
      const f = 780;
      for (const n of this.nodes) {
        const x1 = n.X * cy + n.Z * sy;
        const z1 = -n.X * sy + n.Z * cy;
        const y2 = n.Y * cx - z1 * sx;
        const z2 = n.Y * sx + z1 * cx;
        const s = (f / (f + z2)) * this.zoom3;
        n.px = area.x + w / 2 + x1 * s; n.py = area.cy + y2 * s; n.ps = s; n.pz = z2;
      }
    }
    draw() {
      const ctx = this.ctx, dpr = this.dpr || 1;
      const w = this.cv.clientWidth, h = this.cv.clientHeight;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      /* точечная сетка */
      ctx.fillStyle = "rgba(245,240,230,0.045)";
      const gs = 26;
      const ox = this.mode === "2d" ? ((this.transform.x % gs) + gs) % gs : 0;
      const oy = this.mode === "2d" ? ((this.transform.y % gs) + gs) % gs : 0;
      for (let x = ox; x < w; x += gs) for (let y = oy; y < h; y += gs) ctx.fillRect(x, y, 1, 1);

      if (!this.nodes.length) {
        ctx.fillStyle = "rgba(245,240,230,0.35)";
        ctx.font = "600 13px Inter, ui-sans-serif, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Граф появится после загрузки источников", w / 2, h / 2);
        this.mini?.classList.add("hide");
        return;
      }
      this.mini?.classList.toggle("hide", !(this.opts.minimap && this.mode === "2d"));

      if (this.mode === "2d") {
        const { x: tx, y: ty, k } = this.transform;
        for (const n of this.nodes) { n.px = n.x * k + tx; n.py = n.y * k + ty; n.ps = k; }
      } else this.project();

      const order = this.mode === "3d" ? [...this.nodes].sort((a, b) => b.pz - a.pz) : this.nodes;

      /* рёбра */
      for (const e of this.edges) {
        const a = Math.min(e.a, (e.s.a + e.t.a) / 2);
        if (a < 0.02) continue;
        const onPath = this.pathEdges && this.pathEdges.has(e.id) && this.focusSet;
        const depth = this.mode === "3d" ? Math.max(0.25, Math.min(1, (e.s.ps + e.t.ps) / 2)) : 1;
        ctx.beginPath();
        ctx.moveTo(e.s.px, e.s.py); ctx.lineTo(e.t.px, e.t.py);
        if (onPath) {
          ctx.strokeStyle = `rgba(224,152,48,${0.9 * a * depth})`;
          ctx.lineWidth = 1.8;
          ctx.setLineDash([7, 5]);
          ctx.lineDashOffset = -this.dash;
        } else {
          ctx.strokeStyle = `rgba(245,240,230,${0.16 * a * depth})`;
          ctx.lineWidth = 1;
          ctx.setLineDash([]);
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);

      /* вершины */
      ctx.textAlign = "center";
      const labelBoxes = [];
      for (const n of order) {
        if (n.a < 0.02) continue;
        const r = Math.max(3.5, n.r * (this.mode === "3d" ? n.ps * 0.85 : Math.min(1.15, n.ps)));
        const col = (TYPE[n.type] || {}).c || FG;
        const depth = this.mode === "3d" ? Math.max(0.3, Math.min(1, n.ps)) : 1;
        const a = n.a * depth;
        const focused = this.focusSet && this.focusSet.has(n.id);
        const isSel = this.selNode === n;

        if ((focused && (n.type === "hypothesis" || n.id === this.kpiId)) || isSel) {
          ctx.beginPath(); ctx.arc(n.px, n.py, r + 7, 0, 7);
          ctx.strokeStyle = `rgba(224,152,48,${0.4 * a})`; ctx.lineWidth = 1.5; ctx.stroke();
        }
        ctx.beginPath(); ctx.arc(n.px, n.py, r, 0, 7);
        ctx.globalAlpha = a;
        ctx.fillStyle = "#0C0B09"; ctx.fill();
        ctx.fillStyle = this.hexA(col, n.type === "source" ? 0.5 : 0.92); ctx.fill();
        ctx.strokeStyle = this.hexA(col, 1); ctx.lineWidth = n === this.hoverNode ? 2.2 : 1.2; ctx.stroke();
        ctx.globalAlpha = 1;

        /* подпись */
        const showLabel = this.mode === "2d"
          ? (a > 0.5 && (this.transform.k > 0.82 || n.weight > 5 || focused || isSel))
          : (a > 0.5 && (n.ps > 0.92 || focused));
        if (showLabel) {
          const fs = this.mode === "3d" ? Math.round(11 * Math.min(1.15, n.ps)) : 11;
          ctx.font = `500 ${fs}px Inter, ui-sans-serif, system-ui, sans-serif`;
          const label = this.trim(n.label, focused || isSel ? 58 : 34);
          const tw = ctx.measureText(label).width;
          const ly = n.py + r + 7;
          const box = { x: n.px - tw / 2 - 5, y: ly, w: tw + 10, h: fs + 8 };
          if (!focused && !isSel && labelBoxes.some((other) => this.intersects(box, other))) continue;
          labelBoxes.push(box);
          ctx.globalAlpha = a;
          ctx.fillStyle = "rgba(12,11,9,0.82)";
          ctx.fillRect(box.x, box.y, box.w, box.h);
          ctx.fillStyle = focused || n.weight > 5.5 ? FG : "rgba(245,240,230,0.62)";
          ctx.fillText(label, n.px, ly + fs + 1);
          ctx.globalAlpha = 1;
        }
      }

      /* миникарта */
      if (this.opts.minimap && this.mode === "2d") this.drawMini();
    }
    drawMini() {
      const m = this.mctx, dpr = this.dpr || 1;
      m.setTransform(dpr, 0, 0, dpr, 0, 0);
      m.clearRect(0, 0, 148, 100);
      const b = this.bounds();
      const sx = 136 / (b.x1 - b.x0), sy = 88 / (b.y1 - b.y0);
      const s = Math.min(sx, sy);
      const map = (x, y) => [6 + (x - b.x0) * s, 6 + (y - b.y0) * s];
      m.strokeStyle = "rgba(245,240,230,0.1)"; m.lineWidth = 0.5;
      for (const e of this.edges) {
        if (e.a < 0.1) continue;
        const [x1, y1] = map(e.s.x, e.s.y), [x2, y2] = map(e.t.x, e.t.y);
        m.beginPath(); m.moveTo(x1, y1); m.lineTo(x2, y2); m.stroke();
      }
      for (const n of this.nodes) {
        if (n.a < 0.1) continue;
        const [x, y] = map(n.x, n.y);
        m.fillStyle = this.hexA((TYPE[n.type] || {}).c || FG, Math.min(1, n.a));
        m.beginPath(); m.arc(x, y, n.weight > 5 ? 2.4 : 1.6, 0, 7); m.fill();
      }
      /* видимая область */
      const { x: tx, y: ty, k } = this.transform;
      const area = this.visibleArea();
      const [vx0, vy0] = map((area.x - tx) / k, (area.y - ty) / k);
      const [vx1, vy1] = map((area.x + area.width - tx) / k, (area.y + area.height - ty) / k);
      m.strokeStyle = "rgba(224,152,48,0.55)"; m.lineWidth = 1;
      m.strokeRect(vx0, vy0, vx1 - vx0, vy1 - vy0);
    }
    hexA(hex, a) {
      const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
      return `rgba(${r},${g},${b},${a})`;
    }
    trim(value, length) {
      const text = String(value || "");
      return text.length > length ? `${text.slice(0, length - 1)}…` : text;
    }
    intersects(a, b) {
      return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
    }
    esc(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
  }
  customElements.define("knowledge-graph", KnowledgeGraph);
})();
