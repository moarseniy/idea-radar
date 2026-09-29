const state = {
  projects: [],
  projectId: null,
  state: null,
  weights: { novelty: 0.25, feasibility: 0.25, impact: 0.35, risk: 0.15 },
  openHypothesisId: null,
  dockTab: "docs",
  dockExpanded: false,
  graphKey: "",
  promptFiles: [],
  settingsTab: "project",
  busyTimer: null,
};

const graphView = {
  key: "",
  nodes: [],
  edges: [],
  nodeById: new Map(),
  edgeById: new Map(),
  transform: { x: 0, y: 0, k: 1 },
  selected: null,
  width: 0,
  height: 0,
  dragging: null,
  panning: null,
  moved: false,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

document.addEventListener("DOMContentLoaded", init);

async function init() {
  $("#actorInput").value = localStorage.getItem("hl.actor") || "User";
  renderActorBadge();
  bindEvents();
  await refreshProjects();
}

function bindEvents() {
  $("#actorInput").addEventListener("input", (event) => {
    localStorage.setItem("hl.actor", event.target.value.trim() || "User");
    renderActorBadge();
  });

  $("#projectSelect").addEventListener("change", async (event) => {
    state.projectId = event.target.value;
    localStorage.setItem("hl.projectId", state.projectId);
    await loadState();
  });

  $("#newProjectBtn").addEventListener("click", createProject);
  $("#projectForm").addEventListener("submit", saveProject);
  $("#deleteProjectBtn").addEventListener("click", deleteProject);
  $("#uploadBtn").addEventListener("click", () => $("#fileInput").click());
  $("#fileInput").addEventListener("change", uploadFiles);
  $("#promptFilesBtn").addEventListener("click", () => $("#promptFileInput").click());
  $("#promptFileInput").addEventListener("change", selectPromptFiles);
  $("#generateBtn").addEventListener("click", (event) => {
    if (event.currentTarget.dataset.mode === "clear") {
      clearHypotheses();
    } else {
      generateHypotheses();
    }
  });
  $("#exportCsvBtn").addEventListener("click", () => openExport("csv"));
  $("#exportMdBtn").addEventListener("click", () => openExport("md"));
  $("#exportPdfBtn").addEventListener("click", () => openExport("pdf"));
  $("#chatForm").addEventListener("submit", sendChat);
  $("#chatInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("#chatForm").requestSubmit();
    }
  });
  $("#graphZoomOut")?.addEventListener("click", () => zoomGraph(0.82));
  $("#graphZoomIn")?.addEventListener("click", () => zoomGraph(1.22));
  $("#graphFit")?.addEventListener("click", () => fitGraphToViewport(true));

  const graphSvg = $("#graphSvg");
  if (graphSvg) {
    graphSvg.addEventListener("wheel", onGraphWheel, { passive: false });
    graphSvg.addEventListener("pointerdown", onGraphPointerDown);
    graphSvg.addEventListener("pointermove", onGraphPointerMove);
    graphSvg.addEventListener("pointerup", onGraphPointerUp);
    graphSvg.addEventListener("pointercancel", onGraphPointerUp);
  }

  $("#knowledgeGraph")?.addEventListener("nodeselect", (event) => {
    openHypothesisByGraphNode(event.detail?.id);
  });

  $$("[data-dock-tab]").forEach((button) => {
    button.addEventListener("click", () => setDockTab(button.dataset.dockTab));
  });
  $$("[data-settings-tab]").forEach((button) => {
    button.addEventListener("click", () => setSettingsTab(button.dataset.settingsTab));
  });
  $("#dockExpandBtn")?.addEventListener("click", () => {
    state.dockExpanded = !state.dockExpanded;
    renderDockState();
  });

  $("#promptFilesList").addEventListener("click", (event) => {
    const button = event.target.closest("[data-remove-prompt-file]");
    if (!button) return;
    state.promptFiles.splice(Number(button.dataset.removePromptFile), 1);
    renderPromptFiles();
  });

  window.addEventListener("resize", () => {
    clearTimeout(window.__graphResizeTimer);
    window.__graphResizeTimer = setTimeout(drawGraph, 120);
  });

  $$(".range-row input").forEach((input) => {
    input.addEventListener("input", () => {
      state.weights[input.dataset.weight] = Number(input.value);
      renderWeights();
    });
  });

  $("#hypothesesList").addEventListener("click", async (event) => {
    const graphButton = event.target.closest("[data-focus-node], [data-path-node]");
    if (graphButton) {
      event.preventDefault();
      event.stopPropagation();
      const nodeId = graphButton.dataset.focusNode || graphButton.dataset.pathNode;
      if (graphButton.dataset.pathNode) pathGraphNode(nodeId);
      else focusGraphNode(nodeId);
      return;
    }

    const reactionButton = event.target.closest("[data-reaction]");
    if (reactionButton) {
      await sendReaction(reactionButton.dataset.id, reactionButton.dataset.reaction);
      return;
    }

    const button = event.target.closest("[data-status]");
    if (button) {
      await updateStatus(button.dataset.id, button.dataset.status);
      return;
    }

    const card = event.target.closest(".hypothesis-card[data-hypothesis-id]");
    if (!card || event.target.closest("button, input, textarea, select, form, a")) return;
    state.openHypothesisId = state.openHypothesisId === card.dataset.hypothesisId ? null : card.dataset.hypothesisId;
    renderHypotheses();
  });

  $("#hypothesesList").addEventListener("submit", async (event) => {
    if (!event.target.matches(".feedback-form")) return;
    event.preventDefault();
    await sendFeedback(event.target);
  });
}

async function refreshProjects() {
  const payload = await api("/api/projects");
  state.projects = payload.projects || [];
  if (!state.projects.length) {
    await createProject({
      preventDefault() {},
      silent: true,
      name: "Новый исследовательский проект",
    });
    return;
  }
  const saved = localStorage.getItem("hl.projectId");
  state.projectId = state.projects.some((item) => item.id === saved) ? saved : state.projects[0].id;
  renderProjectSelect();
  await loadState();
}

async function createProject(event) {
  event?.preventDefault?.();
  const silent = Boolean(event?.silent);
  const projectName = String(event?.name || (silent ? "Новый исследовательский проект" : window.prompt("Название нового проекта", "")) || "").trim();
  if (!projectName) return;
  setBusy({ label: "Создание проекта", progress: 18 });
  try {
    const payload = await api("/api/projects", {
      method: "POST",
      json: {
        name: projectName.slice(0, 160),
        domain: "Обогащение цветных и благородных металлов",
        goal: "Повысить извлечение ценных компонентов из хвостов при сохранении качества концентрата.",
        constraints: "Использовать доступное лабораторное флотационное оборудование, минимизировать рост расхода реагентов и CAPEX.",
        team: [actor()],
      },
    });
    state.projectId = payload.project.id;
    localStorage.setItem("hl.projectId", state.projectId);
    await refreshProjects();
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function deleteProject() {
  if (!state.projectId) return;
  const project = state.state?.project || state.projects.find((item) => item.id === state.projectId);
  const name = project?.name || "текущий проект";
  if (!window.confirm(`Удалить проект «${name}»? Это действие нельзя отменить.`)) return;
  setBusy({ label: "Удаление проекта", progress: 25 });
  try {
    await api(`/api/projects/${state.projectId}`, { method: "DELETE" });
    localStorage.removeItem("hl.projectId");
    state.projectId = null;
    state.state = null;
    state.openHypothesisId = null;
    await refreshProjects();
    toast("Проект удален");
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function saveProject(event) {
  event.preventDefault();
  if (!state.projectId) return;
  setBusy({ label: "Обновление проекта", progress: 35 });
  try {
    await api(`/api/projects/${state.projectId}`, {
      method: "PATCH",
      json: {
        name: $("#projectName").value,
        domain: $("#projectDomain").value,
        goal: $("#projectGoal").value,
        constraints: $("#projectConstraints").value,
        settings: state.weights,
      },
    });
    await refreshProjects();
    toast("Проект обновлен");
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function loadState(options = {}) {
  if (!state.projectId) return;
  const manageBusy = options.busy !== false;
  if (manageBusy) setBusy({ label: "Обновление данных", progress: 45 });
  try {
    state.state = await api(`/api/projects/${state.projectId}/state`);
    renderAll();
  } catch (error) {
    toast(error.message);
  } finally {
    if (manageBusy) setBusy(false);
  }
}

async function uploadFiles(event) {
  const files = Array.from(event.target.files || []);
  if (!files.length || !state.projectId) return;
  setBusy({ label: `Подготовка файлов 0/${files.length}`, progress: 0 });
  try {
    for (const [index, file] of files.entries()) {
      const shortName = trim(file.name, 24);
      const form = new FormData();
      form.append("file", file);
      const ocrLanguages = $("#sourceLanguageSelect").value;
      if (ocrLanguages) form.append("ocr_languages", ocrLanguages);
      await apiUpload(
        `/api/projects/${state.projectId}/documents`,
        { method: "POST", body: form },
        {
          onProgress: (percent) => {
            const overall = ((index + (percent / 100) * 0.62) / files.length) * 100;
            setBusy({ label: `Загрузка ${index + 1}/${files.length}: ${shortName}`, progress: overall });
          },
          onUploaded: () => {
            const overall = ((index + 0.74) / files.length) * 100;
            setBusy({ label: `Парсинг и граф ${index + 1}/${files.length}: ${shortName}`, progress: overall });
            startBusyStages(fileProcessingBusyStages(index, files.length, shortName));
          },
        },
      );
      stopBusyStages();
      setBusy({ label: `Файл обработан ${index + 1}/${files.length}`, progress: ((index + 1) / files.length) * 100 });
    }
    setBusy({ label: "Обновление графа", progress: 96 });
    await loadState({ busy: false });
    toast(`Загружено файлов: ${files.length}`);
  } catch (error) {
    toast(error.message);
  } finally {
    event.target.value = "";
    setBusy(false);
  }
}

function fileProcessingBusyStages(index, totalFiles, filename) {
  const start = ((index + 0.74) / totalFiles) * 100;
  const ocr = ((index + 0.84) / totalFiles) * 100;
  const graph = ((index + 0.92) / totalFiles) * 100;
  const final = ((index + 0.96) / totalFiles) * 100;
  return [
    { label: `Парсинг PDF ${index + 1}/${totalFiles}: ${filename}`, progress: start, duration: 900 },
    { label: `OCR / vision ${index + 1}/${totalFiles}: ${filename}`, progress: ocr, duration: 9000 },
    { label: `Построение графа ${index + 1}/${totalFiles}: ${filename}`, progress: graph, duration: 9000 },
    { label: `Сохранение источника ${index + 1}/${totalFiles}: ${filename}`, progress: final, target: final, duration: 5000 },
  ];
}

async function importSamples() {
  if (!state.projectId) return;
  setBusy({ label: "Импорт источников", progress: 15 });
  try {
    const payload = await api(`/api/projects/${state.projectId}/documents/import-samples`, {
      method: "POST",
      json: {
        max_files: 12,
        extensions: [".png", ".jpg", ".jpeg", ".docx", ".xlsx", ".pdf"],
        ocr_languages: selectedSourceLanguages(),
      },
    });
    setBusy({ label: "Обновление графа", progress: 94 });
    await loadState({ busy: false });
    toast(`Импортировано: ${payload.imported.length}`);
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

function selectPromptFiles(event) {
  const incoming = Array.from(event.target.files || []);
  const seen = new Set(state.promptFiles.map((file) => `${file.name}:${file.size}:${file.lastModified}`));
  incoming.forEach((file) => {
    const key = `${file.name}:${file.size}:${file.lastModified}`;
    if (!seen.has(key)) state.promptFiles.push(file);
  });
  event.target.value = "";
  renderPromptFiles();
}

function selectedSourceLanguages() {
  const value = $("#sourceLanguageSelect")?.value || "";
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

async function generateHypotheses() {
  if (!state.projectId) return;
  try {
    const exclusions = $("#exclusionsInput").value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    const requestPayload = {
      count: Number($("#countInput").value || 5),
      weights: state.weights,
      exclusions,
      include_roadmap: true,
      research_enabled: Boolean($("#researchEnabledInput")?.checked),
      research_query: $("#researchQueryInput")?.value.trim() || "",
      research_sources: Number($("#researchSourcesInput")?.value || 6),
    };
    const stages = generationBusyStages({
      research: requestPayload.research_enabled,
      attachments: state.promptFiles.length,
      start: state.promptFiles.length ? 32 : 8,
    });
    if (!state.promptFiles.length) startBusyStages(stages);
    const payload = state.promptFiles.length
      ? await generateWithPromptFiles(requestPayload, stages)
      : await api(`/api/projects/${state.projectId}/generate`, {
          method: "POST",
          json: requestPayload,
        });
    setBusy({ label: "Обновление интерфейса", progress: 97 });
    state.state = payload.state;
    state.promptFiles = [];
    renderPromptFiles();
    renderAll();
    toast(generationToast(payload.meta));
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function clearHypotheses() {
  if (!state.projectId) return;
  if (!confirm("Удалить все сгенерированные гипотезы и вернуть граф к исходному состоянию?")) return;
  setBusy({ label: "Очистка гипотез", progress: 40 });
  try {
    const payload = await api(`/api/projects/${state.projectId}/hypotheses`, { method: "DELETE" });
    state.state = payload.state;
    state.openHypothesisId = null;
    renderAll();
    toast("Гипотезы очищены");
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function generateWithPromptFiles(requestPayload, stages = []) {
  const form = new FormData();
  form.append("payload_json", JSON.stringify(requestPayload));
  const ocrLanguages = $("#sourceLanguageSelect").value;
  if (ocrLanguages) form.append("ocr_languages", ocrLanguages);
  state.promptFiles.forEach((file) => form.append("files", file));
  return apiUpload(
    `/api/projects/${state.projectId}/generate-with-files`,
    { method: "POST", body: form },
    {
      onProgress: (percent) => setBusy({ label: `Загрузка вложений к промпту`, progress: Math.min(30, percent * 0.3) }),
      onUploaded: () => startBusyStages(stages),
    },
  );
}

function generationBusyStages({ research, attachments, start = 8 }) {
  const steps = [
    { label: "Подготовка контекста", progress: start, duration: 900 },
  ];
  if (attachments) {
    steps.push({ label: "Разбор вложений к промпту", progress: Math.max(34, start + 4), duration: 3600 });
  }
  if (research) {
    steps.push({ label: "Web research и источники", progress: attachments ? 48 : 26, duration: 8000 });
  }
  steps.push(
    { label: "Генерация гипотез", progress: research ? 66 : attachments ? 58 : 38, duration: 10000 },
    { label: "Сохранение и обновление графа", progress: 86, duration: 6000 },
  );
  return steps;
}

function generationToast(meta = {}) {
  const parts = ["Гипотезы сгенерированы"];
  if (meta.prompt_attachments) parts.push(`вложений: ${meta.prompt_attachments}`);
  if (meta.research?.enabled) {
    const sourceCount = Array.isArray(meta.research.sources) ? meta.research.sources.length : Number(meta.research.sources || 0);
    parts.push(`research: ${sourceCount} ссылок`);
  }
  return parts.join(" · ");
}

async function updateStatus(id, status) {
  setBusy({ label: "Обновление статуса", progress: 45 });
  try {
    const payload = await api(`/api/projects/${state.projectId}/hypotheses/${id}/status`, {
      method: "PATCH",
      json: { status },
    });
    state.state = payload.state;
    renderAll();
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function sendReaction(id, reaction) {
  if (!id || !["liked", "disliked"].includes(reaction)) return;
  const current = reactionSummary(id).mine;
  const nextReaction = current === reaction ? "neutral" : reaction;
  setBusy({ label: "Сохранение реакции", progress: 45 });
  try {
    const payload = await api(`/api/projects/${state.projectId}/hypotheses/${id}/feedback`, {
      method: "POST",
      json: {
        rating: nextReaction === "liked" ? 5 : nextReaction === "disliked" ? 1 : 3,
        outcome: nextReaction,
        comment: `quick_reaction:${nextReaction}`,
      },
    });
    state.state = payload.state;
    renderAll();
    toast(nextReaction === "neutral" ? "Реакция снята" : nextReaction === "liked" ? "Лайк учтен для следующих генераций" : "Дизлайк учтен для самоулучшения");
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function sendFeedback(form) {
  const id = form.dataset.id;
  setBusy({ label: "Сохранение фидбэка", progress: 45 });
  try {
    const payload = await api(`/api/projects/${state.projectId}/hypotheses/${id}/feedback`, {
      method: "POST",
      json: {
        rating: Number($("[name='rating']", form).value || 0) || null,
        outcome: $("[name='outcome']", form).value || null,
        comment: $("[name='comment']", form).value,
      },
    });
    state.state = payload.state;
    renderAll();
    toast("Фидбэк сохранен");
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function sendChat(event) {
  event.preventDefault();
  const input = $("#chatInput");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  startBusyStages([
    { label: "Отправка сообщения", progress: 18, duration: 700 },
    { label: "Анализ контекста", progress: 38, duration: 2400 },
    { label: "Ответ ассистента", progress: 68, duration: 5200 },
    { label: "Обновление диалога", progress: 90, duration: 2200 },
  ]);
  try {
    const payload = await api(`/api/projects/${state.projectId}/chat`, {
      method: "POST",
      json: { message },
    });
    state.state = payload.state;
    renderAll();
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

function renderAll() {
  renderProjectSelect();
  renderProjectForm();
  renderWeights();
  renderPromptFiles();
  renderRuntime();
  renderDocuments();
  renderHypotheses();
  renderGenerateButton();
  renderEvents();
  renderChat();
  renderDockState();
  renderSettingsTabs();
  drawGraph();
}

function renderGenerateButton() {
  const button = $("#generateBtn");
  if (!button) return;
  const hasHypotheses = Boolean(state.state?.hypotheses?.length);
  if (hasHypotheses) {
    button.dataset.mode = "clear";
    button.title = "Удалить гипотезы и вернуть граф к исходному состоянию";
    button.innerHTML = "<span>🗑</span><b>Очистить гипотезы</b>";
  } else {
    button.dataset.mode = "generate";
    button.title = "Сгенерировать гипотезы";
    button.innerHTML = "<span>✦</span><b>Генерация</b>";
  }
}

function renderDockState() {
  $$("[data-dock-tab]").forEach((button) => {
    button.classList.toggle("active", button.dataset.dockTab === state.dockTab);
  });
  $$("[data-dock-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.dockPanel !== state.dockTab;
  });
  const expanded = state.dockTab === "chat" && state.dockExpanded;
  $(".dock")?.classList.toggle("expanded", expanded);
  $(".graph-stage")?.classList.toggle("dock-expanded", expanded);
  const expandBtn = $("#dockExpandBtn");
  if (expandBtn) {
    expandBtn.hidden = state.dockTab !== "chat";
    expandBtn.textContent = expanded ? "Свернуть" : "Развернуть";
    expandBtn.classList.toggle("active", expanded);
  }
}

function setDockTab(tab) {
  state.dockTab = tab || "docs";
  if (state.dockTab === "chat") state.dockExpanded = true;
  renderDockState();
}

function renderSettingsTabs() {
  const knownTabs = new Set($$("[data-settings-tab]").map((button) => button.dataset.settingsTab));
  if (!knownTabs.has(state.settingsTab)) state.settingsTab = "project";
  $$("[data-settings-tab]").forEach((button) => {
    const active = button.dataset.settingsTab === state.settingsTab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  $$("[data-settings-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.settingsPanel !== state.settingsTab;
  });
}

function setSettingsTab(tab) {
  state.settingsTab = tab || "project";
  renderSettingsTabs();
}

function renderActorBadge() {
  const actorName = actor();
  const badge = $("#actorBadge");
  if (badge) badge.textContent = actorName.slice(0, 1).toUpperCase() || "R";
}

function focusGraphNode(nodeId) {
  const graph = $("#knowledgeGraph");
  if (graph && typeof graph.focusNode === "function") graph.focusNode(nodeId);
}

function pathGraphNode(nodeId) {
  const graph = $("#knowledgeGraph");
  if (!graph) return;
  if (typeof graph.highlightPath === "function") graph.highlightPath(nodeId);
  else if (typeof graph.focusNode === "function") graph.focusNode(nodeId);
}

function openHypothesisByGraphNode(nodeId) {
  if (!String(nodeId || "").startsWith("hypothesis:")) return;
  const hypothesisId = String(nodeId).slice("hypothesis:".length);
  if (!state.state?.hypotheses?.some((item) => item.id === hypothesisId)) return;
  state.openHypothesisId = hypothesisId;
  renderHypotheses();
  requestAnimationFrame(() => {
    $(`.hypothesis-card[data-hypothesis-id="${cssEscape(hypothesisId)}"]`)?.scrollIntoView({ block: "nearest" });
  });
}

function renderPromptFiles() {
  const list = $("#promptFilesList");
  if (!list) return;
  list.innerHTML = state.promptFiles.length
    ? state.promptFiles
        .map(
          (file, index) => `<div class="prompt-file-chip">
            <span title="${escapeHtml(file.name)}">${escapeHtml(trim(file.name, 28))}</span>
            <b>${formatBytes(file.size)}</b>
            <button type="button" data-remove-prompt-file="${index}" title="Убрать файл">×</button>
          </div>`,
        )
        .join("")
    : `<div class="prompt-files-empty">Файлы к промпту не выбраны</div>`;
}

function renderProjectSelect() {
  $("#projectSelect").innerHTML = state.projects
    .map((project) => `<option value="${project.id}" ${project.id === state.projectId ? "selected" : ""}>${escapeHtml(project.name)}</option>`)
    .join("");
}

function renderProjectForm() {
  const project = state.state?.project;
  if (!project) return;
  $("#projectName").value = project.name || "";
  $("#projectDomain").value = project.domain || "";
  $("#projectGoal").value = project.goal || "";
  $("#projectConstraints").value = project.constraints || "";
  state.weights = { ...state.weights, ...(project.settings || {}) };
  $("#projectMeta").textContent = `${state.state.documents.length} источников · ${state.state.hypotheses.length} гипотез`;
}

function renderWeights() {
  const map = {
    novelty: "#weightNovelty",
    feasibility: "#weightFeasibility",
    impact: "#weightImpact",
    risk: "#weightRisk",
  };
  Object.entries(map).forEach(([key, selector]) => {
    const value = Number(state.weights[key] ?? 0).toFixed(2);
    const input = $(`input[data-weight="${key}"]`);
    if (input) input.value = value;
    $(selector).textContent = value;
  });
}

function renderRuntime() {
  const runtime = state.state?.runtime;
  const badge = $("#runtimeBadge");
  if (!runtime) return;
  badge.classList.toggle("ok", Boolean(runtime.openrouter_enabled));
  badge.textContent = runtime.openrouter_enabled ? "OpenRouter подключен" : "Нужен ключ OpenRouter";
  badge.title = "";
}

function renderDocuments() {
  const docs = state.state?.documents || [];
  $("#documentCount").textContent = String(docs.length);
  $("#documentsList").innerHTML = docs.length
    ? docs
        .map(
          (doc) => `<div class="doc-row">
            <span class="doc-type">${escapeHtml(fileType(doc.filename, doc.content_type))}</span>
            <div class="doc-main">
              <b title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</b>
              <span>${Number(doc.metadata?.chars || 0).toLocaleString("ru-RU")} знаков</span>
            </div>
          </div>`,
        )
        .join("")
    : `<div class="empty">Источников нет</div>`;
}

function renderHypotheses() {
  const hypotheses = state.state?.hypotheses || [];
  const list = $("#hypothesesList");
  if (!hypotheses.length) {
    state.openHypothesisId = null;
    list.innerHTML = `<div class="empty">Гипотез пока нет</div>`;
    return;
  }
  if (!hypotheses.some((item) => item.id === state.openHypothesisId)) {
    state.openHypothesisId = hypotheses[0].id;
  }
  list.innerHTML = hypotheses.map(renderHypothesis).join("");
}

function renderHypothesis(item) {
  const isOpen = item.id === state.openHypothesisId;
  const graphNode = `hypothesis:${item.id}`;
  const reactions = reactionSummary(item.id);
  const metrics = [
    ["Новизна", item.novelty],
    ["Реализ.", item.feasibility],
    ["Эффект", item.impact],
    ["Риск", item.risk],
  ];
  const evidence = renderEvidenceReport(item.evidence || []);
  const roadmap = renderRoadmapReport(item.roadmap || []);
  const economics = renderEconomicsReport(item.economics || []);
  const statuses = [
    ["draft", "Черновик"],
    ["review", "Проверка"],
    ["experiment", "Опыт"],
    ["confirmed", "Да"],
    ["rejected", "Нет"],
  ];
  const statusLabel = statuses.find(([status]) => status === item.status)?.[1] || "Черновик";
  return `<article class="hypothesis-card ${escapeHtml(item.status)} ${isOpen ? "open" : ""}" data-hypothesis-id="${escapeHtml(item.id)}">
    <div class="hypothesis-title">
      <div>
        <span class="status-label">${escapeHtml(statusLabel)}</span>
        <h3>${escapeHtml(item.title)}</h3>
      </div>
      <div class="score">${displayScore(item)}<span>score</span></div>
    </div>
    <div class="statement">${escapeHtml(item.statement)}</div>
    <div class="reaction-row" aria-label="Быстрая обратная связь">
      <button type="button" class="reaction-button liked ${reactions.mine === "liked" ? "active" : ""}" data-id="${escapeHtml(item.id)}" data-reaction="liked" title="Нравится: усилить похожие гипотезы в следующих генерациях"><span>👍</span><b>${reactions.likes}</b></button>
      <button type="button" class="reaction-button disliked ${reactions.mine === "disliked" ? "active" : ""}" data-id="${escapeHtml(item.id)}" data-reaction="disliked" title="Не нравится: избегать похожих гипотез в следующих генерациях"><span>👎</span><b>${reactions.dislikes}</b></button>
    </div>
    <div class="metrics">${metrics
      .map(([label, value]) => `<div class="metric"><span>${label}</span><b>${displayMetric(value)}</b></div>`)
      .join("")}</div>
    ${
      isOpen
        ? `<div class="hypothesis-extra">
            <section class="report-section">
              <h4>Обоснование</h4>
              <p>${escapeHtml(item.rationale || "Обоснование не сформировано для этой версии гипотезы.")}</p>
            </section>
            <section class="report-section">
              <h4>Механизм</h4>
              <p>${escapeHtml(item.mechanism || "Механизм не описан.")}</p>
            </section>
            <section class="report-section report-section-muted">
              <h4>Неопределенности</h4>
              <p>${escapeHtml(item.uncertainty || "Ключевые неопределенности не указаны.")}</p>
            </section>
            <section class="report-section">
              <h4>План внедрения / проверки</h4>
              ${roadmap}
            </section>
            <section class="report-section">
              <h4>Экономический контур</h4>
              ${economics}
            </section>
            <section class="report-section">
              <h4>Источники и контекст ссылок</h4>
              ${evidence}
            </section>
            <div class="graph-actions">
              <button type="button" data-path-node="${escapeHtml(graphNode)}">Путь к KPI</button>
              <button type="button" data-focus-node="${escapeHtml(graphNode)}">Фокус в графе</button>
            </div>
            <div class="status-row">${statuses
              .map(
                ([status, label]) =>
                  `<button type="button" class="${item.status === status ? "active" : ""}" data-id="${item.id}" data-status="${status}">${label}</button>`,
              )
              .join("")}</div>
            <form class="feedback-form" data-id="${item.id}">
              <select name="rating" aria-label="Оценка">
                <option value="">Оценка</option>
                <option value="5">5</option><option value="4">4</option><option value="3">3</option><option value="2">2</option><option value="1">1</option>
              </select>
              <select name="outcome" aria-label="Исход">
                <option value="">Исход</option>
                <option value="confirmed">Подтв.</option>
                <option value="rejected">Опр.</option>
                <option value="experiment">В опыт</option>
              </select>
              <textarea name="comment" rows="1" placeholder="Комментарий эксперта"></textarea>
              <button type="submit">✓</button>
            </form>
          </div>`
        : ""
    }
  </article>`;
}

function reactionSummary(hypothesisId) {
  const latest = new Map();
  (state.state?.feedback || []).forEach((item) => {
    if (String(item.hypothesis_id || "") !== String(hypothesisId)) return;
    const outcome = normalizeReaction(item.outcome);
    if (!outcome) return;
    const key = item.actor || "expert";
    if (!latest.has(key)) latest.set(key, outcome);
  });
  let likes = 0;
  let dislikes = 0;
  latest.forEach((outcome) => {
    if (outcome === "liked") likes += 1;
    if (outcome === "disliked") dislikes += 1;
  });
  return { likes, dislikes, mine: latest.get(actor()) || "" };
}

function normalizeReaction(value) {
  const outcome = String(value || "").trim().toLowerCase();
  if (["liked", "like"].includes(outcome)) return "liked";
  if (["disliked", "dislike"].includes(outcome)) return "disliked";
  if (["neutral", "reaction_removed"].includes(outcome)) return "neutral";
  return "";
}

function renderEvidenceReport(items) {
  if (!items.length) return `<div class="report-empty">Источники не указаны для этой версии гипотезы.</div>`;
  return `<div class="evidence report-list">${items
    .slice(0, 6)
    .map((ev) => {
      const url = safeExternalUrl(ev?.url);
      const source = escapeHtml(ev?.source || "source");
      const sourceHtml = url ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${source}</a>` : `<b>${source}</b>`;
      const quote = ev?.quote ? `<p>${escapeHtml(ev.quote)}</p>` : "";
      const why = ev?.why ? `<small>${escapeHtml(ev.why)}</small>` : "";
      return `<article>${sourceHtml}${quote}${why}</article>`;
    })
    .join("")}</div>`;
}

function renderRoadmapReport(items) {
  if (!items.length) return `<div class="report-empty">План не сформирован. Для старых гипотез можно перегенерировать с текущим промптом.</div>`;
  return `<div class="roadmap report-list">${items
    .slice(0, 6)
    .map((step, index) => {
      const owner = step.owner ? `<em>${escapeHtml(step.owner)}</em>` : "";
      return `<article>
        <b>${escapeHtml(step.step || index + 1)}. ${escapeHtml(step.title || "Шаг")}</b>
        ${owner}
        <p>${escapeHtml(step.output || "")}</p>
      </article>`;
    })
    .join("")}</div>`;
}

function renderEconomicsReport(items) {
  if (!items.length) {
    return `<div class="report-empty">Экономический расчет не сформирован. Новая генерация будет запрашивать формулу, допущения и данные для уточнения.</div>`;
  }
  return `<div class="economics report-list">${items
    .slice(0, 5)
    .map((economic) => {
      const details = [
        ["Допущение", economic?.assumption],
        ["Расчет", economic?.calculation],
        ["Эффект", economic?.expected_effect],
        ["Данные", economic?.data_needed],
        ["Доверие", economic?.confidence],
      ]
        .filter(([, value]) => String(value || "").trim())
        .map(([label, value]) => `<span><b>${label}</b>${escapeHtml(value)}</span>`)
        .join("");
      return `<article><strong>${escapeHtml(economic?.item || "Оценка")}</strong>${details}</article>`;
    })
    .join("")}</div>`;
}

function renderEvents() {
  const events = state.state?.events || [];
  $("#versionCount").textContent = String(events[0]?.version_no || events.length || 0);
  $("#eventsList").innerHTML = events.length
    ? events
        .map(
          (event) => `<div class="event-row">
            <span class="event-version">v${event.version_no}</span>
            <div class="event-main">
              <b>${escapeHtml(event.action)}</b>
              <span>${escapeHtml(event.actor)} · ${formatDate(event.created_at)}</span>
            </div>
          </div>`,
        )
        .join("")
    : `<div class="empty">Событий нет</div>`;
}

function renderChat() {
  const chat = state.state?.chat || [];
  const log = $("#chatLog");
  log.innerHTML = chat.length
    ? chat
        .map(
          (message) => `<div class="message ${escapeHtml(message.role)}">
            <strong>${escapeHtml(message.actor)} · ${formatDate(message.created_at)}</strong>
            <div class="message-body">${
              message.role === "assistant"
                ? renderMarkdown(message.content)
                : escapeHtml(message.content).replaceAll("\n", "<br>")
            }</div>
          </div>`,
        )
        .join("")
    : `<div class="empty">Чат пуст</div>`;
  log.scrollTop = log.scrollHeight;
}

function prepareGraphData(rawGraph, hypotheses) {
  const nodeById = new Map();
  (rawGraph.nodes || []).forEach((node) => {
    if (!node?.id) return;
    nodeById.set(String(node.id), {
      ...node,
      id: String(node.id),
      label: String(node.label || node.id),
      type: String(node.type || "concept"),
      summary: String(node.summary || ""),
      weight: Number(node.weight || 1),
    });
  });

  hypotheses.forEach((hypothesis) => {
    const id = `hypothesis:${hypothesis.id}`;
    if (!nodeById.has(id)) {
      nodeById.set(id, {
        id,
        label: hypothesis.title || "Гипотеза",
        type: "hypothesis",
        summary: hypothesis.statement || "",
        weight: 3,
      });
    }
  });

  const nodes = [...nodeById.values()].slice(0, 180);
  const nodeIds = new Set(nodes.map((node) => node.id));
  const edges = (rawGraph.edges || [])
    .filter((edge) => nodeIds.has(String(edge.source)) && nodeIds.has(String(edge.target)))
    .map((edge, index) => ({
      ...edge,
      id: String(edge.id || `edge:${index}`),
      source: String(edge.source),
      target: String(edge.target),
      relation: String(edge.relation || "related_to"),
      evidence: String(edge.evidence || ""),
      weight: Number(edge.weight || 1),
    }))
    .slice(0, 360);

  return { nodes, edges, kpi: chooseKpiNode(nodes) };
}

function chooseKpiNode(nodes) {
  const candidates = nodes.filter((node) => ["property", "metric"].includes(node.type));
  const preferred = candidates.find((node) => /извлеч|recovery|kpi|эффект|impact|yield/i.test(node.label));
  return (preferred || candidates.sort((a, b) => Number(b.weight || 0) - Number(a.weight || 0))[0])?.id || null;
}

function graphDataKey(graph) {
  return [
    graph.kpi || "",
    ...graph.nodes.map((node) => `${node.id}:${node.type}:${node.label}:${Number(node.weight || 0).toFixed(2)}`).sort(),
    ...graph.edges.map((edge) => `${edge.id}:${edge.source}:${edge.target}:${edge.relation}:${Number(edge.weight || 0).toFixed(2)}`).sort(),
  ].join("|");
}

function displayMetric(value) {
  return Number(normalizeMetricForDisplay(value)).toFixed(0);
}

function displayScore(item) {
  const values = [item.novelty, item.feasibility, item.impact, item.risk].map((value) => Number(value || 0));
  const looksUnitScaled = values.some((value) => value > 0 && value <= 1) && values.every((value) => value <= 1);
  if (!looksUnitScaled) return Number(item.score || 0).toFixed(1);
  const novelty = normalizeMetricForDisplay(item.novelty);
  const feasibility = normalizeMetricForDisplay(item.feasibility);
  const impact = normalizeMetricForDisplay(item.impact);
  const risk = normalizeMetricForDisplay(item.risk);
  return (
    novelty * Number(state.weights.novelty || 0) +
    feasibility * Number(state.weights.feasibility || 0) +
    impact * Number(state.weights.impact || 0) +
    (100 - risk) * Number(state.weights.risk || 0)
  ).toFixed(1);
}

function normalizeMetricForDisplay(value) {
  const number = Number(value || 0);
  return number > 0 && number <= 1 ? number * 100 : number;
}

function drawGraph() {
  const graph = prepareGraphData(state.state?.graph || { nodes: [], edges: [] }, state.state?.hypotheses || []);
  const statsText = `${graph.nodes.length} узлов · ${graph.edges.length} связей`;
  $("#graphStats").textContent = statsText;
  $("#dockGraphStats").textContent = statsText;

  const engine = $("#knowledgeGraph");
  if (engine) {
    const key = graphDataKey(graph);
    if (state.graphKey !== key && typeof engine.setData === "function") {
      state.graphKey = key;
      engine.setData(graph);
    }
    return;
  }

  const svg = $("#graphSvg");
  if (!svg) return;
  svg.innerHTML = "";
  const width = Math.max(360, svg.clientWidth || 720);
  const height = Math.max(260, svg.clientHeight || 360);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  graphView.width = width;
  graphView.height = height;

  const previousNodes = graphView.nodeById;
  const nodes = graph.nodes.slice(0, 90).map((node, index) => {
    const previous = previousNodes.get(node.id);
    return {
      ...node,
      index,
      x: previous?.x,
      y: previous?.y,
      pinned: previous?.pinned || false,
      degree: 0,
    };
  });
  const nodeIds = new Set(nodes.map((node) => node.id));
  const edges = graph.edges
    .filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
    .slice(0, 180)
    .map((edge, index) => ({ ...edge, index }));
  $("#graphStats").textContent = statsText;

  if (!nodes.length) {
    graphView.key = "";
    graphView.nodes = [];
    graphView.edges = [];
    graphView.nodeById = new Map();
    graphView.edgeById = new Map();
    graphView.selected = null;
    $("#graphDetails").hidden = true;
    svg.appendChild(svgText(width / 2, height / 2, "Граф пуст", "middle", "empty-text"));
    return;
  }

  edges.forEach((edge) => {
    const source = nodes.find((node) => node.id === edge.source);
    const target = nodes.find((node) => node.id === edge.target);
    if (source) source.degree += 1;
    if (target) target.degree += 1;
  });

  const key = graphKey(nodes, edges);
  const needsLayout = key !== graphView.key || nodes.some((node) => !Number.isFinite(node.x) || !Number.isFinite(node.y));
  if (needsLayout) {
    layoutGraph(nodes, edges, width, height);
    graphView.transform = { x: 0, y: 0, k: 1 };
  }

  graphView.key = key;
  graphView.nodes = nodes;
  graphView.edges = edges;
  graphView.nodeById = new Map(nodes.map((node) => [node.id, node]));
  graphView.edgeById = new Map(edges.map((edge) => [edge.id, edge]));
  if (graphView.selected?.type === "node" && !graphView.nodeById.has(graphView.selected.id)) graphView.selected = null;
  if (graphView.selected?.type === "edge" && !graphView.edgeById.has(graphView.selected.id)) graphView.selected = null;

  if (needsLayout) fitGraphToViewport(false);
  renderGraphScene();
  renderGraphDetails();
}

function renderGraphScene() {
  const svg = $("#graphSvg");
  svg.innerHTML = "";
  svg.appendChild(graphDefs());

  const viewport = svgEl("g", {
    transform: `translate(${graphView.transform.x},${graphView.transform.y}) scale(${graphView.transform.k})`,
  });
  const edgeLayer = svgEl("g", { class: "edge-layer" });
  const hitLayer = svgEl("g", { class: "edge-hit-layer" });
  const nodeLayer = svgEl("g", { class: "node-layer" });
  const labelLayer = svgEl("g", { class: "label-layer" });
  viewport.append(edgeLayer, hitLayer, nodeLayer, labelLayer);
  svg.appendChild(viewport);

  graphView.edges.forEach((edge) => {
    const source = graphView.nodeById.get(edge.source);
    const target = graphView.nodeById.get(edge.target);
    if (!source || !target) return;
    const pathValue = edgePath(source, target, edge.index);
    const selected = graphView.selected?.type === "edge" && graphView.selected.id === edge.id;
    const path = svgEl("path", {
      d: pathValue,
      class: `graph-edge ${selected ? "selected" : ""}`,
      fill: "none",
      stroke: selected ? "#08786f" : "#9db0a8",
      "stroke-width": Math.max(1.1, Math.min(4.2, Number(edge.weight || 1))),
      opacity: selected ? 0.95 : 0.52,
      "marker-end": "url(#arrow)",
      "data-edge-id": edge.id,
    });
    const title = svgEl("title");
    title.textContent = `${edge.relation}: ${edge.evidence || ""}`;
    path.appendChild(title);
    edgeLayer.appendChild(path);
    hitLayer.appendChild(svgEl("path", { d: pathValue, class: "edge-hit", "data-edge-id": edge.id }));
  });

  graphView.nodes.forEach((node) => {
    const radius = nodeRadius(node);
    node.r = radius;
    const selected = graphView.selected?.type === "node" && graphView.selected.id === node.id;
    const group = svgEl("g", {
      class: `graph-node ${selected ? "selected" : ""}`,
      transform: `translate(${node.x},${node.y})`,
      "data-node-id": node.id,
    });
    const halo = svgEl("circle", {
      r: radius + 8,
      fill: colorForType(node.type),
      opacity: selected ? 0.18 : 0.08,
    });
    const circle = svgEl("circle", {
      r: radius,
      fill: colorForType(node.type),
      stroke: "#fff",
      "stroke-width": 2,
    });
    const title = svgEl("title");
    title.textContent = `${node.label} · ${node.type}\n${node.summary || ""}`;
    circle.appendChild(title);
    group.append(halo, circle);
    nodeLayer.appendChild(group);
  });

  drawGraphLabels(labelLayer, graphView.nodes, graphView.width);
}

function drawGraphLabels(layer, nodes, width) {
  const placed = [];
  const sorted = [...nodes].sort((a, b) => {
    const priority = (node) => (node.type === "hypothesis" ? 3 : node.type === "source" ? 2 : 1);
    return priority(b) - priority(a) || Number(b.degree || 0) - Number(a.degree || 0) || Number(b.weight || 0) - Number(a.weight || 0);
  });

  sorted.forEach((node) => {
    const text = trim(node.label, node.type === "hypothesis" || node.type === "source" ? 30 : 22);
    const textWidth = Math.min(186, Math.max(42, text.length * 7.1));
    const textHeight = 18;
    let x = node.x + (node.r || 8) + 7;
    if (x + textWidth + 10 > width) {
      x = node.x - (node.r || 8) - textWidth - 12;
    }
    const y = node.y - textHeight / 2;
    const box = { x, y, w: textWidth + 8, h: textHeight };
    if (placed.some((other) => intersects(box, other))) return;
    placed.push(box);

    const selected = graphView.selected?.type === "node" && graphView.selected.id === node.id;
    const group = svgEl("g", {
      class: `graph-label ${selected ? "selected" : ""}`,
      "data-node-id": node.id,
    });
    group.appendChild(
      svgEl("rect", {
        x: box.x,
        y: box.y,
        width: box.w,
        height: box.h,
        rx: 4,
        class: "label-bg",
        "data-node-id": node.id,
      }),
    );
    group.appendChild(svgText(box.x + 4, box.y + 13, text, "start", "node-label"));
    layer.appendChild(group);
  });
}

function layoutGraph(nodes, edges, width, height) {
  const centerX = width / 2;
  const centerY = height / 2;
  const byType = groupBy(nodes, (node) => node.type || "concept");
  Object.values(byType).forEach((items) => {
    items.sort((a, b) => Number(b.degree || 0) - Number(a.degree || 0) || String(a.label).localeCompare(String(b.label)));
  });
  const orderedTypes = ["hypothesis", "source", "process", "material", "reagent", "property", "parameter", "metric", "equipment", "risk", "observation", "concept"];
  orderedTypes.forEach((type, typeIndex) => {
    const items = byType[type] || [];
    const ring = typeRing(type, width, height);
    const phase = stableNumber(type) * Math.PI * 2 + typeIndex * 0.23;
    items.forEach((node, index) => {
      const angle = phase + (Math.PI * 2 * index) / Math.max(1, items.length);
      const jitter = 1 + (stableNumber(node.id) - 0.5) * 0.12;
      node.x = centerX + Math.cos(angle) * ring * jitter;
      node.y = centerY + Math.sin(angle) * ring * jitter;
      node.vx = 0;
      node.vy = 0;
    });
  });

  const byId = new Map(nodes.map((node) => [node.id, node]));
  const minDistance = 48;
  for (let step = 0; step < 210; step += 1) {
    const cooling = 1 - step / 210;
    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        const a = nodes[i];
        const b = nodes[j];
        const dx = a.x - b.x || 0.1;
        const dy = a.y - b.y || 0.1;
        const distance = Math.max(1, Math.hypot(dx, dy));
        const desired = minDistance + nodeRadius(a) + nodeRadius(b);
        const repulsion = (desired * desired * 0.018) / distance;
        a.vx += (dx / distance) * repulsion;
        a.vy += (dy / distance) * repulsion;
        b.vx -= (dx / distance) * repulsion;
        b.vy -= (dy / distance) * repulsion;
        if (distance < desired) {
          const push = (desired - distance) * 0.018;
          a.vx += (dx / distance) * push;
          a.vy += (dy / distance) * push;
          b.vx -= (dx / distance) * push;
          b.vy -= (dy / distance) * push;
        }
      }
    }
    edges.forEach((edge) => {
      const a = byId.get(edge.source);
      const b = byId.get(edge.target);
      if (!a || !b) return;
      const dx = b.x - a.x || 0.1;
      const dy = b.y - a.y || 0.1;
      const distance = Math.max(1, Math.hypot(dx, dy));
      const target = edgeTargetDistance(a, b);
      const force = (distance - target) * 0.0055 * cooling;
      a.vx += (dx / distance) * force;
      a.vy += (dy / distance) * force;
      b.vx -= (dx / distance) * force;
      b.vy -= (dy / distance) * force;
    });
    nodes.forEach((node) => {
      const ring = typeRing(node.type, width, height);
      const dx = node.x - centerX || 0.1;
      const dy = node.y - centerY || 0.1;
      const distance = Math.max(1, Math.hypot(dx, dy));
      const radial = (ring - distance) * 0.003;
      node.vx += (dx / distance) * radial;
      node.vy += (dy / distance) * radial;
      node.vx += (centerX - node.x) * 0.0009;
      node.vy += (centerY - node.y) * 0.0009;
      node.x = clamp(node.x + node.vx, 36, width - 36);
      node.y = clamp(node.y + node.vy, 34, height - 34);
      node.vx *= 0.72;
      node.vy *= 0.72;
    });
  }
}

function onGraphWheel(event) {
  event.preventDefault();
  zoomGraph(event.deltaY > 0 ? 0.88 : 1.14, graphPointFromEvent(event));
}

function onGraphPointerDown(event) {
  const nodeElement = event.target.closest?.("[data-node-id]");
  const edgeElement = event.target.closest?.("[data-edge-id]");
  const point = graphPointFromEvent(event);
  graphView.moved = false;
  $("#graphSvg").setPointerCapture?.(event.pointerId);

  if (nodeElement) {
    const node = graphView.nodeById.get(nodeElement.dataset.nodeId);
    if (!node) return;
    const world = screenToWorld(point);
    graphView.dragging = {
      pointerId: event.pointerId,
      nodeId: node.id,
      offsetX: node.x - world.x,
      offsetY: node.y - world.y,
    };
    return;
  }

  if (edgeElement) {
    graphView.pendingSelection = { type: "edge", id: edgeElement.dataset.edgeId };
    return;
  }

  graphView.panning = {
    pointerId: event.pointerId,
    startX: point.x,
    startY: point.y,
    originX: graphView.transform.x,
    originY: graphView.transform.y,
  };
}

function onGraphPointerMove(event) {
  const point = graphPointFromEvent(event);
  if (graphView.dragging?.pointerId === event.pointerId) {
    const node = graphView.nodeById.get(graphView.dragging.nodeId);
    if (!node) return;
    const world = screenToWorld(point);
    node.x = clamp(world.x + graphView.dragging.offsetX, -2000, 2000);
    node.y = clamp(world.y + graphView.dragging.offsetY, -2000, 2000);
    node.pinned = true;
    graphView.moved = true;
    renderGraphScene();
    renderGraphDetails();
    return;
  }

  if (graphView.panning?.pointerId === event.pointerId) {
    graphView.transform.x = graphView.panning.originX + point.x - graphView.panning.startX;
    graphView.transform.y = graphView.panning.originY + point.y - graphView.panning.startY;
    graphView.moved = true;
    renderGraphScene();
  }
}

function onGraphPointerUp(event) {
  $("#graphSvg").releasePointerCapture?.(event.pointerId);
  if (graphView.dragging?.pointerId === event.pointerId) {
    if (!graphView.moved) selectGraphItem("node", graphView.dragging.nodeId);
    graphView.dragging = null;
    return;
  }
  if (graphView.pendingSelection && !graphView.moved) {
    selectGraphItem(graphView.pendingSelection.type, graphView.pendingSelection.id);
  } else if (graphView.panning && !graphView.moved) {
    graphView.selected = null;
    renderGraphScene();
    renderGraphDetails();
  }
  graphView.pendingSelection = null;
  graphView.panning = null;
}

function selectGraphItem(type, id) {
  graphView.selected = { type, id };
  renderGraphScene();
  renderGraphDetails();
}

function renderGraphDetails() {
  const details = $("#graphDetails");
  const selected = graphView.selected;
  if (!selected) {
    details.hidden = true;
    details.innerHTML = "";
    return;
  }

  if (selected.type === "node") {
    const node = graphView.nodeById.get(selected.id);
    if (!node) {
      details.hidden = true;
      return;
    }
    const related = graphView.edges.filter((edge) => edge.source === node.id || edge.target === node.id);
    details.innerHTML = `<button class="tool-button graph-close" type="button" title="Закрыть">×</button>
      <h3>${escapeHtml(node.label)}</h3>
      <div class="graph-detail-meta">
        <span class="graph-chip">${escapeHtml(typeLabel(node.type))}</span>
        <span class="graph-chip">${related.length} связей</span>
        <span class="graph-chip">w ${Number(node.weight || 1).toFixed(1)}</span>
      </div>
      <p>${escapeHtml(node.summary || "Нет краткого описания.")}</p>
      ${related.length ? `<p>${escapeHtml(related.slice(0, 5).map((edge) => relationLine(edge, node.id)).join("; "))}</p>` : ""}`;
  } else {
    const edge = graphView.edgeById.get(selected.id);
    if (!edge) {
      details.hidden = true;
      return;
    }
    const source = graphView.nodeById.get(edge.source);
    const target = graphView.nodeById.get(edge.target);
    details.innerHTML = `<button class="tool-button graph-close" type="button" title="Закрыть">×</button>
      <h3>${escapeHtml(source?.label || edge.source)} → ${escapeHtml(target?.label || edge.target)}</h3>
      <div class="graph-detail-meta">
        <span class="graph-chip">${escapeHtml(edge.relation || "relation")}</span>
        <span class="graph-chip">w ${Number(edge.weight || 1).toFixed(1)}</span>
      </div>
      <p>${escapeHtml(edge.evidence || "Нет фрагмента-доказательства.")}</p>`;
  }
  details.hidden = false;
  $(".graph-close", details)?.addEventListener("click", () => {
    graphView.selected = null;
    renderGraphScene();
    renderGraphDetails();
  });
}

function fitGraphToViewport(animate) {
  if (!graphView.nodes.length) return;
  const padding = 34;
  const bounds = graphBounds(graphView.nodes);
  const width = Math.max(1, bounds.maxX - bounds.minX);
  const height = Math.max(1, bounds.maxY - bounds.minY);
  const k = clamp(Math.min((graphView.width - padding * 2) / width, (graphView.height - padding * 2) / height), 0.36, 1.7);
  graphView.transform = {
    k,
    x: graphView.width / 2 - ((bounds.minX + bounds.maxX) / 2) * k,
    y: graphView.height / 2 - ((bounds.minY + bounds.maxY) / 2) * k,
  };
  if (animate) toast("Граф выровнен");
  renderGraphScene();
}

function zoomGraph(factor, center = { x: graphView.width / 2, y: graphView.height / 2 }) {
  const previous = graphView.transform;
  const nextK = clamp(previous.k * factor, 0.25, 3.2);
  const world = screenToWorld(center);
  graphView.transform = {
    k: nextK,
    x: center.x - world.x * nextK,
    y: center.y - world.y * nextK,
  };
  renderGraphScene();
}

function screenToWorld(point) {
  return {
    x: (point.x - graphView.transform.x) / graphView.transform.k,
    y: (point.y - graphView.transform.y) / graphView.transform.k,
  };
}

function graphPointFromEvent(event) {
  const rect = $("#graphSvg").getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function graphDefs() {
  const defs = svgEl("defs");
  const marker = svgEl("marker", {
    id: "arrow",
    viewBox: "0 0 10 10",
    refX: 9,
    refY: 5,
    markerWidth: 5,
    markerHeight: 5,
    orient: "auto-start-reverse",
  });
  marker.appendChild(svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#9db0a8", opacity: 0.85 }));
  defs.appendChild(marker);
  return defs;
}

function edgePath(source, target, index) {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const distance = Math.max(1, Math.hypot(dx, dy));
  const offset = ((index % 5) - 2) * Math.min(18, distance * 0.04);
  const nx = -dy / distance;
  const ny = dx / distance;
  const sourceR = source.r || nodeRadius(source);
  const targetR = target.r || nodeRadius(target);
  const x1 = source.x + (dx / distance) * sourceR;
  const y1 = source.y + (dy / distance) * sourceR;
  const x2 = target.x - (dx / distance) * (targetR + 5);
  const y2 = target.y - (dy / distance) * (targetR + 5);
  const cx = (x1 + x2) / 2 + nx * offset;
  const cy = (y1 + y2) / 2 + ny * offset;
  return `M ${x1.toFixed(1)} ${y1.toFixed(1)} Q ${cx.toFixed(1)} ${cy.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`;
}

function graphBounds(nodes) {
  return nodes.reduce(
    (bounds, node) => ({
      minX: Math.min(bounds.minX, node.x - nodeRadius(node) - 44),
      maxX: Math.max(bounds.maxX, node.x + nodeRadius(node) + 80),
      minY: Math.min(bounds.minY, node.y - nodeRadius(node) - 28),
      maxY: Math.max(bounds.maxY, node.y + nodeRadius(node) + 28),
    }),
    { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity },
  );
}

function graphKey(nodes, edges) {
  return `${nodes.map((node) => node.id).sort().join("|")}::${edges.map((edge) => edge.id).sort().join("|")}`;
}

function nodeRadius(node) {
  const base = node.type === "hypothesis" ? 12 : node.type === "source" ? 10 : 8;
  return Math.max(base, Math.min(22, base + Number(node.weight || 1) * 1.25 + Math.sqrt(Number(node.degree || 0)) * 1.5));
}

function edgeTargetDistance(source, target) {
  if (source.type === "hypothesis" || target.type === "hypothesis") return 118;
  if (source.type === "source" || target.type === "source") return 138;
  return 104;
}

function typeRing(type, width, height) {
  const radius = Math.min(width, height) * 0.42;
  const factors = {
    hypothesis: 0.28,
    source: 0.34,
    process: 0.56,
    material: 0.66,
    reagent: 0.76,
    property: 0.86,
    parameter: 0.9,
    metric: 0.94,
    equipment: 0.72,
    risk: 0.9,
    observation: 0.82,
  };
  return radius * (factors[type] || 0.84);
}

function relationLine(edge, nodeId) {
  const otherId = edge.source === nodeId ? edge.target : edge.source;
  const other = graphView.nodeById.get(otherId);
  return `${edge.relation || "related"}: ${other?.label || otherId}`;
}

function typeLabel(type) {
  return (
    {
      source: "источник",
      material: "материал",
      process: "процесс",
      reagent: "реагент",
      property: "свойство",
      parameter: "параметр",
      metric: "метрика",
      equipment: "оборудование",
      risk: "риск",
      hypothesis: "гипотеза",
      observation: "наблюдение",
    }[type] || "концепт"
  );
}

function groupBy(items, getter) {
  return items.reduce((groups, item) => {
    const key = getter(item);
    groups[key] = groups[key] || [];
    groups[key].push(item);
    return groups;
  }, {});
}

function stableNumber(value) {
  let hash = 0;
  const text = String(value || "");
  for (let index = 0; index < text.length; index += 1) {
    hash = (hash * 31 + text.charCodeAt(index)) >>> 0;
  }
  return (hash % 10000) / 10000;
}

async function api(path, options = {}) {
  const headers = { "X-User": encodeURIComponent(actor()), ...(options.headers || {}) };
  const fetchOptions = { ...options, headers };
  if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
    fetchOptions.body = JSON.stringify(options.json);
    delete fetchOptions.json;
  }
  const response = await fetch(path, fetchOptions);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (_) {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  return response.json();
}

function apiUpload(path, options = {}, handlers = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open(options.method || "POST", path);
    xhr.setRequestHeader("X-User", encodeURIComponent(actor()));
    Object.entries(options.headers || {}).forEach(([key, value]) => xhr.setRequestHeader(key, value));
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      handlers.onProgress?.((event.loaded / event.total) * 100, event);
    };
    xhr.upload.onload = () => handlers.onUploaded?.();
    xhr.onerror = () => reject(new Error("Не удалось выполнить запрос"));
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText || "{}"));
        } catch (error) {
          reject(new Error(`Некорректный ответ сервера: ${error.message}`));
        }
        return;
      }
      let detail = xhr.statusText || "Ошибка запроса";
      try {
        const payload = JSON.parse(xhr.responseText || "{}");
        detail = payload.detail || detail;
      } catch (_) {
        detail = xhr.responseText || detail;
      }
      reject(new Error(detail));
    };
    xhr.send(options.body || null);
  });
}

function openExport(type) {
  if (!state.projectId) return;
  window.open(`/api/projects/${state.projectId}/export.${type}`, "_blank", "noopener");
}

function startBusyStages(steps) {
  stopBusyStages();
  if (!steps?.length) {
    setBusy(true);
    return;
  }
  const timeline = [];
  let total = 0;
  steps.forEach((step) => {
    const duration = Number(step.duration || 1000);
    timeline.push({ ...step, start: total, end: total + duration });
    total += duration;
  });
  const startedAt = performance.now();
  const tick = () => {
    const rawElapsed = performance.now() - startedAt;
    const elapsed = Math.min(rawElapsed, Math.max(1, total - 1));
    const current = timeline.find((step) => elapsed >= step.start && elapsed < step.end) || timeline.at(-1);
    const next = timeline[timeline.indexOf(current) + 1];
    const localElapsed = next ? elapsed : rawElapsed;
    const local = clamp((localElapsed - current.start) / Math.max(1, current.end - current.start), 0, 1);
    const target = next ? Number(next.progress ?? current.progress ?? 0) : Number(current.target ?? 94);
    const progress = Number(current.progress ?? 0) + (target - Number(current.progress ?? 0)) * local;
    setBusy({ label: current.label, progress });
    state.busyTimer = window.setTimeout(tick, 420);
  };
  tick();
}

function stopBusyStages() {
  if (state.busyTimer) {
    window.clearTimeout(state.busyTimer);
    state.busyTimer = null;
  }
}

function setBusy(value) {
  const indicator = $("#busyIndicator");
  if (!indicator) return;
  const textNode = $("#busyText");
  const percentNode = $("#busyPercent");
  const fillNode = $("#busyProgressFill");
  if (!value) {
    stopBusyStages();
    indicator.hidden = true;
    indicator.classList.remove("is-determinate");
    indicator.setAttribute("aria-busy", "false");
    $(".busy-progress", indicator)?.removeAttribute("aria-valuenow");
    if (fillNode) fillNode.style.width = "0%";
    if (percentNode) percentNode.textContent = "";
    if (textNode) textNode.textContent = "Выполняется";
    return;
  }

  const options = typeof value === "object" ? value : { label: value === true ? "Выполняется" : String(value) };
  const progress = Number.isFinite(Number(options.progress)) ? clamp(Number(options.progress), 0, 100) : null;
  const progressBar = $(".busy-progress", indicator);
  indicator.hidden = false;
  indicator.setAttribute("aria-busy", "true");
  indicator.classList.toggle("is-determinate", progress !== null);
  if (textNode) textNode.textContent = options.label || "Выполняется";
  if (percentNode) percentNode.textContent = progress !== null ? `${Math.round(progress)}%` : "";
  if (fillNode) fillNode.style.width = progress !== null ? `${progress}%` : "0%";
  if (progress !== null) progressBar?.setAttribute("aria-valuenow", String(Math.round(progress)));
  else progressBar?.removeAttribute("aria-valuenow");
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => {
    node.hidden = true;
  }, 3600);
}

function actor() {
  return $("#actorInput").value.trim() || "User";
}

function svgEl(name, attrs = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function svgText(x, y, text, anchor, className) {
  const element = svgEl("text", { x, y, "text-anchor": anchor, class: className });
  element.textContent = text;
  return element;
}

function colorForType(type) {
  return {
    source: "#1f2523",
    material: "#08786f",
    process: "#6653a8",
    reagent: "#c95243",
    property: "#2f8d58",
    parameter: "#8caf82",
    metric: "#a77a11",
    equipment: "#4c6f88",
    risk: "#9d3c35",
    hypothesis: "#0b9a8c",
    observation: "#607064",
  }[type] || "#607064";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMarkdown(src) {
  const raw = String(src ?? "");
  const placeholders = [];
  const stash = (htmlValue) => {
    placeholders.push(htmlValue);
    return `\x00${placeholders.length - 1}\x00`;
  };

  // Fenced code blocks first, so their content is not further processed.
  let work = raw.replace(/```[^\n]*\n?([\s\S]*?)```/g, (_, code) =>
    stash(`<pre class="md-pre"><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`),
  );

  work = escapeHtml(work);

  // Inline code.
  work = work.replace(/`([^`\n]+)`/g, (_, code) => stash(`<code class="md-code">${code}</code>`));
  // Links [text](url).
  work = work.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (match, label, url) => {
    const safe = safeExternalUrl(url);
    return safe ? `<a href="${safe}" target="_blank" rel="noopener noreferrer">${label}</a>` : match;
  });
  // Bold and italic.
  work = work.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  work = work.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  work = work.replace(/(^|[^\w])_([^_\n]+)_/g, "$1<em>$2</em>");

  // Block-level assembly.
  const lines = work.split("\n");
  const out = [];
  let listType = null;
  let paragraph = [];
  const flushParagraph = () => {
    if (paragraph.length) {
      out.push(`<p>${paragraph.join("<br>")}</p>`);
      paragraph = [];
    }
  };
  const closeList = () => {
    if (listType) {
      out.push(`</${listType}>`);
      listType = null;
    }
  };
  for (const line of lines) {
    if (/^\s*$/.test(line)) {
      flushParagraph();
      closeList();
      continue;
    }
    let match;
    if ((match = line.match(/^\s*(#{1,6})\s+(.*)$/))) {
      flushParagraph();
      closeList();
      const level = match[1].length;
      out.push(`<h${level} class="md-h">${match[2].trim()}</h${level}>`);
    } else if ((match = line.match(/^\s*(?:[-*+])\s+(.*)$/))) {
      flushParagraph();
      if (listType !== "ul") {
        closeList();
        out.push('<ul class="md-list">');
        listType = "ul";
      }
      out.push(`<li>${match[1].trim()}</li>`);
    } else if ((match = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
      flushParagraph();
      if (listType !== "ol") {
        closeList();
        out.push('<ol class="md-list">');
        listType = "ol";
      }
      out.push(`<li>${match[1].trim()}</li>`);
    } else if (/^\s*(?:---+|\*\*\*+|___+)\s*$/.test(line)) {
      flushParagraph();
      closeList();
      out.push('<hr class="md-hr">');
    } else {
      closeList();
      paragraph.push(line.trim());
    }
  }
  flushParagraph();
  closeList();

  let html = out.join("");
  html = html.replace(/\x00(\d+)\x00/g, (_, index) => placeholders[Number(index)] ?? "");
  return html;
}

function safeExternalUrl(value) {
  const text = String(value || "").trim();
  if (!text) return "";
  try {
    const url = new URL(text);
    return ["http:", "https:"].includes(url.protocol) ? url.href : "";
  } catch (_) {
    return "";
  }
}

function formatDate(value) {
  if (!value) return "";
  return new Date(value).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function fileType(filename, contentType) {
  const extension = String(filename || "").split(".").pop();
  if (extension && extension !== filename) return extension.slice(0, 6).toUpperCase();
  return String(contentType || "file").split("/").pop().slice(0, 6).toUpperCase();
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function cssEscape(value) {
  if (window.CSS?.escape) return window.CSS.escape(value);
  return String(value).replaceAll('"', '\\"').replaceAll("\\", "\\\\");
}

function trim(value, length) {
  const text = String(value || "");
  return text.length > length ? `${text.slice(0, length - 1)}…` : text;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function intersects(a, b) {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}
