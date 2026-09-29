"use strict";
const $ = (selector) => document.querySelector(selector);
const state = {run: null, runId: null, tab: "signals", timer: null, catalog: null, config: null, view: "search"};
const esc = (x) => String(x ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const url = (x) => {try {const u = new URL(x); return ["http:","https:"].includes(u.protocol) ? u.href : "";} catch {return "";}};
const dateText = (x) => x ? new Date(x).toLocaleDateString("ru-RU", {day:"numeric", month:"short", year:"numeric"}) : "Дата не установлена";
const statuses = {queued:"В очереди",running:"В работе",completed:"Завершено",partial:"Частичная выдача",failed:"Ошибка",cancelled:"Остановлено",interrupted:"Прервано"};
const verdicts = {mature:"Зрелая технология",hype_or_noise:"Недостаточно содержательных подтверждений",irrelevant:"Не соответствует запросу",insufficient_evidence:"Недостаточно доказательств",weak_signal:"Слабый сигнал"};
const types = {preprint:"Препринт",publication:"Научная публикация",institution:"Институциональный источник",industry_media:"Отраслевое медиа",press_release:"Пресс-релиз",website:"Веб-источник",patent:"Патентный источник"};
const trust = {high:"Высокая",medium:"Средняя",low:"Пониженная",unverified:"Не установлена"};
const axes = {early_stage:"Ранняя стадия",novelty:"Конкретная новизна",momentum:"Значимое изменение",evidence:"Доказательства",maturity:"Зрелость"};
const values = {strong:"Сильное подтверждение",partial:"Частичное подтверждение",unknown:"Нет данных",contradictory:"Противоречие"};
const stages = ["План","Поиск","Документы","Кандидаты","Проверка","Анализ","Результат"];

async function api(path, options={}) {
  const headers = {...options.headers};
  if (options.body && typeof options.body !== "string") {headers["Content-Type"]="application/json"; options.body=JSON.stringify(options.body);}
  const response = await fetch(path, {...options,headers});
  if (!response.ok) {
    let message = `Ошибка запроса (${response.status})`;
    try {const data = await response.json(); message = typeof data.detail === "string" ? data.detail : data.detail?.map(x=>x.msg).join("; ") || message;} catch {}
    throw new Error(message);
  }
  return response.json();
}
function toast(message) {$("#toast").textContent=message; $("#toast").hidden=false; clearTimeout(toast.timer); toast.timer=setTimeout(()=>$("#toast").hidden=true,6500);}
function view(name) {
  state.view=name;
  $("#searchView").hidden=name!=="search"; $("#catalogView").hidden=name!=="catalog";
  $("#navSearch").classList.toggle("active",name==="search"); $("#navCatalog").classList.toggle("active",name==="catalog");
  $("#pageTitle").textContent=name==="search"?"Обзор сигналов":"Каталог примеров";
}
async function refreshConfig() {
  state.config=await api("/api/radar/config");
  $("#connectionDot").classList.toggle("ready",state.config.ready);
  $("#connectionText").textContent=state.config.ready?"Сервис готов" : "Поиск не настроен";
  $("#configNotice").hidden=state.config.ready;
  $("#configNotice").textContent=state.config.message || "";
}
async function history() {
  const data=await api("/api/searches");
  $("#history").innerHTML=data.runs.length?data.runs.map(r=>`<button class="history-item ${r.id===state.runId?"selected":""}" data-run="${esc(r.id)}"><strong>${esc(r.query)}</strong><small>${dateText(r.created_at)} · ${esc(statuses[r.status]||r.status)} · ${r.signal_count}</small></button>`).join(""):'<p class="sidebar-note">Здесь появятся ваши исследования.</p>';
}
async function loadRun(id) {
  clearTimeout(state.timer); state.runId=id; sessionStorage.setItem("idea.radar.run",id);
  const run=await api(`/api/searches/${encodeURIComponent(id)}`);
  if(state.runId!==id)return;
  state.run=run; renderRun();
  if(["running","queued"].includes(run.status)) state.timer=setTimeout(()=>loadRun(id).catch(e=>{toast(e.message); state.timer=setTimeout(()=>loadRun(id).catch(e=>toast(e.message)),5000);}),1800);
  else history().catch(()=>{});
}
async function startSearch(event) {
  event?.preventDefault();
  const button=$("#searchButton"); button.disabled=true;
  try {
    await refreshConfig();
    const response=await api("/api/searches",{method:"POST",body:{query:$("#queryInput").value.trim(),as_of:$("#asOfInput").value,limit:15}});
    state.tab="signals"; view("search"); await loadRun(response.id); await history();
    $("#runPanel").scrollIntoView({behavior:"smooth",block:"start"});
  } catch(e){toast(e.message);} finally{button.disabled=false;}
}
function renderRun() {
  const r=state.run, active=["queued","running"].includes(r.status);
  $("#runPanel").hidden=false; $("#welcome").hidden=true;
  $("#runQuery").textContent=r.query;
  $("#runSubtitle").textContent=`${statuses[r.status]||r.status} · На дату ${dateText(r.as_of)}`;
  $("#runStage").textContent=r.stage;
  $("#runTiming").textContent=r.elapsed_seconds?`${Math.round(r.elapsed_seconds)} сек.`:"";
  $("#runSpinner").hidden=!active; $("#cancelButton").hidden=!active;
  $("#stageSteps").innerHTML=stages.map((x,i)=>`<span class="stage-step ${i<r.stage_index?"done":i===r.stage_index?"active":""}" title="${x}"></span>`).join("");
  $("#latestEvent").textContent=r.events.at(-1)?.message || "Подготовка исследования";
  $("#statFound").textContent=r.counters.found; $("#statRead").textContent=r.counters.read;
  $("#statDuplicates").textContent=`Повторных документов: ${r.counters.duplicates}`;
  $("#statCandidates").textContent=r.counters.candidates; $("#statAssessed").textContent=`Проверено: ${r.counters.assessed}`+(r.counters.confident!==undefined?` · уверенных > 75%: ${r.counters.confident}`:"");
  $("#statSignals").innerHTML=`${r.signals.length}<span>/ ${r.limit}</span>`;
  $("#signalsCount").textContent=r.signals.length; $("#rejectedCount").textContent=r.rejected.length; $("#sourcesCount").textContent=r.sources.length;
  $("#runWarnings").hidden=!r.warnings.length;
  $("#runWarnings").innerHTML=[...new Set(r.warnings)].map(w=>`<p>${esc(w)}</p>`).join("");
  renderResults();
  const totals=r.calls.reduce((t,c)=>({input:t.input+(c.usage?.input_tokens||0),output:t.output+(c.usage?.output_tokens||0)}),{input:0,output:0});
  $("#auditContent").innerHTML=`<p>${esc(r.plan?.interpretation||"")}</p><p>${esc(r.score_note)} Сохранённые результаты этого запуска не переоцениваются при открытии истории.</p>${r.plan?`<ul>${r.plan.branches.map(b=>`<li>${esc(b.topic)}: ${esc(b.query_ru)} / ${esc(b.query_en)}</li>`).join("")}</ul>`:""}<p>Автоматических этапов: ${r.calls.length}. Токены: ${totals.input.toLocaleString("ru-RU")} входящих / ${totals.output.toLocaleString("ru-RU")} исходящих.</p><table><thead><tr><th>Этап</th><th>Время</th><th>Результат</th></tr></thead><tbody>${r.calls.map(c=>`<tr><td>${esc(c.role)}</td><td>${c.seconds} с</td><td>${esc(c.status)}</td></tr>`).join("")}</tbody></table>`;
}
function empty(title,text){return `<div class="empty"><strong>${esc(title)}</strong>${esc(text)}</div>`;}
function renderResults() {
  document.querySelectorAll("[data-tab]").forEach(b=>{b.classList.toggle("active",b.dataset.tab===state.tab);b.setAttribute("aria-selected",String(b.dataset.tab===state.tab));});
  const r=state.run;if(!r)return;
  if(state.tab==="signals"){
    $("#results").innerHTML=r.signals.length?r.signals.map(s=>`<article class="signal-card"><button class="signal-main" data-signal="${esc(s.id)}"><span class="rank">${String(s.rank).padStart(2,"0")}</span><div><span class="signal-domain">${esc(s.domain)}</span><h3 class="signal-title">${esc(s.title)}</h3><p class="signal-summary">${esc(s.why_now?.text||s.summary)}</p></div><div class="score-badge">${s.score}<small>${s.score_kind==="model"?"% уверенность":s.score_kind==="model_partial"?"% предварительно":"из 100"}</small></div></button><div class="signal-foot"><span>${s.source_ids.length} источн.</span><span>${esc(s.stage)}</span><button data-signal="${esc(s.id)}">Открыть инсайт ↗</button></div></article>`).join(""):empty("Подтверждённых сигналов пока нет",["queued","running"].includes(r.status)?"Результаты появятся после чтения источников и проверки кандидатов.":"Проверьте замечания к запуску. Попробуйте более широкое направление или повторите поиск.");
  }else if(state.tab==="rejected"){
    $("#results").innerHTML=r.rejected.length?r.rejected.map(s=>`<article class="rejected-card"><span class="reason-badge">${esc(verdicts[s.verdict])}</span><h3>${esc(s.title)}</h3><p>${esc(s.reason)}</p>${(s.evidence?.length||s.model)?`<button class="subtle" data-signal="${esc(s.id)}">Посмотреть основания ↗</button>`:""}</article>`).join(""):empty("Исключений пока нет","Здесь появятся кандидаты, не прошедшие проверку, и причины решения.");
  }else{
    $("#results").innerHTML=r.sources.length?r.sources.map(sourceCard).join(""):empty("Источники ещё не загружены","Поиск и чтение документов выполняются последовательно.");
  }
}
function sourceCard(s){
  const stateLabel=s.duplicate_of?"Повтор документа":s.status==="read"?"Текст получен":s.status==="excluded_date"?"Исключён по дате":"Не удалось прочитать";
  return `<article class="source-card"><span class="tag">${esc(s.type_label||types[s.type]||s.type)}</span>${s.primary_only?'<span class="tag">Только первичный индикатор</span>':""}<span class="tag">${esc(stateLabel)}</span>${s.cached?'<span class="tag">Из кэша</span>':""}<h3><a href="${esc(url(s.url))}" target="_blank" rel="noopener noreferrer">${esc(s.title)} ↗</a></h3><small>${esc(s.publisher)} · ${dateText(s.published_at)} · ${esc(s.language)}</small><p>Доверенность: ${esc(trust[s.trust]||s.trust)}. ${esc(s.trust_reason)}</p>${s.error?`<p>${esc(s.error)}</p>`:""}<div class="source-actions">${s.status==="read"?`<button class="subtle" data-source="${esc(s.id)}">Прочитанный текст</button>`:""}<span>Получен: ${dateText(s.retrieved_at)}</span></div></article>`;
}
function sourceLinks(ids){return (ids||[]).map(id=>`<button class="evidence-link" data-source="${esc(id)}">${esc(state.run.sources.find(s=>s.id===id)?.publisher||"Источник")} ↗</button>`).join("");}
function modelSection(s){
  const m=s.model; if(!m)return "";
  const row=c=>`<div class="predictor"><strong>${esc(c.label)}<small>${esc(c.display_value)} · ${esc(c.relative)}</small></strong><div>${c.contribution>0?"+":""}${c.contribution.toFixed(2)} к логиту</div></div>`;
  const partial=m.fallback, note=partial?"Полный сбор признаков не завершился. Оценка рассчитана с медианной подстановкой и не отражает индивидуальные данные кандидата; она не подтверждает слабый сигнал.":"Вклады складываются в итоговую оценку точно.";
  return `<section class="report-section"><h3>Почему модель так решила</h3><p class="muted">${partial?"Предварительная ":""}оценка слабого сигнала ${Math.round(m.probability*100)}% при пороге ${Math.round(m.threshold*100)}%. ${m.search_term?`Поисковый термин: ${esc(m.search_term)}. `:""}${note}</p><h4>За сигнал</h4>${(m.top_for||[]).map(row).join("")||"<p class=muted>Нет</p>"}<h4>Против</h4>${(m.top_against||[]).map(row).join("")||"<p class=muted>Нет</p>"}${(m.exclusion_reasons||[]).length?`<h4>Причины исключения</h4><ul class="report-limitations">${m.exclusion_reasons.map(x=>`<li>${esc(x)}</li>`).join("")}</ul>`:""}</section>`;
}
async function openSignal(id){
  const result=await api(`/api/signals/${encodeURIComponent(id)}`),s=result.signal;
  const blocks=[["description","Технология"],["why_now","Почему сейчас"],["why_early","Почему это ранняя стадия"],["advantage","Потенциальное преимущество"],["case",s.case_kind==="observed"?"Кейс из источника":s.case_kind==="proposed"?"Предполагаемое применение":"Кейс не подтверждён"]];
  $("#reportContent").innerHTML=`<div class="eyebrow">${esc(s.domain)} / ${esc(verdicts[s.verdict])}</div><h2>${esc(s.title)}</h2><div class="report-meta"><span>${esc(s.stage||"Стадия не установлена")}</span><span>${s.score_kind==="model"?`Уверенность модели ${s.score}%`:s.score_kind==="model_partial"?`Предварительная оценка модели ${s.score}%`:`${s.score}/100 по критериям`}</span><span>${dateText(result.as_of)}</span></div><p class="muted">${esc(s.reason)}</p>${blocks.filter(([key])=>s[key]).map(([key,title])=>`<section class="report-section"><h3>${title}</h3><p>${esc(s[key].text)}</p>${sourceLinks(s[key].source_ids)}${!s[key].grounded?'<div class="attribution">Прямое подтверждение этого вывода не установлено.</div>':""}</section>`).join("")}${s.business_use?`<section class="report-section"><h3>Возможное применение для бизнеса</h3><p>${esc(s.business_use)}</p><div class="attribution">Аналитическое предположение системы.</div></section>`:""}${modelSection(s)}<section class="report-section"><h3>${["model","model_partial"].includes(s.score_kind)?"Проверка источников (LLM)":"Признаки решения"}</h3>${(s.predictors||[]).map(p=>`<div class="predictor"><strong>${esc(axes[p.name])}<small>${esc(values[p.value])}</small></strong><div>${esc(p.explanation)}${sourceLinks(p.source_ids)}</div></div>`).join("")}</section><section class="report-section"><h3>Доказательства и возражения</h3>${(s.evidence||[]).map(e=>{const src=result.sources.find(x=>x.id===e.source_id);return `<div class="evidence-item"><span class="tag">${e.relation==="contradicts"?"Контрдоказательство":e.relation==="context"?"Контекст":"Основание"}</span><p>${esc(e.claim)}</p><blockquote>${esc(e.quote)}</blockquote><a href="${esc(url(src?.url))}" target="_blank" rel="noopener noreferrer">${esc(src?.title)} ↗</a><div class="attribution">${dateText(src?.published_at)} · ${esc(src?.language)} · Фрагмент найден в тексте</div>${sourceLinks([e.source_id])}</div>`;}).join("")}</section><section class="report-section"><h3>Ограничения</h3><ul class="report-limitations">${(s.limitations||[]).map(x=>`<li>${esc(x)}</li>`).join("")||"<li>Дополнительные ограничения моделью не указаны; вывод требует экспертной оценки.</li>"}</ul></section><p class="muted" style="font-size:10px">Русское резюме и интерпретация сформированы ИИ. Оригинальные цитаты сохранены. ${s.score_kind==="model"?"Уверенность — калиброванная вероятность обученной модели.":s.score_kind==="model_partial"?"Предварительная оценка логистической регрессии рассчитана с медианной подстановкой пропущенных признаков.":"Оценка не является калиброванной вероятностью."}</p>`;
  $("#reportDialog").showModal();
}
async function openSource(id){
  const runId=state.runId;
  const s=await api(`/api/searches/${encodeURIComponent(runId)}/sources/${encodeURIComponent(id)}`);
  $("#sourceContent").innerHTML=`<div class="eyebrow">Прочитанный источник</div><h2>${esc(s.title)}</h2><p class="muted" style="font-size:11px">${dateText(s.published_at)} · ${esc(s.language)} · ${esc(types[s.type])}</p><p style="font-size:11px"><a href="${esc(url(s.url))}" target="_blank" rel="noopener noreferrer">Открыть оригинал ↗</a></p><p class="notice">${esc(s.trust_reason)}${s.truncated?" Текст сохранён с ограничением длины.":""}</p><div class="source-text">${esc(s.text||s.error||"Текст отсутствует")}</div>`;
  if(!$("#sourceDialog").open)$("#sourceDialog").showModal();
}
async function catalog(){
  view("catalog");
  if(!state.catalog){state.catalog=await api("/api/catalog"); $("#domainFilter").innerHTML='<option value="">Все области</option>'+Object.keys(state.catalog.domains).map(d=>`<option>${esc(d)}</option>`).join("");}
  renderCatalog();
}
function renderCatalog(){
  if(!state.catalog)return;
  const q=$("#catalogFilter").value.toLowerCase(),domain=$("#domainFilter").value;
  const records=state.catalog.records.filter(r=>(!domain||r.domain===domain)&&`${r.title} ${r.companies} ${r.domain}`.toLowerCase().includes(q));
  $("#catalogCount").textContent=`${records.length} из ${state.catalog.count}`;
  $("#catalogList").innerHTML=records.map(r=>`<article class="catalog-card"><span class="signal-domain">${esc(r.domain)}</span><h3>${r.id}. ${esc(r.title)}</h3><p>${esc(r.companies)}</p><span class="tag">${esc(r.stage)}</span><span class="tag">Авторский балл: ${r.reference_score}</span><details><summary>Объяснение и источники из Excel</summary><p>${esc(r.reference_explanation)}</p><p>${esc(r.mention_trend)}</p>${r.sources.map(s=>`<a href="${esc(url(s.url))}" target="_blank" rel="noopener noreferrer">${esc(s.title)} ↗</a>`).join("")}</details><button class="subtle" data-check-example="${r.id}">Проверить по открытым источникам ↗</button></article>`).join("")||empty("Ничего не найдено","Измените поисковый запрос или область.");
}
async function download(){
  const res=await fetch(`/api/searches/${state.runId}/export.md`);
  if(!res.ok)throw new Error("Не удалось выгрузить отчёт");
  const href=URL.createObjectURL(await res.blob()),a=document.createElement("a");a.href=href;a.download=`idea-${state.runId}.md`;a.click();setTimeout(()=>URL.revokeObjectURL(href),1000);
}
document.addEventListener("click",async e=>{
  const button=e.target.closest("button"); if(!button)return;
  try {
    if(button.dataset.query){view("search");$("#queryInput").value=button.dataset.query;$("#queryInput").focus();}
    if(button.dataset.run){view("search");await loadRun(button.dataset.run);await history();}
    if(button.dataset.tab){state.tab=button.dataset.tab;renderResults();}
    if(button.dataset.signal)await openSignal(button.dataset.signal);
    if(button.dataset.source)await openSource(button.dataset.source);
    if(button.dataset.close)document.getElementById(button.dataset.close).close();
    if(button.dataset.checkExample){const r=state.catalog.records.find(x=>x.id===Number(button.dataset.checkExample));view("search");$("#queryInput").value=`Проверить раннюю стадию технологии: ${r.title}`;$("#queryInput").focus();window.scrollTo({top:0,behavior:"smooth"});toast("Запрос подготовлен. Нажмите «Найти сигналы», чтобы выполнить новый поиск.");}
  }catch(error){toast(error.message);}
});
$("#navSearch").onclick=()=>view("search");
$("#navCatalog").onclick=()=>catalog().catch(e=>toast(e.message));
$("#navMethod").onclick=()=>$("#methodDialog").showModal();
$("#searchForm").onsubmit=startSearch;
$("#refreshHistory").onclick=()=>Promise.all([history(),refreshConfig()]).catch(e=>toast(e.message));
$("#catalogFilter").oninput=renderCatalog;$("#domainFilter").onchange=renderCatalog;
$("#cancelButton").onclick=async()=>{try{await api(`/api/searches/${state.runId}/cancel`,{method:"POST"});toast("Остановка запрошена. Текущие сетевые операции завершатся в пределах таймаута.");}catch(e){toast(e.message);}};
$("#exportButton").onclick=()=>download().catch(e=>toast(e.message));
const today=new Date();today.setMinutes(today.getMinutes()-today.getTimezoneOffset());const iso=today.toISOString().slice(0,10);
$("#asOfInput").value=iso;$("#asOfInput").max=iso;$("#today").textContent=dateText(iso);
Promise.all([refreshConfig(),history()]).then(()=>{const saved=sessionStorage.getItem("idea.radar.run");if(saved)return loadRun(saved);}).catch(e=>toast(e.message));
