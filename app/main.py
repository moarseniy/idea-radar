from __future__ import annotations

import csv
import html
import io
import logging
import mimetypes
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import fitz
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db import Database
from app.document_parser import (
    SUPPORTED_EXTENSIONS,
    normalize_ocr_languages,
    parse_document,
    safe_filename,
)
from app.knowledge import graph_from_hypothesis, heuristic_graph_from_text
from app.openrouter_service import OpenRouterService, OpenRouterServiceError
from app.radar.api import router as radar_router
from app.radar.config import load_radar_settings
from app.radar.pipeline import SearchManager
from app.radar.store import RadarStore
from app.schemas import (
    ChatRequest,
    FeedbackRequest,
    GenerateRequest,
    ProjectCreate,
    ProjectUpdate,
    SampleImportRequest,
    StatusUpdate,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("hypothesis_lab")

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(radar_router)

db = Database(settings.db_path)
ai = OpenRouterService(settings)


@app.middleware("http")
async def no_cache_static_assets(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.on_event("startup")
def startup() -> None:
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    db.initialize()
    radar_settings = load_radar_settings()
    app.state.radar_store = RadarStore(radar_settings.database_url, radar_settings.storage_dir)
    app.state.radar_store.initialize()
    app.state.radar_store.recover()
    app.state.radar_manager = SearchManager(app.state.radar_store, radar_settings)
    logger.info(
        "Hypothesis Lab started: model=%s research_model=%s provider=OpenRouter key_present=%s graph_extraction=%s pdf_ocr=%s ocr_langs=%s storage=%s",
        settings.openrouter_model,
        settings.openrouter_research_model,
        bool(settings.openrouter_api_key),
        settings.openrouter_graph_extraction,
        settings.pdf_ocr_enabled,
        ",".join(settings.pdf_ocr_languages),
        settings.storage_dir,
    )


REACT_INDEX = STATIC_DIR / "app" / "index.html"  # сборка frontend/ (npm run build)


def spa_index() -> FileResponse:
    """React-интерфейс, если собран; иначе прежняя статическая страница радара."""
    return FileResponse(REACT_INDEX if REACT_INDEX.exists() else STATIC_DIR / "radar.html")


@app.get("/")
def index() -> FileResponse:
    return spa_index()


@app.get("/legacy")
def legacy_radar() -> FileResponse:
    return FileResponse(STATIC_DIR / "radar.html")


# Маршруты клиентского роутера React: прямой заход или обновление страницы.
for _path in ("/runs/{rest:path}", "/signals/{rest:path}", "/stats", "/registry", "/catalog", "/method"):
    app.add_api_route(_path, spa_index, methods=["GET"], include_in_schema=False)


@app.get("/lab")
def legacy_lab() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.on_event("shutdown")
def shutdown() -> None:
    app.state.radar_manager.close()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "app": settings.app_name,
        "openrouter_enabled": ai.enabled,
        "openrouter_model": settings.openrouter_model,
        "openrouter_research_model": settings.openrouter_research_model,
        "openrouter_research_max_sources": settings.openrouter_research_max_sources,
        "provider": "OpenRouter",
        "graph_extraction": settings.openrouter_graph_extraction,
        "pdf_ocr_enabled": settings.pdf_ocr_enabled,
        "pdf_ocr_languages": settings.pdf_ocr_languages,
        "pdf_text_layer_min_chars": settings.pdf_text_layer_min_chars,
        "pdf_vision_max_pages": settings.pdf_vision_max_pages,
    }


@app.post("/api/openrouter/check")
def check_openrouter() -> dict[str, Any]:
    try:
        return ai.check_connection()
    except OpenRouterServiceError as exc:
        logger.exception(
            "OpenRouter connectivity check failed: model=%s error=%s",
            settings.openrouter_model,
            exc,
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/projects")
def list_projects() -> dict[str, Any]:
    return {"projects": db.list_projects()}


@app.post("/api/projects")
def create_project(payload: ProjectCreate, x_user: str | None = Header(default=None, alias="X-User")) -> dict[str, Any]:
    project = db.create_project(payload.model_dump(), _actor(x_user))
    return {"project": project}


@app.patch("/api/projects/{project_id}")
def update_project(
    project_id: str,
    payload: ProjectUpdate,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict[str, Any]:
    _require_project(project_id)
    project = db.update_project(project_id, payload.model_dump(exclude_unset=True), _actor(x_user))
    return {"project": project}


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str) -> dict[str, Any]:
    if not db.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Проект не найден")
    return {"deleted": True}


@app.get("/api/projects/{project_id}/state")
def project_state(project_id: str) -> dict[str, Any]:
    project = _require_project(project_id)
    return {
        "project": project,
        "documents": db.list_documents(project_id, include_text=False),
        "hypotheses": db.list_hypotheses(project_id),
        "graph": db.get_graph(project_id),
        "feedback": db.list_feedback(project_id),
        "chat": db.list_chat(project_id),
        "events": db.list_events(project_id),
        "runtime": {
            "openrouter_enabled": ai.enabled,
            "openrouter_model": settings.openrouter_model,
            "openrouter_research_model": settings.openrouter_research_model,
        },
    }


@app.post("/api/projects/{project_id}/documents")
async def upload_document(
    project_id: str,
    file: UploadFile = File(...),
    ocr_languages: str | None = Form(default=None),
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict[str, Any]:
    _require_project(project_id)
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Файл превышает MAX_UPLOAD_BYTES")
    result = _ingest_document_bytes(
        project_id=project_id,
        filename=file.filename or "document",
        content_type=file.content_type or "application/octet-stream",
        data=data,
        actor=_actor(x_user),
        ocr_languages=_request_ocr_languages(ocr_languages),
    )
    return result


@app.post("/api/projects/{project_id}/documents/import-samples")
def import_samples(
    project_id: str,
    payload: SampleImportRequest,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict[str, Any]:
    _require_project(project_id)
    if not settings.sample_data_dir.exists():
        raise HTTPException(status_code=404, detail=f"Папка с данными не найдена: {settings.sample_data_dir}")

    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in payload.extensions}
    imported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    actor = _actor(x_user)
    sample_paths = [
        path
        for path in settings.sample_data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and path.suffix.lower() in extensions
    ]

    for path in _ordered_sample_paths(sample_paths):
        if len(imported) >= payload.max_files:
            break
        if path.stat().st_size > settings.max_upload_bytes:
            skipped.append({"path": str(path), "reason": "too_large"})
            continue
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            result = _ingest_document_bytes(
                project_id=project_id,
                filename=str(path.relative_to(settings.sample_data_dir)),
                content_type=content_type,
                data=path.read_bytes(),
                actor=actor,
                ocr_languages=_request_ocr_languages(payload.ocr_languages),
            )
            imported.append(result["document"])
        except Exception as exc:  # noqa: BLE001
            skipped.append({"path": str(path), "reason": str(exc)})

    return {"imported": imported, "skipped": skipped}


@app.post("/api/projects/{project_id}/generate")
def generate_hypotheses(
    project_id: str,
    payload: GenerateRequest,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict[str, Any]:
    return _generate_hypotheses_core(project_id, payload, _actor(x_user))


@app.post("/api/projects/{project_id}/generate-with-files")
async def generate_hypotheses_with_files(
    project_id: str,
    payload_json: str = Form(...),
    files: list[UploadFile] | None = File(default=None),
    ocr_languages: str | None = Form(default=None),
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict[str, Any]:
    try:
        payload = GenerateRequest.model_validate_json(payload_json)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Некорректные параметры генерации: {exc}") from exc

    prompt_documents: list[dict[str, Any]] = []
    selected_ocr_languages = _request_ocr_languages(ocr_languages)
    for file in files or []:
        data = await file.read()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"Файл {file.filename} превышает MAX_UPLOAD_BYTES")
        prompt_documents.append(
            _parse_prompt_attachment(
                filename=file.filename or "prompt-attachment",
                content_type=file.content_type or "application/octet-stream",
                data=data,
                ocr_languages=selected_ocr_languages,
            )
        )
    return _generate_hypotheses_core(project_id, payload, _actor(x_user), prompt_documents=prompt_documents)


@app.delete("/api/projects/{project_id}/hypotheses")
def clear_hypotheses(
    project_id: str,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict[str, Any]:
    _require_project(project_id)
    result = db.clear_hypotheses(project_id, _actor(x_user))
    logger.info(
        "Hypotheses cleared: project_id=%s hypotheses=%s nodes=%s edges=%s",
        project_id,
        result.get("hypotheses"),
        result.get("nodes"),
        result.get("edges"),
    )
    return {"cleared": result, "state": project_state(project_id)}


def _generate_hypotheses_core(
    project_id: str,
    payload: GenerateRequest,
    actor: str,
    prompt_documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    project = _require_project(project_id)
    prompt_documents = prompt_documents or []
    documents = [*prompt_documents, *db.list_documents(project_id, include_text=True)]
    graph = db.get_graph(project_id)
    feedback = db.list_feedback(project_id)
    weights = {**project.get("settings", {}), **payload.weights}
    research_context: dict[str, Any] | None = None
    research_meta: dict[str, Any] = {"enabled": False}

    if payload.research_enabled:
        try:
            research_context, research_meta = ai.research_topic(
                project=project,
                documents=documents,
                graph=graph,
                query=payload.research_query,
                max_sources=min(payload.research_sources, settings.openrouter_research_max_sources),
            )
            research_meta = {
                **research_meta,
                "enabled": True,
                "query": research_context.get("query"),
                "sources": research_context.get("sources", []),
            }
            logger.info(
                "Research completed: project_id=%s model=%s actor=%s sources=%s query=%s",
                project_id,
                settings.openrouter_research_model,
                actor,
                len(research_context.get("sources", [])),
                research_context.get("query"),
            )
        except OpenRouterServiceError as exc:
            logger.exception(
                "Research failed: project_id=%s model=%s actor=%s documents=%s graph_nodes=%s graph_edges=%s query=%s error=%s",
                project_id,
                settings.openrouter_research_model,
                actor,
                len(documents),
                len(graph.get("nodes", [])),
                len(graph.get("edges", [])),
                payload.research_query,
                exc,
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        hypotheses, meta = ai.generate_hypotheses(
            project=project,
            documents=documents,
            graph=graph,
            feedback=feedback,
            count=payload.count,
            weights=weights,
            exclusions=payload.exclusions,
            include_roadmap=payload.include_roadmap,
            research_context=research_context,
        )
    except OpenRouterServiceError as exc:
        logger.exception(
            "Hypothesis generation failed: project_id=%s model=%s actor=%s documents=%s graph_nodes=%s graph_edges=%s feedback=%s count=%s error=%s",
            project_id,
            settings.openrouter_model,
            actor,
            len(documents),
            len(graph.get("nodes", [])),
            len(graph.get("edges", [])),
            len(feedback),
            payload.count,
            exc,
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    meta = {**meta, "prompt_attachments": len(prompt_documents), "research": research_meta}
    created = db.add_hypotheses(project_id, hypotheses, actor)
    for hypothesis in created:
        nodes, edges = graph_from_hypothesis(hypothesis)
        db.upsert_graph(project_id, nodes, edges, actor, "hypothesis-generated", source_id=hypothesis["id"])
    return {"hypotheses": created, "meta": meta, "state": project_state(project_id)}


@app.post("/api/projects/{project_id}/chat")
def chat(
    project_id: str,
    payload: ChatRequest,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict[str, Any]:
    project = _require_project(project_id)
    actor = _actor(x_user)
    try:
        answer, meta = ai.chat(
            project=project,
            hypotheses=db.list_hypotheses(project_id),
            graph=db.get_graph(project_id),
            feedback=db.list_feedback(project_id),
            chat_history=db.list_chat(project_id),
            message=payload.message,
        )
    except OpenRouterServiceError as exc:
        logger.exception(
            "Chat failed: project_id=%s model=%s actor=%s message_chars=%s error=%s",
            project_id,
            settings.openrouter_model,
            actor,
            len(payload.message),
            exc,
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    user_message = db.add_chat_message(project_id, "user", actor, payload.message, event=True)
    assistant_message = db.add_chat_message(project_id, "assistant", "AI", answer)
    return {"messages": [user_message, assistant_message], "meta": meta, "state": project_state(project_id)}


@app.patch("/api/projects/{project_id}/hypotheses/{hypothesis_id}/status")
def update_hypothesis_status(
    project_id: str,
    hypothesis_id: str,
    payload: StatusUpdate,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict[str, Any]:
    _require_project(project_id)
    hypothesis = db.update_hypothesis_status(project_id, hypothesis_id, payload.status, _actor(x_user))
    if not hypothesis:
        raise HTTPException(status_code=404, detail="Гипотеза не найдена")
    return {"hypothesis": hypothesis, "state": project_state(project_id)}


@app.post("/api/projects/{project_id}/hypotheses/{hypothesis_id}/feedback")
def add_hypothesis_feedback(
    project_id: str,
    hypothesis_id: str,
    payload: FeedbackRequest,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict[str, Any]:
    _require_project(project_id)
    if not db.get_hypothesis(hypothesis_id):
        raise HTTPException(status_code=404, detail="Гипотеза не найдена")
    feedback = db.add_feedback(
        project_id=project_id,
        hypothesis_id=hypothesis_id,
        actor=_actor(x_user),
        rating=payload.rating,
        outcome=payload.outcome,
        comment=payload.comment,
    )
    return {"feedback": feedback, "state": project_state(project_id)}


@app.get("/api/projects/{project_id}/export.json")
def export_json(project_id: str) -> JSONResponse:
    _require_project(project_id)
    return JSONResponse(db.export_project(project_id))


@app.get("/api/projects/{project_id}/export.csv")
def export_csv(project_id: str) -> Response:
    project = _require_project(project_id)
    hypotheses = db.list_hypotheses(project_id)
    weights = _export_weights(project)
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "id",
            "title",
            "statement",
            "score",
            "novelty",
            "feasibility",
            "impact",
            "risk",
            "status",
            "evidence_urls",
            "economics",
            "created_at",
        ],
        delimiter=";",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    for item in hypotheses:
        values = _export_hypothesis_values(item, weights)
        writer.writerow(
            {
                **{key: values.get(key, item.get(key)) for key in writer.fieldnames},
                "score": f"{values['score']:.1f}",
                "novelty": f"{values['novelty']:.0f}",
                "feasibility": f"{values['feasibility']:.0f}",
                "impact": f"{values['impact']:.0f}",
                "risk": f"{values['risk']:.0f}",
                "evidence_urls": _evidence_url_text(item),
                "economics": _economics_export_text(item.get("economics")),
            }
        )
    return Response(
        content="\ufeff" + buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="hypotheses.csv"'},
    )


@app.get("/api/projects/{project_id}/export.md")
def export_markdown(project_id: str) -> Response:
    _require_project(project_id)
    data = db.export_project(project_id)
    content = _render_markdown_export(data)
    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="hypotheses.md"'},
    )


@app.get("/api/projects/{project_id}/export.pdf")
def export_pdf(project_id: str) -> Response:
    _require_project(project_id)
    data = db.export_project(project_id)
    content = _render_pdf_export(data)
    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="hypotheses.pdf"'},
    )


def _render_markdown_export(data: dict[str, Any]) -> str:
    project = data.get("project") or {}
    hypotheses = data.get("hypotheses") or []
    documents = data.get("documents") or []
    weights = _export_weights(project)
    lines = [
        f"# {project.get('name') or 'Hypothesis Lab'}",
        "",
        f"**Домен:** {project.get('domain') or '-'}",
        f"**Цель / KPI:** {project.get('goal') or '-'}",
        f"**Ограничения:** {project.get('constraints') or '-'}",
        "",
        f"_Источников: {len(documents)} · Гипотез: {len(hypotheses)}_",
        "",
        "## Ранжирование",
        "",
        "| # | Score | Статус | Гипотеза | Новизна | Реализ. | Эффект | Риск |",
        "|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for index, item in enumerate(hypotheses, start=1):
        values = _export_hypothesis_values(item, weights)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"{values['score']:.1f}",
                    _md_cell(values.get("status")),
                    _md_cell(values.get("title")),
                    f"{values['novelty']:.0f}",
                    f"{values['feasibility']:.0f}",
                    f"{values['impact']:.0f}",
                    f"{values['risk']:.0f}",
                ]
            )
            + " |"
        )
    for index, item in enumerate(hypotheses, start=1):
        values = _export_hypothesis_values(item, weights)
        lines.extend(
            [
                "",
                f"## {index}. {item.get('title') or 'Гипотеза'}",
                "",
                f"**Score:** {values['score']:.1f} · **Статус:** {item.get('status') or '-'}",
                "",
                item.get("statement") or "",
                "",
            ]
        )
        mechanism = item.get("mechanism") or item.get("rationale")
        if mechanism:
            lines.extend(["**Механизм / обоснование:**", "", str(mechanism), ""])
        if item.get("evidence"):
            lines.extend(["**Источники:**", ""])
            for evidence in item.get("evidence") or []:
                source = evidence.get("source") or "source"
                quote = evidence.get("quote") or evidence.get("why") or ""
                url = evidence.get("url") or ""
                suffix = f" ([link]({url}))" if url else ""
                why = evidence.get("why") or ""
                context = f" — {why}" if why and why != quote else ""
                lines.append(f"- **{source}:** {quote}{context}{suffix}")
            lines.append("")
        if item.get("roadmap"):
            lines.extend(["**План внедрения / проверки:**", ""])
            for step in item.get("roadmap") or []:
                lines.append(f"- {step.get('step')}. {step.get('title')} -> {step.get('output')}")
            lines.append("")
        if item.get("economics"):
            lines.extend(["**Возможная экономика:**", ""])
            for economic in item.get("economics") or []:
                lines.append(f"- {_economics_line(economic)}")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def _render_pdf_export(data: dict[str, Any]) -> bytes:
    project = data.get("project") or {}
    hypotheses = data.get("hypotheses") or []
    documents = data.get("documents") or []
    weights = _export_weights(project)
    doc = fitz.open()

    def add_html_page(title: str, body: str, *, subtitle: str = "") -> None:
        page = doc.new_page(width=595, height=842)
        page.draw_rect(fitz.Rect(0, 0, 595, 842), color=(0.98, 0.96, 0.92), fill=(0.98, 0.96, 0.92))
        content = f"""
        <style>
          body {{ font-family: sans-serif; color: #2d2922; font-size: 10.5pt; line-height: 1.35; }}
          h1 {{ margin: 0 0 8pt; font-size: 19pt; color: #15130f; }}
          h2 {{ margin: 0 0 8pt; font-size: 14pt; color: #15130f; }}
          h3 {{ margin: 13pt 0 5pt; font-size: 10.5pt; color: #735121; }}
          p {{ margin: 0 0 8pt; }}
          ul, ol {{ margin: 2pt 0 8pt 18pt; padding: 0; }}
          li {{ margin: 0 0 4pt; }}
          .hero {{ background: #e09830; border-radius: 6pt; padding: 16pt; margin-bottom: 14pt; }}
          .hero h1, .hero p {{ color: #15130f; }}
          .muted {{ color: #766d5e; }}
          .metrics {{ display: flex; gap: 6pt; margin: 8pt 0 12pt; }}
          .metric {{ border: 1px solid #d8cbb5; border-radius: 5pt; padding: 5pt 7pt; background: #fffdf7; }}
          .metric b {{ display: block; font-size: 13pt; color: #15130f; }}
          .score {{ float: right; font-size: 23pt; font-weight: 700; color: #e09830; }}
          .section {{ border-top: 1px solid #ded3c1; padding-top: 8pt; margin-top: 9pt; }}
          a {{ color: #7b4b12; }}
        </style>
        <body>
          <div class="hero">
            <h1>{_html_escape(title)}</h1>
            <p>{_html_escape(subtitle)}</p>
          </div>
          {body}
        </body>
        """
        page.insert_htmlbox(fitz.Rect(42, 38, 553, 806), content)

    ranking_items = []
    for index, item in enumerate(hypotheses, start=1):
        values = _export_hypothesis_values(item, weights)
        ranking_items.append(
            f"<li><b>{index}. {_html_escape(item.get('title') or 'Гипотеза')}</b> "
            f"<span class='muted'>score {values['score']:.1f}; "
            f"новизна {values['novelty']:.0f}; реализуемость {values['feasibility']:.0f}; "
            f"эффект {values['impact']:.0f}; риск {values['risk']:.0f}</span></li>"
        )
    add_html_page(
        str(project.get("name") or "IDEA"),
        "\n".join(
            [
                f"<p><b>Домен:</b> {_html_escape(project.get('domain') or '-')}</p>",
                f"<p><b>Цель / KPI:</b> {_html_escape(project.get('goal') or '-')}</p>",
                f"<p><b>Ограничения:</b> {_html_escape(project.get('constraints') or '-')}</p>",
                "<h3>Ранжирование</h3>",
                f"<ol>{''.join(ranking_items) if ranking_items else '<li>Гипотез пока нет</li>'}</ol>",
            ]
        ),
        subtitle=f"{len(documents)} источников · {len(hypotheses)} гипотез",
    )

    for index, item in enumerate(hypotheses, start=1):
        values = _export_hypothesis_values(item, weights)
        metrics = f"""
        <div class="metrics">
          <div class="metric"><span>Score</span><b>{values['score']:.1f}</b></div>
          <div class="metric"><span>Новизна</span><b>{values['novelty']:.0f}</b></div>
          <div class="metric"><span>Реализуемость</span><b>{values['feasibility']:.0f}</b></div>
          <div class="metric"><span>Эффект</span><b>{values['impact']:.0f}</b></div>
          <div class="metric"><span>Риск</span><b>{values['risk']:.0f}</b></div>
        </div>
        """
        body = "\n".join(
            [
                metrics,
                f"<p><b>Статус:</b> {_html_escape(item.get('status') or '-')}</p>",
                f"<div class='section'><h3>Проверяемое утверждение</h3><p>{_html_escape(_clip_text(item.get('statement'), 1400))}</p></div>",
                f"<div class='section'><h3>Обоснование</h3><p>{_html_escape(_clip_text(item.get('rationale'), 1800))}</p></div>",
                f"<div class='section'><h3>Механизм</h3><p>{_html_escape(_clip_text(item.get('mechanism'), 1400))}</p></div>",
                f"<div class='section'><h3>План внедрения / проверки</h3>{_pdf_roadmap_html(item.get('roadmap'))}</div>",
                f"<div class='section'><h3>Экономический контур</h3>{_pdf_economics_html(item.get('economics'))}</div>",
                f"<div class='section'><h3>Источники</h3>{_pdf_evidence_html(item.get('evidence'))}</div>",
            ]
        )
        add_html_page(f"{index}. {item.get('title') or 'Гипотеза'}", body, subtitle="Структурированный отчет по гипотезе")

    payload = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    return payload


def _export_weights(project: dict[str, Any]) -> dict[str, float]:
    settings_weights = project.get("settings") if isinstance(project, dict) else {}
    return {
        "novelty": float((settings_weights or {}).get("novelty", 0.25)),
        "feasibility": float((settings_weights or {}).get("feasibility", 0.25)),
        "impact": float((settings_weights or {}).get("impact", 0.35)),
        "risk": float((settings_weights or {}).get("risk", 0.15)),
    }


def _export_hypothesis_values(item: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
    novelty = _metric_for_export(item.get("novelty"))
    feasibility = _metric_for_export(item.get("feasibility"))
    impact = _metric_for_export(item.get("impact"))
    risk = _metric_for_export(item.get("risk"))
    raw_values = [float(item.get(key) or 0) for key in ("novelty", "feasibility", "impact", "risk")]
    looks_unit_scaled = any(0 < value <= 1 for value in raw_values) and all(value <= 1 for value in raw_values)
    score = (
        novelty * weights["novelty"]
        + feasibility * weights["feasibility"]
        + impact * weights["impact"]
        + (100 - risk) * weights["risk"]
        if looks_unit_scaled
        else float(item.get("score") or 0)
    )
    return {
        **item,
        "score": score,
        "novelty": novelty,
        "feasibility": feasibility,
        "impact": impact,
        "risk": risk,
    }


def _metric_for_export(value: Any) -> float:
    number = float(value or 0)
    return number * 100 if 0 < number <= 1 else number


def _evidence_url_text(item: dict[str, Any]) -> str:
    urls = [str(evidence.get("url")) for evidence in item.get("evidence") or [] if isinstance(evidence, dict) and evidence.get("url")]
    return " ".join(urls)


def _economics_export_text(economics: Any) -> str:
    if not isinstance(economics, list):
        return ""
    return "; ".join(_economics_line(item) for item in economics if isinstance(item, dict))


def _economics_line(item: dict[str, Any]) -> str:
    parts = [
        str(item.get("item") or "оценка"),
        str(item.get("assumption") or ""),
        str(item.get("calculation") or ""),
        str(item.get("expected_effect") or ""),
        str(item.get("data_needed") or ""),
    ]
    confidence = item.get("confidence")
    if confidence:
        parts.append(f"confidence={confidence}")
    return " | ".join(part for part in parts if part)


def _html_escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=True).replace("\n", "<br>")


def _clip_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _pdf_roadmap_html(roadmap: Any) -> str:
    if not isinstance(roadmap, list) or not roadmap:
        return "<p class='muted'>План не сформирован.</p>"
    items = []
    for index, step in enumerate(roadmap[:6], start=1):
        if not isinstance(step, dict):
            continue
        number = step.get("step") or index
        title = _html_escape(step.get("title") or "Шаг")
        output = _html_escape(_clip_text(step.get("output"), 420))
        owner = _html_escape(step.get("owner") or "")
        suffix = f" <span class='muted'>({owner})</span>" if owner else ""
        items.append(f"<li><b>{number}. {title}</b>{suffix}<br>{output}</li>")
    return f"<ol>{''.join(items)}</ol>" if items else "<p class='muted'>План не сформирован.</p>"


def _pdf_economics_html(economics: Any) -> str:
    if not isinstance(economics, list) or not economics:
        return "<p class='muted'>Экономический контур не сформирован.</p>"
    items = []
    for economic in economics[:5]:
        if not isinstance(economic, dict):
            continue
        title = _html_escape(economic.get("item") or "Оценка")
        details = [
            ("Допущение", economic.get("assumption")),
            ("Расчет", economic.get("calculation")),
            ("Эффект", economic.get("expected_effect")),
            ("Данные", economic.get("data_needed")),
            ("Доверие", economic.get("confidence")),
        ]
        text = "; ".join(f"{label}: {_clip_text(value, 320)}" for label, value in details if str(value or "").strip())
        items.append(f"<li><b>{title}</b><br>{_html_escape(text)}</li>")
    return f"<ul>{''.join(items)}</ul>" if items else "<p class='muted'>Экономический контур не сформирован.</p>"


def _pdf_evidence_html(evidence: Any) -> str:
    if not isinstance(evidence, list) or not evidence:
        return "<p class='muted'>Источники не указаны.</p>"
    items = []
    for item in evidence[:6]:
        if not isinstance(item, dict):
            continue
        source = _html_escape(item.get("source") or "source")
        quote = _html_escape(_clip_text(item.get("quote") or item.get("why"), 420))
        why = _html_escape(_clip_text(item.get("why"), 420))
        url = _html_escape(item.get("url") or "")
        link = f"<br><a href='{url}'>{url}</a>" if url else ""
        context = f"<br><span class='muted'>{why}</span>" if why and why != quote else ""
        items.append(f"<li><b>{source}</b><br>{quote}{context}{link}</li>")
    return f"<ul>{''.join(items)}</ul>" if items else "<p class='muted'>Источники не указаны.</p>"


def _pdf_hypothesis_text(item: dict[str, Any]) -> str:
    lines = [str(item.get("statement") or "")]
    economics = _economics_export_text(item.get("economics"))
    if economics:
        lines.append(f"Economics: {economics[:240]}")
    urls = _evidence_url_text(item)
    if urls:
        lines.append(f"Sources: {urls[:220]}")
    return "\n".join(line for line in lines if line)


def _md_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _ingest_document_bytes(
    project_id: str,
    filename: str,
    content_type: str,
    data: bytes,
    actor: str,
    ocr_languages: list[str] | None = None,
) -> dict[str, Any]:
    display_name = safe_filename(filename)
    selected_ocr_languages = ocr_languages or settings.pdf_ocr_languages
    started = time.perf_counter()
    logger.info(
        "Document parsing started: project_id=%s filename=%s bytes=%s ocr_langs=%s fast_mode=%s max_pages=%s image_max_pages=%s vision_max_pages=%s",
        project_id,
        display_name,
        len(data),
        ",".join(selected_ocr_languages),
        settings.pdf_fast_mode,
        settings.pdf_max_pages,
        settings.pdf_image_max_pages,
        settings.pdf_vision_max_pages,
    )
    parsed = parse_document(
        display_name,
        content_type,
        data,
        settings.max_document_chars,
        pdf_ocr_enabled=settings.pdf_ocr_enabled,
        pdf_ocr_languages=selected_ocr_languages,
        pdf_ocr_model_dir=settings.pdf_ocr_model_dir,
        pdf_fast_mode=settings.pdf_fast_mode,
        pdf_max_pages=settings.pdf_max_pages,
        pdf_image_max_pages=settings.pdf_image_max_pages,
        pdf_ocr_min_chars=settings.pdf_ocr_min_chars,
        pdf_text_layer_min_chars=settings.pdf_text_layer_min_chars,
        pdf_render_dpi=settings.pdf_render_dpi,
        pdf_vision_max_pages=settings.pdf_vision_max_pages,
    )
    logger.info(
        "Document parsing finished: project_id=%s filename=%s seconds=%.1f chars=%s meta=%s",
        project_id,
        display_name,
        time.perf_counter() - started,
        len(parsed.text),
        parsed.metadata.get("pdf_parse") or {},
    )
    vision_meta = _enrich_vision_images(parsed, selected_ocr_languages)
    upload_dir = settings.uploads_dir / project_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}_{display_name}"
    stored_path = upload_dir / stored_name
    stored_path.write_bytes(data)

    document = db.add_document(
        project_id=project_id,
        filename=display_name,
        content_type=content_type,
        text=parsed.text,
        metadata=parsed.metadata,
        path=str(stored_path),
        actor=actor,
    )
    nodes, edges = heuristic_graph_from_text(parsed.text, document["id"], document["filename"])
    llm_nodes, llm_edges, extractor_meta = ai.extract_graph(parsed.text, document["id"], document["filename"])
    graph_update = db.upsert_graph(
        project_id,
        [*nodes, *llm_nodes],
        [*edges, *llm_edges],
        actor,
        "document-ingested",
        source_id=document["id"],
    )
    return {"document": document, "graph_update": graph_update, "extractor": extractor_meta, "vision": vision_meta}


def _parse_prompt_attachment(
    filename: str,
    content_type: str,
    data: bytes,
    ocr_languages: list[str] | None = None,
) -> dict[str, Any]:
    display_name = safe_filename(filename)
    selected_ocr_languages = ocr_languages or settings.pdf_ocr_languages
    parsed = parse_document(
        display_name,
        content_type,
        data,
        settings.max_document_chars,
        pdf_ocr_enabled=settings.pdf_ocr_enabled,
        pdf_ocr_languages=selected_ocr_languages,
        pdf_ocr_model_dir=settings.pdf_ocr_model_dir,
        pdf_fast_mode=settings.pdf_fast_mode,
        pdf_max_pages=settings.pdf_max_pages,
        pdf_image_max_pages=settings.pdf_image_max_pages,
        pdf_ocr_min_chars=settings.pdf_ocr_min_chars,
        pdf_text_layer_min_chars=settings.pdf_text_layer_min_chars,
        pdf_render_dpi=settings.pdf_render_dpi,
        pdf_vision_max_pages=settings.pdf_vision_max_pages,
    )
    vision_meta = _enrich_vision_images(parsed, selected_ocr_languages)
    return {
        "id": f"prompt:{uuid.uuid4()}",
        "filename": f"[тестовый промпт] {display_name}",
        "content_type": content_type,
        "text": parsed.text,
        "metadata": {**parsed.metadata, "prompt_attachment": True, "vision": vision_meta},
        "path": None,
    }


def _enrich_vision_images(parsed: Any, ocr_languages: list[str]) -> dict[str, Any]:
    if not parsed.vision_images:
        return {"mode": "not_needed", "count": 0}

    items: list[dict[str, Any]] = []
    vision_chunks: list[str] = []

    def _describe(image: Any) -> tuple[str, dict[str, Any]]:
        return ai.describe_image(image.data, image.label, image.content_type, ocr_languages=ocr_languages)

    max_workers = max(1, min(settings.vision_max_workers, len(parsed.vision_images)))
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_describe, parsed.vision_images))
    else:
        results = [_describe(image) for image in parsed.vision_images]

    for image, (vision_text, vision_meta) in zip(parsed.vision_images, results):
        item_meta = {
            "label": image.label,
            "reason": image.reason,
            "page_no": image.page_no,
            **vision_meta,
        }
        items.append(item_meta)
        if vision_text:
            vision_chunks.append(f"[{image.label}]\n{vision_text}")
            logger.info("Vision extracted text: label=%s chars=%s", image.label, len(vision_text))
        elif vision_meta.get("mode") == "failed":
            logger.warning("Vision failed: label=%s reason=%s", image.label, vision_meta.get("reason"))

    vision_summary = {
        "mode": "processed",
        "count": len(parsed.vision_images),
        "text_items": len(vision_chunks),
        "items": items,
    }
    parsed.metadata["vision"] = vision_summary
    if vision_chunks:
        separator = "\n\n" if parsed.text else ""
        combined = f"{parsed.text}{separator}Vision-анализ OpenRouter:\n" + "\n\n".join(vision_chunks)
        if len(combined) > settings.max_document_chars:
            parsed.metadata["truncated"] = True
            parsed.metadata["original_chars"] = len(combined)
            combined = combined[: settings.max_document_chars]
        parsed.text = combined
        parsed.metadata["chars"] = len(parsed.text)
    return vision_summary


def _request_ocr_languages(value: str | list[str] | None) -> list[str] | None:
    languages = normalize_ocr_languages(value, default=[])
    return languages or None


def _ordered_sample_paths(paths: list[Path]) -> list[Path]:
    buckets: dict[str, list[Path]] = {
        "regulation_images": [],
        "scheme_images": [],
        "core_documents": [],
        "technical_pdfs": [],
        "other": [],
    }
    for path in paths:
        buckets[_sample_bucket(path)].append(path)

    for bucket_paths in buckets.values():
        bucket_paths.sort(key=_sample_path_key)

    ordered: list[Path] = []
    cycle = ("regulation_images", "scheme_images", "core_documents", "regulation_images", "scheme_images", "core_documents", "technical_pdfs")
    while any(buckets.values()):
        before = len(ordered)
        for bucket in cycle:
            if buckets[bucket]:
                ordered.append(buckets[bucket].pop(0))
        if len(ordered) == before:
            ordered.extend(buckets["other"])
            buckets["other"] = []
    return ordered


def _sample_bucket(path: Path) -> str:
    path_key = _sample_path_key(path)
    extension = path.suffix.lower()
    if extension in IMAGE_EXTENSIONS and "регламенты/" in path_key:
        return "regulation_images"
    if extension in IMAGE_EXTENSIONS and "схемы флотации/" in path_key:
        return "scheme_images"
    if extension in {".docx", ".xlsx", ".xlsm", ".txt", ".md", ".csv"}:
        return "core_documents"
    if extension == ".pdf" and "дополнительные материалы/" not in path_key:
        return "technical_pdfs"
    return "other"


def _sample_path_key(path: Path) -> str:
    try:
        relative = path.relative_to(settings.sample_data_dir)
    except ValueError:
        relative = path
    return "/".join(part.casefold() for part in relative.parts)


def _require_project(project_id: str) -> dict[str, Any]:
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден")
    return project


def _actor(value: str | None) -> str:
    actor = unquote(value or "User").strip()
    return actor[:80] or "User"
