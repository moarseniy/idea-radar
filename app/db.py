from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.knowledge import is_low_signal_numeric_node


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    goal TEXT NOT NULL DEFAULT '',
                    constraints TEXT NOT NULL DEFAULT '',
                    domain TEXT NOT NULL DEFAULT '',
                    team_json TEXT NOT NULL DEFAULT '[]',
                    settings_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    meta_json TEXT NOT NULL DEFAULT '{}',
                    path TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS knowledge_nodes (
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    node_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    type TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    weight REAL NOT NULL DEFAULT 1,
                    source_ids_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (project_id, node_id)
                );

                CREATE TABLE IF NOT EXISTS knowledge_edges (
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    edge_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    evidence TEXT NOT NULL DEFAULT '',
                    weight REAL NOT NULL DEFAULT 1,
                    source_ids_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (project_id, edge_id)
                );

                CREATE TABLE IF NOT EXISTS hypotheses (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    rationale TEXT NOT NULL DEFAULT '',
                    mechanism TEXT NOT NULL DEFAULT '',
                    novelty REAL NOT NULL DEFAULT 50,
                    feasibility REAL NOT NULL DEFAULT 50,
                    impact REAL NOT NULL DEFAULT 50,
                    risk REAL NOT NULL DEFAULT 50,
                    uncertainty TEXT NOT NULL DEFAULT '',
                    score REAL NOT NULL DEFAULT 50,
                    status TEXT NOT NULL DEFAULT 'draft',
                    evidence_json TEXT NOT NULL DEFAULT '[]',
                    roadmap_json TEXT NOT NULL DEFAULT '[]',
                    economics_json TEXT NOT NULL DEFAULT '[]',
                    version_no INTEGER NOT NULL DEFAULT 1,
                    parent_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    hypothesis_id TEXT REFERENCES hypotheses(id) ON DELETE SET NULL,
                    actor TEXT NOT NULL,
                    rating INTEGER,
                    outcome TEXT,
                    comment TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    version_no INTEGER NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id);
                CREATE INDEX IF NOT EXISTS idx_hypotheses_project ON hypotheses(project_id);
                CREATE INDEX IF NOT EXISTS idx_events_project ON events(project_id, version_no);
                CREATE INDEX IF NOT EXISTS idx_chat_project ON chat_messages(project_id, created_at);
                """
            )
            self._ensure_column(conn, "hypotheses", "economics_json", "TEXT NOT NULL DEFAULT '[]'")

    def create_project(self, payload: dict[str, Any], actor: str) -> dict[str, Any]:
        project_id = str(uuid.uuid4())
        now = utcnow()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO projects (id, name, goal, constraints, domain, team_json, settings_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    payload.get("name") or "Новый исследовательский проект",
                    payload.get("goal") or "",
                    payload.get("constraints") or "",
                    payload.get("domain") or "Обогащение и металлургия",
                    json_dumps(payload.get("team") or []),
                    json_dumps(payload.get("settings") or _default_project_settings()),
                    now,
                    now,
                ),
            )
            self._add_event_conn(conn, project_id, actor, "project.created", "project", project_id, payload)
        return self.get_project(project_id) or {}

    def update_project(self, project_id: str, payload: dict[str, Any], actor: str) -> dict[str, Any] | None:
        allowed = {"name", "goal", "constraints", "domain"}
        updates = {key: value for key, value in payload.items() if key in allowed and value is not None}
        now = utcnow()
        with self.connect() as conn:
            if updates:
                assignments = ", ".join(f"{key}=?" for key in updates)
                conn.execute(
                    f"UPDATE projects SET {assignments}, updated_at=? WHERE id=?",
                    (*updates.values(), now, project_id),
                )
            if payload.get("team") is not None:
                conn.execute(
                    "UPDATE projects SET team_json=?, updated_at=? WHERE id=?",
                    (json_dumps(payload["team"]), now, project_id),
                )
            if payload.get("settings") is not None:
                current = self._get_project_conn(conn, project_id)
                settings = dict(json_loads(current["settings_json"] if current else None, {}))
                settings.update(payload["settings"])
                conn.execute(
                    "UPDATE projects SET settings_json=?, updated_at=? WHERE id=?",
                    (json_dumps(settings), now, project_id),
                )
            self._add_event_conn(conn, project_id, actor, "project.updated", "project", project_id, payload)
        return self.get_project(project_id)

    def delete_project(self, project_id: str) -> bool:
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
        return cursor.rowcount > 0

    def list_projects(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT p.*,
                       (SELECT COUNT(*) FROM documents d WHERE d.project_id = p.id) AS document_count,
                       (SELECT COUNT(*) FROM hypotheses h WHERE h.project_id = p.id) AS hypothesis_count
                FROM projects p
                ORDER BY p.updated_at DESC
                """
            ).fetchall()
        return [self._project_from_row(row) for row in rows]

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = self._get_project_conn(conn, project_id)
        return self._project_from_row(row) if row else None

    def add_document(
        self,
        project_id: str,
        filename: str,
        content_type: str,
        text: str,
        metadata: dict[str, Any],
        path: str,
        actor: str,
    ) -> dict[str, Any]:
        doc_id = str(uuid.uuid4())
        now = utcnow()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, project_id, filename, content_type, text, meta_json, path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (doc_id, project_id, filename, content_type, text, json_dumps(metadata), path, now),
            )
            conn.execute("UPDATE projects SET updated_at=? WHERE id=?", (now, project_id))
            self._add_event_conn(
                conn,
                project_id,
                actor,
                "document.uploaded",
                "document",
                doc_id,
                {"filename": filename, "chars": len(text), "metadata": metadata},
            )
        return self.get_document(doc_id) or {}

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        return self._document_from_row(row) if row else None

    def list_documents(self, project_id: str, include_text: bool = False) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE project_id=? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        docs = [self._document_from_row(row) for row in rows]
        if not include_text:
            for doc in docs:
                doc.pop("text", None)
        return docs

    def upsert_graph(
        self,
        project_id: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        actor: str,
        reason: str,
        source_id: str | None = None,
    ) -> dict[str, int]:
        now = utcnow()
        skipped_node_ids = {str(node.get("id") or "") for node in nodes if is_low_signal_numeric_node(node)}
        nodes = [node for node in nodes if str(node.get("id") or "") not in skipped_node_ids]
        edges = [
            edge
            for edge in edges
            if str(edge.get("source") or "") not in skipped_node_ids and str(edge.get("target") or "") not in skipped_node_ids
        ]
        inserted_nodes = 0
        inserted_edges = 0
        with self.connect() as conn:
            for node in nodes:
                node_id = str(node.get("id") or "")
                if not node_id:
                    continue
                source_ids = _merge_source_ids(node.get("source_ids"), source_id)
                row = conn.execute(
                    "SELECT weight, source_ids_json FROM knowledge_nodes WHERE project_id=? AND node_id=?",
                    (project_id, node_id),
                ).fetchone()
                if row:
                    merged_sources = sorted(set(json_loads(row["source_ids_json"], []) + source_ids))
                    conn.execute(
                        """
                        UPDATE knowledge_nodes
                        SET label=?, type=?, summary=?, weight=?, source_ids_json=?, updated_at=?
                        WHERE project_id=? AND node_id=?
                        """,
                        (
                            str(node.get("label") or node_id),
                            str(node.get("type") or "concept"),
                            str(node.get("summary") or ""),
                            max(float(row["weight"]), float(node.get("weight") or 1)),
                            json_dumps(merged_sources),
                            now,
                            project_id,
                            node_id,
                        ),
                    )
                else:
                    inserted_nodes += 1
                    conn.execute(
                        """
                        INSERT INTO knowledge_nodes
                        (project_id, node_id, label, type, summary, weight, source_ids_json, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            project_id,
                            node_id,
                            str(node.get("label") or node_id),
                            str(node.get("type") or "concept"),
                            str(node.get("summary") or ""),
                            float(node.get("weight") or 1),
                            json_dumps(source_ids),
                            now,
                        ),
                    )
            for edge in edges:
                edge_id = str(edge.get("id") or "")
                source = str(edge.get("source") or "")
                target = str(edge.get("target") or "")
                if not edge_id or not source or not target:
                    continue
                source_ids = _merge_source_ids(edge.get("source_ids"), source_id)
                row = conn.execute(
                    "SELECT weight, source_ids_json FROM knowledge_edges WHERE project_id=? AND edge_id=?",
                    (project_id, edge_id),
                ).fetchone()
                if row:
                    merged_sources = sorted(set(json_loads(row["source_ids_json"], []) + source_ids))
                    conn.execute(
                        """
                        UPDATE knowledge_edges
                        SET source=?, target=?, relation=?, evidence=?, weight=?, source_ids_json=?, updated_at=?
                        WHERE project_id=? AND edge_id=?
                        """,
                        (
                            source,
                            target,
                            str(edge.get("relation") or "related_to"),
                            str(edge.get("evidence") or ""),
                            max(float(row["weight"]), float(edge.get("weight") or 1)),
                            json_dumps(merged_sources),
                            now,
                            project_id,
                            edge_id,
                        ),
                    )
                else:
                    inserted_edges += 1
                    conn.execute(
                        """
                        INSERT INTO knowledge_edges
                        (project_id, edge_id, source, target, relation, evidence, weight, source_ids_json, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            project_id,
                            edge_id,
                            source,
                            target,
                            str(edge.get("relation") or "related_to"),
                            str(edge.get("evidence") or ""),
                            float(edge.get("weight") or 1),
                            json_dumps(source_ids),
                            now,
                        ),
                    )
            if nodes or edges:
                self._add_event_conn(
                    conn,
                    project_id,
                    actor,
                    "graph.updated",
                    "graph",
                    source_id,
                    {"reason": reason, "nodes": len(nodes), "edges": len(edges)},
                )
        return {"inserted_nodes": inserted_nodes, "inserted_edges": inserted_edges}

    def get_graph(self, project_id: str) -> dict[str, list[dict[str, Any]]]:
        with self.connect() as conn:
            node_rows = conn.execute(
                "SELECT * FROM knowledge_nodes WHERE project_id=? ORDER BY weight DESC, label LIMIT 320",
                (project_id,),
            ).fetchall()
            edge_rows = conn.execute(
                "SELECT * FROM knowledge_edges WHERE project_id=? ORDER BY weight DESC LIMIT 720",
                (project_id,),
            ).fetchall()
        nodes = [self._node_from_row(row) for row in node_rows]
        nodes = [node for node in nodes if not is_low_signal_numeric_node(node)][:180]
        node_ids = {str(node.get("id")) for node in nodes}
        edges = [
            edge
            for edge in (self._edge_from_row(row) for row in edge_rows)
            if str(edge.get("source")) in node_ids and str(edge.get("target")) in node_ids
        ][:360]
        return {
            "nodes": nodes,
            "edges": edges,
        }

    def add_hypotheses(self, project_id: str, hypotheses: list[dict[str, Any]], actor: str) -> list[dict[str, Any]]:
        now = utcnow()
        created: list[dict[str, Any]] = []
        with self.connect() as conn:
            for item in hypotheses:
                hypothesis_id = str(uuid.uuid4())
                version_no = self._next_version_conn(conn, project_id)
                conn.execute(
                    """
                    INSERT INTO hypotheses (
                        id, project_id, title, statement, rationale, mechanism, novelty, feasibility,
                        impact, risk, uncertainty, score, status, evidence_json, roadmap_json, economics_json,
                        version_no, parent_id, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        hypothesis_id,
                        project_id,
                        str(item.get("title") or "Гипотеза"),
                        str(item.get("statement") or ""),
                        str(item.get("rationale") or ""),
                        str(item.get("mechanism") or ""),
                        float(item["novelty"] if item.get("novelty") is not None else 50),
                        float(item["feasibility"] if item.get("feasibility") is not None else 50),
                        float(item["impact"] if item.get("impact") is not None else 50),
                        float(item["risk"] if item.get("risk") is not None else 50),
                        str(item.get("uncertainty") or ""),
                        float(item["score"] if item.get("score") is not None else 50),
                        str(item.get("status") or "draft"),
                        json_dumps(item.get("evidence") or []),
                        json_dumps(item.get("roadmap") or []),
                        json_dumps(item.get("economics") or []),
                        version_no,
                        item.get("parent_id"),
                        now,
                        now,
                    ),
                )
                self._add_event_conn(
                    conn,
                    project_id,
                    actor,
                    "hypothesis.created",
                    "hypothesis",
                    hypothesis_id,
                    {"title": item.get("title"), "score": item.get("score"), "version_no": version_no},
                    version_no=version_no,
                )
                row = conn.execute("SELECT * FROM hypotheses WHERE id=?", (hypothesis_id,)).fetchone()
                created.append(self._hypothesis_from_row(row))
            conn.execute("UPDATE projects SET updated_at=? WHERE id=?", (now, project_id))
        return created

    def list_hypotheses(self, project_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hypotheses WHERE project_id=? ORDER BY score DESC, created_at DESC",
                (project_id,),
            ).fetchall()
        return [self._hypothesis_from_row(row) for row in rows]

    def get_hypothesis(self, hypothesis_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM hypotheses WHERE id=?", (hypothesis_id,)).fetchone()
        return self._hypothesis_from_row(row) if row else None

    def clear_hypotheses(self, project_id: str, actor: str) -> dict[str, int]:
        """Delete all hypotheses of a project and roll back their contribution to the graph.

        Nodes/edges whose only source is a hypothesis are removed; those also backed by a
        document keep their remaining sources so the graph returns to its pre-generation state.
        """
        removed_nodes = 0
        removed_edges = 0
        with self.connect() as conn:
            hyp_ids = {
                str(row["id"])
                for row in conn.execute("SELECT id FROM hypotheses WHERE project_id=?", (project_id,)).fetchall()
            }
            if not hyp_ids:
                return {"hypotheses": 0, "nodes": 0, "edges": 0}

            now = utcnow()
            for table, id_column in (("knowledge_nodes", "node_id"), ("knowledge_edges", "edge_id")):
                rows = conn.execute(
                    f"SELECT {id_column} AS entity_id, source_ids_json FROM {table} WHERE project_id=?",
                    (project_id,),
                ).fetchall()
                for row in rows:
                    source_ids = [str(item) for item in json_loads(row["source_ids_json"], []) if item]
                    remaining = [source for source in source_ids if source not in hyp_ids]
                    if len(remaining) == len(source_ids):
                        continue
                    if remaining:
                        conn.execute(
                            f"UPDATE {table} SET source_ids_json=?, updated_at=? WHERE project_id=? AND {id_column}=?",
                            (json_dumps(sorted(set(remaining))), now, project_id, row["entity_id"]),
                        )
                    else:
                        conn.execute(
                            f"DELETE FROM {table} WHERE project_id=? AND {id_column}=?",
                            (project_id, row["entity_id"]),
                        )
                        if table == "knowledge_nodes":
                            removed_nodes += 1
                        else:
                            removed_edges += 1

            removed_hypotheses = conn.execute(
                "DELETE FROM hypotheses WHERE project_id=?", (project_id,)
            ).rowcount
            conn.execute("UPDATE projects SET updated_at=? WHERE id=?", (now, project_id))
            self._add_event_conn(
                conn,
                project_id,
                actor,
                "hypotheses.cleared",
                "hypothesis",
                None,
                {"hypotheses": removed_hypotheses, "nodes": removed_nodes, "edges": removed_edges},
            )
        return {"hypotheses": removed_hypotheses, "nodes": removed_nodes, "edges": removed_edges}

    def update_hypothesis_status(self, project_id: str, hypothesis_id: str, status: str, actor: str) -> dict[str, Any] | None:
        now = utcnow()
        with self.connect() as conn:
            version_no = self._next_version_conn(conn, project_id)
            conn.execute(
                "UPDATE hypotheses SET status=?, updated_at=?, version_no=? WHERE id=? AND project_id=?",
                (status, now, version_no, hypothesis_id, project_id),
            )
            self._add_event_conn(
                conn,
                project_id,
                actor,
                "hypothesis.status_changed",
                "hypothesis",
                hypothesis_id,
                {"status": status},
                version_no=version_no,
            )
            conn.execute("UPDATE projects SET updated_at=? WHERE id=?", (now, project_id))
        return self.get_hypothesis(hypothesis_id)

    def add_feedback(
        self,
        project_id: str,
        hypothesis_id: str | None,
        actor: str,
        rating: int | None,
        outcome: str | None,
        comment: str,
    ) -> dict[str, Any]:
        feedback_id = str(uuid.uuid4())
        now = utcnow()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback (id, project_id, hypothesis_id, actor, rating, outcome, comment, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (feedback_id, project_id, hypothesis_id, actor, rating, outcome, comment, now),
            )
            if outcome and hypothesis_id:
                status = _status_from_outcome(outcome)
                if status:
                    conn.execute(
                        "UPDATE hypotheses SET status=?, updated_at=? WHERE id=? AND project_id=?",
                        (status, now, hypothesis_id, project_id),
                    )
            self._add_event_conn(
                conn,
                project_id,
                actor,
                "feedback.added",
                "hypothesis" if hypothesis_id else "project",
                hypothesis_id,
                {"rating": rating, "outcome": outcome, "comment": comment},
            )
            conn.execute("UPDATE projects SET updated_at=? WHERE id=?", (now, project_id))
        return self.get_feedback(feedback_id) or {}

    def get_feedback(self, feedback_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT f.*, f.rowid AS feedback_rowid, h.title AS hypothesis_title
                FROM feedback f
                LEFT JOIN hypotheses h ON h.id = f.hypothesis_id
                WHERE f.id=?
                """,
                (feedback_id,),
            ).fetchone()
        return self._feedback_from_row(row) if row else None

    def list_feedback(self, project_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT f.*, f.rowid AS feedback_rowid, h.title AS hypothesis_title
                FROM feedback f
                LEFT JOIN hypotheses h ON h.id = f.hypothesis_id
                WHERE f.project_id=?
                ORDER BY f.created_at DESC, f.rowid DESC
                LIMIT 100
                """,
                (project_id,),
            ).fetchall()
        return [self._feedback_from_row(row) for row in rows]

    def add_chat_message(self, project_id: str, role: str, actor: str, content: str, event: bool = False) -> dict[str, Any]:
        message_id = str(uuid.uuid4())
        now = utcnow()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (id, project_id, role, actor, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id, project_id, role, actor, content, now),
            )
            if event:
                self._add_event_conn(
                    conn,
                    project_id,
                    actor,
                    "chat.correction",
                    "chat",
                    message_id,
                    {"role": role, "content": content[:700]},
                )
        return self.get_chat_message(message_id) or {}

    def get_chat_message(self, message_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM chat_messages WHERE id=?", (message_id,)).fetchone()
        return self._chat_from_row(row) if row else None

    def list_chat(self, project_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE project_id=? ORDER BY created_at ASC LIMIT 120",
                (project_id,),
            ).fetchall()
        return [self._chat_from_row(row) for row in rows]

    def list_events(self, project_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE project_id=? ORDER BY version_no DESC, created_at DESC LIMIT 140",
                (project_id,),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def export_project(self, project_id: str) -> dict[str, Any]:
        return {
            "project": self.get_project(project_id),
            "documents": self.list_documents(project_id, include_text=False),
            "hypotheses": self.list_hypotheses(project_id),
            "graph": self.get_graph(project_id),
            "feedback": self.list_feedback(project_id),
            "events": self.list_events(project_id),
        }

    def _get_project_conn(self, conn: sqlite3.Connection, project_id: str) -> sqlite3.Row | None:
        return conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()

    def _next_version_conn(self, conn: sqlite3.Connection, project_id: str) -> int:
        row = conn.execute("SELECT COALESCE(MAX(version_no), 0) + 1 AS next_no FROM events WHERE project_id=?", (project_id,)).fetchone()
        return int(row["next_no"])

    def _add_event_conn(
        self,
        conn: sqlite3.Connection,
        project_id: str,
        actor: str,
        action: str,
        entity_type: str,
        entity_id: str | None,
        payload: dict[str, Any],
        version_no: int | None = None,
    ) -> None:
        event_id = str(uuid.uuid4())
        version = version_no or self._next_version_conn(conn, project_id)
        conn.execute(
            """
            INSERT INTO events (id, project_id, version_no, actor, action, entity_type, entity_id, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, project_id, version, actor, action, entity_type, entity_id, json_dumps(payload), utcnow()),
        )

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _project_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["team"] = json_loads(data.pop("team_json"), [])
        settings = json_loads(data.pop("settings_json"), {})
        data["settings"] = {**_default_project_settings(), **settings}
        data["document_count"] = int(data.get("document_count") or 0)
        data["hypothesis_count"] = int(data.get("hypothesis_count") or 0)
        return data

    def _document_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["metadata"] = json_loads(data.pop("meta_json"), {})
        return data

    def _node_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["id"] = data.pop("node_id")
        data["source_ids"] = json_loads(data.pop("source_ids_json"), [])
        data.pop("project_id", None)
        return data

    def _edge_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["id"] = data.pop("edge_id")
        data["source_ids"] = json_loads(data.pop("source_ids_json"), [])
        data.pop("project_id", None)
        return data

    def _hypothesis_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["evidence"] = json_loads(data.pop("evidence_json"), [])
        data["roadmap"] = json_loads(data.pop("roadmap_json"), [])
        data["economics"] = json_loads(data.pop("economics_json", "[]"), [])
        return data

    def _feedback_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return dict(row)

    def _chat_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return dict(row)

    def _event_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["payload"] = json_loads(data.pop("payload_json"), {})
        return data


def _merge_source_ids(value: Any, extra_source_id: str | None) -> list[str]:
    if isinstance(value, list):
        source_ids = [str(item) for item in value if item]
    elif value:
        source_ids = [str(value)]
    else:
        source_ids = []
    if extra_source_id:
        source_ids.append(extra_source_id)
    return sorted(set(source_ids))


def _status_from_outcome(outcome: str) -> str:
    normalized = outcome.lower().strip()
    if normalized in {"liked", "like", "disliked", "dislike", "neutral", "reaction_removed"}:
        return ""
    if normalized in {"confirmed", "подтверждена", "подтверждено", "success"}:
        return "confirmed"
    if normalized in {"rejected", "опровергнута", "опровергнуто", "failed"}:
        return "rejected"
    if normalized in {"experiment", "в эксперимент", "testing"}:
        return "experiment"
    return "review"


def _default_project_settings() -> dict[str, float]:
    return {"novelty": 0.25, "feasibility": 0.25, "impact": 0.35, "risk": 0.15}
