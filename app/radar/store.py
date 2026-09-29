from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RadarStore:
    """PostgreSQL for the stand; explicit SQLite fallback for local development/tests."""

    def __init__(self, database_url: str, storage_dir: Path):
        self.url = database_url
        self.path = storage_dir / "radar.db"
        self.backend = "postgresql" if database_url else "sqlite"
        if database_url and not database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("RADAR_DATABASE_URL must be a PostgreSQL URL")

    @contextmanager
    def connection(self):
        if self.url:
            import psycopg
            from psycopg.rows import dict_row
            conn = psycopg.connect(self.url, row_factory=dict_row, connect_timeout=5)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=15)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, conn, sql: str, args=()):
        return conn.execute(sql.replace("?", "%s") if self.url else sql, args)

    def initialize(self):
        data_type = "JSONB" if self.url else "TEXT"
        with self.connection() as conn:
            conn.execute(f"""CREATE TABLE IF NOT EXISTS radar_runs (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                status TEXT NOT NULL, payload {data_type} NOT NULL)""")
            conn.execute(f"""CREATE TABLE IF NOT EXISTS radar_sources (
                run_id TEXT NOT NULL REFERENCES radar_runs(id) ON DELETE CASCADE,
                id TEXT NOT NULL, payload {data_type} NOT NULL, PRIMARY KEY (run_id,id))""")
            conn.execute("CREATE INDEX IF NOT EXISTS radar_runs_created ON radar_runs(created_at)")

    def _encode(self, value):
        if self.url:
            from psycopg.types.json import Jsonb
            return Jsonb(value)
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _decode(value):
        return json.loads(value) if isinstance(value, str) else value

    def save(self, run: dict[str, Any]):
        run["updated_at"] = now()
        with self.connection() as conn:
            self.execute(conn, """INSERT INTO radar_runs (id,created_at,updated_at,status,payload)
                VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                updated_at=excluded.updated_at,status=excluded.status,payload=excluded.payload""",
                (run["id"], run["created_at"], run["updated_at"], run["status"], self._encode(run)))

    def get(self, run_id: str):
        with self.connection() as conn:
            row = self.execute(conn, "SELECT payload FROM radar_runs WHERE id=?", (run_id,)).fetchone()
        return self._decode(row["payload"]) if row else None

    def list_runs(self, limit=30):
        with self.connection() as conn:
            rows = self.execute(conn, "SELECT payload FROM radar_runs ORDER BY created_at DESC,id DESC LIMIT ?", (limit,)).fetchall()
        return [self._decode(row["payload"]) for row in rows]

    def delete_run(self, run_id: str) -> bool:
        """Delete one saved run and its source records."""
        with self.connection() as conn:
            cursor = self.execute(conn, "DELETE FROM radar_runs WHERE id=?", (run_id,))
            return cursor.rowcount > 0

    def put_source(self, run_id: str, source: dict):
        with self.connection() as conn:
            self.execute(conn, """INSERT INTO radar_sources (run_id,id,payload) VALUES (?,?,?)
                ON CONFLICT(run_id,id) DO UPDATE SET payload=excluded.payload""",
                (run_id, source["id"], self._encode(source)))

    def source(self, run_id: str, source_id: str):
        with self.connection() as conn:
            row = self.execute(conn, "SELECT payload FROM radar_sources WHERE run_id=? AND id=?", (run_id, source_id)).fetchone()
        return self._decode(row["payload"]) if row else None

    def recover(self):
        with self.connection() as conn:
            rows = self.execute(conn, "SELECT payload FROM radar_runs WHERE status IN ('queued','running')").fetchall()
        for row in rows:
            run = self._decode(row["payload"])
            run.update(status="interrupted", stage="Выполнение прервано перезапуском", finished_at=now())
            run["warnings"].append("Сервер был перезапущен. Можно повторить запрос; предыдущие данные сохранены.")
            self.save(run)
