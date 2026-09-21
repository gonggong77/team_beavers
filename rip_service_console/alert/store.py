"""이벤트 저장소.

알림 발송 여부와 무관하게 모든 이벤트를 남긴다.
덕분에 발송이 실패해도 이력이 남고, 재전송과 디버깅이 쉬워진다.

원본 core/store.py 와 스키마가 같다. DB 경로만 이 폴더 기준(alert_only/data/events.db)이다.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from core.schemas import AlertEvent

# 프로젝트 루트 기준 data/events.db 에 저장되도록 설정
DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "events.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id      TEXT PRIMARY KEY,
    occurred_at   TEXT NOT NULL,
    risk_level    TEXT NOT NULL,
    persons_in_rip INTEGER NOT NULL,
    total_persons INTEGER NOT NULL,
    rip_count     INTEGER NOT NULL,
    frame_idx     INTEGER NOT NULL,
    timestamp_sec REAL NOT NULL,
    video_source  TEXT NOT NULL,
    snapshot_path TEXT,
    note          TEXT,
    notify_status TEXT DEFAULT 'pending',
    notify_detail TEXT DEFAULT '',
    trigger_kind  TEXT DEFAULT 'auto',
    image_url     TEXT DEFAULT '',
    handled       INTEGER DEFAULT 0
);
"""

# CREATE TABLE IF NOT EXISTS 는 기존 DB에 새 컬럼을 추가해 주지 않는다.
# 이미 events.db 를 가진 사용자를 위한 수동 마이그레이션.
_MIGRATION_COLUMNS = (
    ("trigger_kind", "TEXT DEFAULT 'auto'"),
    ("image_url", "TEXT DEFAULT ''"),
    ("handled", "INTEGER DEFAULT 0"),
)


class EventStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            have = {row["name"] for row in conn.execute("PRAGMA table_info(events)")}
            for column, ddl in _MIGRATION_COLUMNS:
                if column not in have:
                    conn.execute(f"ALTER TABLE events ADD COLUMN {column} {ddl}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, event: AlertEvent, notify_status: str = "pending", notify_detail: str = "") -> None:
        payload = event.to_dict()
        payload["notify_status"] = notify_status
        payload["notify_detail"] = notify_detail
        columns = ", ".join(payload)
        placeholders = ", ".join(f":{k}" for k in payload)
        with self._connect() as conn:
            conn.execute(f"INSERT OR REPLACE INTO events ({columns}) VALUES ({placeholders})", payload)

    def update_notify(self, event_id: str, status: str, detail: str = "", image_url: str = "") -> None:
        with self._connect() as conn:
            if image_url:
                conn.execute(
                    "UPDATE events SET notify_status = ?, notify_detail = ?, image_url = ? WHERE event_id = ?",
                    (status, detail, image_url, event_id),
                )
                return
            conn.execute(
                "UPDATE events SET notify_status = ?, notify_detail = ? WHERE event_id = ?",
                (status, detail, event_id),
            )

    def mark_handled(self, event_id: str) -> None:
        """관리자가 확인 처리한 이벤트를 표시한다."""
        with self._connect() as conn:
            conn.execute("UPDATE events SET handled = 1 WHERE event_id = ?", (event_id,))

    def list_events(self, limit: int = 500) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY occurred_at DESC, frame_idx DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM events")

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(row, ensure_ascii=False) for row in self.list_events())
