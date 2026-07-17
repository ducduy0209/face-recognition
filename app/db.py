import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EMBEDDING_DTYPE = np.float32

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    email      TEXT NOT NULL,
    phone      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (name, email, phone)
);
CREATE TABLE IF NOT EXISTS faces (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    embedding  BLOB NOT NULL,
    image_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class DuplicateUserError(Exception):
    pass


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA foreign_keys = ON")  # executescript resets pragma state
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_user(conn: sqlite3.Connection, name: str, email: str, phone: str) -> int:
    try:
        cur = conn.execute(
            "INSERT INTO users (name, email, phone, created_at) VALUES (?, ?, ?, ?)",
            (name, email, phone, _now()),
        )
    except sqlite3.IntegrityError as exc:
        raise DuplicateUserError(f"user ({name}, {email}, {phone}) already exists") from exc
    conn.commit()
    return cur.lastrowid


def get_user(conn: sqlite3.Connection, user_id: int) -> dict | None:
    row = conn.execute(
        "SELECT id, name, email, phone FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    return dict(row) if row else None


def list_users(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT u.id, u.name, u.email, u.phone, u.created_at, COUNT(f.id) AS face_count
        FROM users u LEFT JOIN faces f ON f.user_id = u.id
        GROUP BY u.id ORDER BY u.id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def delete_user(conn: sqlite3.Connection, user_id: int) -> list[str] | None:
    if get_user(conn, user_id) is None:
        return None
    paths = [
        r["image_path"]
        for r in conn.execute(
            "SELECT image_path FROM faces WHERE user_id = ?", (user_id,)
        ).fetchall()
    ]
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    return paths


def add_face(conn: sqlite3.Connection, user_id: int, embedding: np.ndarray, image_path: str) -> int:
    if embedding.size != 512:
        raise ValueError(f"embedding must have exactly 512 elements, got {embedding.size}")
    blob = embedding.astype(EMBEDDING_DTYPE).tobytes()
    cur = conn.execute(
        "INSERT INTO faces (user_id, embedding, image_path, created_at) VALUES (?, ?, ?, ?)",
        (user_id, blob, image_path, _now()),
    )
    conn.commit()
    return cur.lastrowid


def delete_faces(conn: sqlite3.Connection, user_id: int) -> list[str]:
    paths = [
        r["image_path"]
        for r in conn.execute(
            "SELECT image_path FROM faces WHERE user_id = ?", (user_id,)
        ).fetchall()
    ]
    conn.execute("DELETE FROM faces WHERE user_id = ?", (user_id,))
    conn.commit()
    return paths


def count_faces(conn: sqlite3.Connection, user_id: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS n FROM faces WHERE user_id = ?", (user_id,)
    ).fetchone()["n"]


def load_all_faces(conn: sqlite3.Connection) -> list[tuple[int, int, np.ndarray]]:
    rows = conn.execute("SELECT id, user_id, embedding FROM faces").fetchall()
    return [
        (r["id"], r["user_id"], np.frombuffer(r["embedding"], dtype=EMBEDDING_DTYPE))
        for r in rows
    ]
