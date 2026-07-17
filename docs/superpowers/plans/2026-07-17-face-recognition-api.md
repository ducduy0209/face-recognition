# Face Recognition API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FastAPI face recognition service for a check-in system: enroll a photo + user info, recognize a photo and return the matching user.

**Architecture:** A single FastAPI service. InsightFace (`buffalo_l`, ONNX CPU) produces 512-d embeddings; the matcher keeps all embeddings in RAM and does brute-force cosine comparison; SQLite stores users + embeddings; original images are stored as files. The app factory `create_app(settings, engine)` allows testing with a fake engine.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, InsightFace + onnxruntime (CPU), OpenCV headless, numpy, SQLite (stdlib `sqlite3`), pytest, Docker.

**Spec:** `docs/superpowers/specs/2026-07-17-face-recognition-api-design.md`

## Global Constraints

- CPU-only: ONNX provider is `CPUExecutionProvider`, `ctx_id=-1`.
- The trio `(name, email, phone)` is UNIQUE in the `users` table.
- Every endpoint (except `GET /health`) requires the `X-API-Key` header matching env `API_KEY`; wrong/missing → 401.
- `POST /recognize` always returns HTTP 200 when the image decodes (even on no match/no face); 422 only for a corrupt image.
- Consistent error body: `{"detail": "...", "reason": "..."}`.
- Env: `API_KEY` (required), `MATCH_THRESHOLD` (default `0.40`), `MODEL_NAME` (default `buffalo_l`), `DATA_DIR` (default `data`), `MIN_FACE_SIZE` (default `80`).
- Embeddings stored as BLOB of 512 float32 elements, L2-normalized.
- Run production with **exactly 1 worker** (the in-RAM matcher is per-process state).
- Tests skip the `slow` group by default (needs the real model): `addopts = "-m 'not slow'"`.

---

### Task 1: Scaffold + config

**Files:**
- Create: `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`, `.gitignore`, `app/__init__.py`, `app/config.py`
- Test: `tests/__init__.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `app.config.Settings` (dataclass: `api_key: str`, `match_threshold: float = 0.40`, `model_name: str = "buffalo_l"`, `data_dir: Path = Path("data")`, `min_face_size: int = 80`), `app.config.load_settings() -> Settings` (reads env, raises `RuntimeError` if `API_KEY` is missing).

- [ ] **Step 1: Create scaffold**

`requirements.txt`:
```
fastapi>=0.111
uvicorn[standard]>=0.30
python-multipart>=0.0.9
numpy>=1.26,<2
opencv-python-headless>=4.9
insightface>=0.7.3
onnxruntime>=1.17
```

`requirements-dev.txt`:
```
-r requirements.txt
pytest>=8
httpx>=0.27
```

`pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = ["slow: integration tests that load the real InsightFace model"]
addopts = "-m 'not slow'"
```

`.gitignore`:
```
.venv/
__pycache__/
data/
*.pyc
.pytest_cache/
```

`app/__init__.py` and `tests/__init__.py`: empty files.

- [ ] **Step 2: Create venv and install dependencies**

Run: `python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt`
Expected: installs successfully (insightface builds from source, takes a few minutes).

- [ ] **Step 3: Write failing test for config**

`tests/test_config.py`:
```python
import pytest

from app.config import Settings, load_settings


def test_load_settings_reads_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("MATCH_THRESHOLD", "0.5")
    s = load_settings()
    assert s.api_key == "secret"
    assert s.match_threshold == 0.5
    assert s.model_name == "buffalo_l"
    assert s.min_face_size == 80


def test_load_settings_requires_api_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        load_settings()
```

- [ ] **Step 4: Run tests, confirm they fail**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` (`app/config.py` does not exist yet).

- [ ] **Step 5: Write `app/config.py`**

```python
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    api_key: str
    match_threshold: float = 0.40
    model_name: str = "buffalo_l"
    data_dir: Path = Path("data")
    min_face_size: int = 80


def load_settings() -> Settings:
    api_key = os.environ.get("API_KEY", "")
    if not api_key:
        raise RuntimeError("API_KEY environment variable is required")
    return Settings(
        api_key=api_key,
        match_threshold=float(os.environ.get("MATCH_THRESHOLD", "0.40")),
        model_name=os.environ.get("MODEL_NAME", "buffalo_l"),
        data_dir=Path(os.environ.get("DATA_DIR", "data")),
        min_face_size=int(os.environ.get("MIN_FACE_SIZE", "80")),
    )
```

- [ ] **Step 6: Run tests, confirm they pass**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt requirements-dev.txt pyproject.toml .gitignore app/ tests/
git commit -m "feat: project scaffold + settings from env"
```

---

### Task 2: DB layer (SQLite)

**Files:**
- Create: `app/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Produces (module `app.db`):
  - `connect(db_path: str | Path) -> sqlite3.Connection` — creates schema if absent, enables foreign_keys, `check_same_thread=False`, `row_factory = sqlite3.Row`
  - `DuplicateUserError(Exception)`
  - `create_user(conn, name: str, email: str, phone: str) -> int` — raises `DuplicateUserError` on a duplicate trio
  - `get_user(conn, user_id: int) -> dict | None` — keys: `id, name, email, phone`
  - `list_users(conn) -> list[dict]` — keys: `id, name, email, phone, created_at, face_count`
  - `delete_user(conn, user_id: int) -> list[str] | None` — returns the deleted image paths, `None` if the user does not exist
  - `add_face(conn, user_id: int, embedding: np.ndarray, image_path: str) -> int`
  - `delete_faces(conn, user_id: int) -> list[str]` — deletes all faces of the user, returns the old image paths
  - `count_faces(conn, user_id: int) -> int`
  - `load_all_faces(conn) -> list[tuple[int, int, np.ndarray]]` — `(face_id, user_id, embedding float32 512-d)`

- [ ] **Step 1: Write failing tests**

`tests/test_db.py`:
```python
import numpy as np
import pytest

from app import db


@pytest.fixture
def conn(tmp_path):
    return db.connect(tmp_path / "test.db")


def _emb(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(512).astype(np.float32)


def test_create_and_get_user(conn):
    uid = db.create_user(conn, "An", "an@x.com", "0901")
    assert db.get_user(conn, uid) == {"id": uid, "name": "An", "email": "an@x.com", "phone": "0901"}


def test_duplicate_trio_raises(conn):
    db.create_user(conn, "An", "an@x.com", "0901")
    with pytest.raises(db.DuplicateUserError):
        db.create_user(conn, "An", "an@x.com", "0901")
    # differing in any single field is not a duplicate
    db.create_user(conn, "An", "an@x.com", "0902")


def test_faces_roundtrip(conn):
    uid = db.create_user(conn, "An", "an@x.com", "0901")
    emb = _emb(1)
    db.add_face(conn, uid, emb, "data/images/a.jpg")
    faces = db.load_all_faces(conn)
    assert len(faces) == 1
    face_id, user_id, loaded = faces[0]
    assert user_id == uid
    np.testing.assert_allclose(loaded, emb, rtol=1e-6)
    assert db.count_faces(conn, uid) == 1


def test_list_users_with_face_count(conn):
    uid = db.create_user(conn, "An", "an@x.com", "0901")
    db.add_face(conn, uid, _emb(1), "a.jpg")
    db.add_face(conn, uid, _emb(2), "b.jpg")
    users = db.list_users(conn)
    assert len(users) == 1
    assert users[0]["face_count"] == 2


def test_delete_user_cascades_and_returns_paths(conn):
    uid = db.create_user(conn, "An", "an@x.com", "0901")
    db.add_face(conn, uid, _emb(1), "a.jpg")
    paths = db.delete_user(conn, uid)
    assert paths == ["a.jpg"]
    assert db.get_user(conn, uid) is None
    assert db.load_all_faces(conn) == []
    assert db.delete_user(conn, 999) is None


def test_delete_faces_returns_old_paths(conn):
    uid = db.create_user(conn, "An", "an@x.com", "0901")
    db.add_face(conn, uid, _emb(1), "a.jpg")
    assert db.delete_faces(conn, uid) == ["a.jpg"]
    assert db.count_faces(conn, uid) == 0
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `.venv/bin/pytest tests/test_db.py -v`
Expected: FAIL — `ImportError: cannot import name 'db'`.

- [ ] **Step 3: Write `app/db.py`**

```python
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
```

- [ ] **Step 4: Run tests, confirm they pass**

Run: `.venv/bin/pytest tests/test_db.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add app/db.py tests/test_db.py
git commit -m "feat: sqlite layer for users and face embeddings"
```

---

### Task 3: Matcher (in-RAM cosine search)

**Files:**
- Create: `app/matcher.py`
- Test: `tests/test_matcher.py`

**Interfaces:**
- Produces: `app.matcher.Matcher`:
  - `__init__(threshold: float)`
  - `add(face_id: int, user_id: int, embedding: np.ndarray) -> None` — L2-normalizes automatically
  - `remove_user(user_id: int) -> None` — removes all faces of the user
  - `search(embedding: np.ndarray) -> tuple[int, float] | None` — best `(user_id, score)` if `score >= threshold`, otherwise `None`
  - `__len__() -> int` — number of faces currently held
  - Thread-safe (every operation holds an internal `threading.Lock`).

- [ ] **Step 1: Write failing tests**

`tests/test_matcher.py`:
```python
import numpy as np

from app.matcher import Matcher


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_empty_matcher_returns_none():
    m = Matcher(threshold=0.4)
    assert m.search(_unit(1)) is None
    assert len(m) == 0


def test_exact_match_returns_user_and_score():
    m = Matcher(threshold=0.4)
    emb = _unit(1)
    m.add(face_id=10, user_id=7, embedding=emb)
    result = m.search(emb)
    assert result is not None
    user_id, score = result
    assert user_id == 7
    assert score > 0.99


def test_below_threshold_returns_none():
    m = Matcher(threshold=0.4)
    m.add(10, 7, _unit(1))
    # a different random vector is nearly orthogonal -> score ~ 0
    assert m.search(_unit(2)) is None


def test_picks_best_of_multiple_users():
    m = Matcher(threshold=0.4)
    a, b = _unit(1), _unit(2)
    m.add(10, 7, a)
    m.add(11, 8, b)
    user_id, _ = m.search(b)
    assert user_id == 8


def test_remove_user_removes_all_their_faces():
    m = Matcher(threshold=0.4)
    emb = _unit(1)
    m.add(10, 7, emb)
    m.add(11, 7, _unit(3))
    m.remove_user(7)
    assert len(m) == 0
    assert m.search(emb) is None


def test_unnormalized_input_is_normalized():
    m = Matcher(threshold=0.4)
    emb = _unit(1)
    m.add(10, 7, emb * 5.0)
    _, score = m.search(emb * 0.1)
    assert score > 0.99
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `.venv/bin/pytest tests/test_matcher.py -v`
Expected: FAIL — `ModuleNotFoundError: app.matcher`.

- [ ] **Step 3: Write `app/matcher.py`**

```python
import threading

import numpy as np

EMBEDDING_DIM = 512


class Matcher:
    """Keeps all embeddings in RAM, brute-force cosine search (sufficient for < 1,000 users)."""

    def __init__(self, threshold: float):
        self._threshold = threshold
        self._lock = threading.Lock()
        self._face_ids: list[int] = []
        self._user_ids: list[int] = []
        self._matrix = np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        vec = embedding.astype(np.float32)
        return vec / np.linalg.norm(vec)

    def add(self, face_id: int, user_id: int, embedding: np.ndarray) -> None:
        vec = self._normalize(embedding)
        with self._lock:
            self._face_ids.append(face_id)
            self._user_ids.append(user_id)
            self._matrix = np.vstack([self._matrix, vec[None, :]])

    def remove_user(self, user_id: int) -> None:
        with self._lock:
            keep = [i for i, uid in enumerate(self._user_ids) if uid != user_id]
            self._face_ids = [self._face_ids[i] for i in keep]
            self._user_ids = [self._user_ids[i] for i in keep]
            self._matrix = (
                self._matrix[keep]
                if keep
                else np.empty((0, EMBEDDING_DIM), dtype=np.float32)
            )

    def search(self, embedding: np.ndarray) -> tuple[int, float] | None:
        vec = self._normalize(embedding)
        with self._lock:
            if not self._face_ids:
                return None
            scores = self._matrix @ vec
            best = int(np.argmax(scores))
            score = float(scores[best])
            if score < self._threshold:
                return None
            return self._user_ids[best], score

    def __len__(self) -> int:
        with self._lock:
            return len(self._face_ids)
```

- [ ] **Step 4: Run tests, confirm they pass**

Run: `.venv/bin/pytest tests/test_matcher.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add app/matcher.py tests/test_matcher.py
git commit -m "feat: in-memory cosine matcher"
```

---

### Task 4: Face engine wrapper (InsightFace)

**Files:**
- Create: `app/face_engine.py`

**Interfaces:**
- Produces (module `app.face_engine`):
  - `InvalidImageError(Exception)` — image cannot be decoded
  - `DetectedFace` (dataclass): `embedding: np.ndarray` (512-d, L2-normalized), `min_side: float` (smaller side of the bbox, pixels)
  - `FaceEngine.__init__(model_name: str = "buffalo_l")` — loads InsightFace, CPU
  - `FaceEngine.extract(image_bytes: bytes) -> list[DetectedFace]` — **sorted largest face first**; raises `InvalidImageError` if decoding fails
- Note: the real engine has no fast unit tests (needs the ~300MB model) — it is verified in Task 8 (integration, marked slow). API tests use a FakeEngine with the same interface.

- [ ] **Step 1: Write `app/face_engine.py`**

```python
from dataclasses import dataclass

import cv2
import numpy as np


class InvalidImageError(Exception):
    pass


@dataclass
class DetectedFace:
    embedding: np.ndarray  # 512-d, L2-normalized
    min_side: float        # smaller side of the bounding box (pixels)


class FaceEngine:
    def __init__(self, model_name: str = "buffalo_l"):
        from insightface.app import FaceAnalysis  # slow import, only when the real engine is needed

        self._app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=-1, det_size=(640, 640))

    def extract(self, image_bytes: bytes) -> list[DetectedFace]:
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise InvalidImageError("cannot decode image")
        faces = self._app.get(img)
        detected = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox
            detected.append(
                DetectedFace(
                    embedding=f.normed_embedding,
                    min_side=float(min(x2 - x1, y2 - y1)),
                )
            )
        detected.sort(key=lambda d: d.min_side, reverse=True)
        return detected
```

- [ ] **Step 2: Confirm the import works (without loading the model)**

Run: `.venv/bin/python -c "from app.face_engine import FaceEngine, DetectedFace, InvalidImageError; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add app/face_engine.py
git commit -m "feat: insightface engine wrapper"
```

---

### Task 5: App factory + auth + enroll endpoint

**Files:**
- Create: `app/auth.py`, `app/main.py`, `app/server.py`
- Test: `tests/conftest.py`, `tests/test_api_enroll.py`

**Interfaces:**
- Consumes: `Settings`, `db.*`, `Matcher`, `DetectedFace`, `InvalidImageError` (Tasks 1–4, signatures as declared).
- Produces:
  - `app.main.create_app(settings: Settings | None = None, engine=None) -> FastAPI` — if `engine is None`, the lifespan loads `FaceEngine(settings.model_name)` at startup. State: `app.state.settings`, `app.state.conn`, `app.state.engine`, `app.state.matcher`, `app.state.write_lock`.
  - `app.auth.verify_api_key` — FastAPI dependency that reads `request.app.state.settings.api_key`, compares via `secrets.compare_digest`, wrong → `HTTPException(401)`.
  - `app.server.app` — instance for uvicorn (`uvicorn app.server:app`).
  - Endpoint `POST /users` (multipart `image`, `name`, `email`, `phone`) → 201 `{"user_id", "name", "email", "phone"}`; 409 `duplicate_user`; 422 `invalid_image | no_face | multiple_faces | face_too_small`.
  - Endpoint `GET /health` (no auth) → `{"status": "ok", "faces": <int>}`.
  - `tests/conftest.py` produces the `client` fixture (TestClient with FakeEngine, `X-API-Key: test-key` header preset) and the `FakeEngine` class — later test tasks reuse these.

- [ ] **Step 1: Write conftest with FakeEngine**

`tests/conftest.py`:
```python
import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.face_engine import DetectedFace, InvalidImageError
from app.main import create_app


def embedding_from_seed(seed: str) -> np.ndarray:
    rng = np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16))
    vec = rng.standard_normal(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


class FakeEngine:
    """Reads the uploaded bytes as a text 'script':
    - b"face:<seed>"        -> 1 large face (min_side=200), embedding derived from seed
    - b"face:<seed>:small"  -> 1 small face (min_side=40)
    - b"faces:<a>,<b>"      -> multiple faces, first one largest
    - b"noface"             -> no face
    - b"garbage"            -> raise InvalidImageError
    Same seed -> same embedding (score ~1.0); different seed -> nearly orthogonal (score ~0).
    """

    def extract(self, image_bytes: bytes) -> list[DetectedFace]:
        text = image_bytes.decode(errors="replace")
        if text == "garbage":
            raise InvalidImageError("cannot decode image")
        if text == "noface":
            return []
        if text.startswith("faces:"):
            seeds = text[len("faces:"):].split(",")
            return [
                DetectedFace(embedding_from_seed(s), 200.0 - 10.0 * i)
                for i, s in enumerate(seeds)
            ]
        if text.startswith("face:"):
            parts = text.split(":")
            size = 40.0 if parts[-1] == "small" else 200.0
            return [DetectedFace(embedding_from_seed(parts[1]), size)]
        raise InvalidImageError("cannot decode image")


@pytest.fixture
def client(tmp_path):
    settings = Settings(api_key="test-key", data_dir=tmp_path)
    app = create_app(settings=settings, engine=FakeEngine())
    with TestClient(app) as c:
        c.headers["X-API-Key"] = "test-key"
        yield c


def upload(content: bytes):
    """Helper building the files= part of a multipart request."""
    return {"image": ("img.jpg", content, "image/jpeg")}
```

- [ ] **Step 2: Write failing tests for auth + enroll**

`tests/test_api_enroll.py`:
```python
from tests.conftest import upload


def test_missing_api_key_returns_401(client):
    client.headers.pop("X-API-Key")
    r = client.post("/users", data={"name": "An", "email": "a@x.com", "phone": "1"},
                    files=upload(b"face:an"))
    assert r.status_code == 401


def test_wrong_api_key_returns_401(client):
    client.headers["X-API-Key"] = "wrong"
    r = client.get("/users")
    assert r.status_code == 401


def test_health_needs_no_auth(client):
    client.headers.pop("X-API-Key")
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_enroll_success(client):
    r = client.post("/users", data={"name": "An", "email": "a@x.com", "phone": "0901"},
                    files=upload(b"face:an"))
    assert r.status_code == 201
    body = r.json()
    assert body["user_id"] == 1
    assert body["name"] == "An"


def test_enroll_duplicate_trio_409(client):
    data = {"name": "An", "email": "a@x.com", "phone": "0901"}
    client.post("/users", data=data, files=upload(b"face:an"))
    r = client.post("/users", data=data, files=upload(b"face:an"))
    assert r.status_code == 409
    assert r.json()["reason"] == "duplicate_user"


def test_enroll_rejects_bad_images(client):
    data = {"name": "An", "email": "a@x.com", "phone": "0901"}
    cases = [
        (b"garbage", "invalid_image"),
        (b"noface", "no_face"),
        (b"faces:an,binh", "multiple_faces"),
        (b"face:an:small", "face_too_small"),
    ]
    for content, reason in cases:
        r = client.post("/users", data=data, files=upload(content))
        assert r.status_code == 422, content
        assert r.json()["reason"] == reason
```

- [ ] **Step 3: Run tests, confirm they fail**

Run: `.venv/bin/pytest tests/test_api_enroll.py -v`
Expected: FAIL — `ModuleNotFoundError: app.main`.

- [ ] **Step 4: Write `app/auth.py`**

```python
import secrets

from fastapi import Header, HTTPException, Request


def verify_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    expected = request.app.state.settings.api_key
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid or missing API key")
```

- [ ] **Step 5: Write `app/main.py` (factory + health + enroll)**

```python
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from . import db
from .auth import verify_api_key
from .config import Settings, load_settings
from .face_engine import InvalidImageError
from .matcher import Matcher


def create_app(settings: Settings | None = None, engine=None) -> FastAPI:
    settings = settings or load_settings()
    images_dir = settings.data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app_: FastAPI):
        if app_.state.engine is None:
            from .face_engine import FaceEngine

            app_.state.engine = FaceEngine(settings.model_name)
        yield

    app = FastAPI(title="Face Recognition API", lifespan=lifespan)
    app.state.settings = settings
    app.state.conn = db.connect(settings.data_dir / "faces.db")
    app.state.engine = engine
    app.state.matcher = Matcher(threshold=settings.match_threshold)
    app.state.write_lock = threading.Lock()

    for face_id, user_id, emb in db.load_all_faces(app.state.conn):
        app.state.matcher.add(face_id, user_id, emb)

    def _error(status: int, detail: str, reason: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"detail": detail, "reason": reason})

    def _extract_faces(image: UploadFile):
        image_bytes = image.file.read()
        return image_bytes, app.state.engine.extract(image_bytes)

    def _validate_single_face(faces) -> str | None:
        if len(faces) == 0:
            return "no_face"
        if len(faces) > 1:
            return "multiple_faces"
        if faces[0].min_side < settings.min_face_size:
            return "face_too_small"
        return None

    def _save_image(image_bytes: bytes) -> str:
        path = images_dir / f"{uuid.uuid4().hex}.jpg"
        path.write_bytes(image_bytes)
        return str(path)

    def _delete_files(paths: list[str]) -> None:
        for p in paths:
            Path(p).unlink(missing_ok=True)

    @app.get("/health")
    def health():
        return {"status": "ok", "faces": len(app.state.matcher)}

    @app.post("/users", status_code=201, dependencies=[Depends(verify_api_key)])
    def enroll(
        image: UploadFile = File(...),
        name: str = Form(...),
        email: str = Form(...),
        phone: str = Form(...),
    ):
        try:
            image_bytes, faces = _extract_faces(image)
        except InvalidImageError:
            return _error(422, "cannot decode image", "invalid_image")
        reason = _validate_single_face(faces)
        if reason:
            return _error(422, f"image rejected: {reason}", reason)
        with app.state.write_lock:
            try:
                user_id = db.create_user(app.state.conn, name, email, phone)
            except db.DuplicateUserError:
                return _error(409, "user already exists", "duplicate_user")
            image_path = _save_image(image_bytes)
            face_id = db.add_face(app.state.conn, user_id, faces[0].embedding, image_path)
            app.state.matcher.add(face_id, user_id, faces[0].embedding)
        return {"user_id": user_id, "name": name, "email": email, "phone": phone}

    return app
```

`app/server.py`:
```python
from app.main import create_app

app = create_app()
```

- [ ] **Step 6: Run tests, confirm they pass**

Run: `.venv/bin/pytest tests/test_api_enroll.py -v`
Expected: 6 passed. (Test `test_wrong_api_key_returns_401` calls `GET /users`, which does not exist yet — does FastAPI return 401 from the dependency before the 404? NO: a missing route returns 404. Fix that test to use `POST /users` with a wrong key instead of `GET /users`.)

> **Note for the implementer:** in `test_wrong_api_key_returns_401` use:
> ```python
> r = client.post("/users", data={"name": "x", "email": "y", "phone": "z"}, files=upload(b"face:x"))
> assert r.status_code == 401
> ```

- [ ] **Step 7: Run the full test suite**

Run: `.venv/bin/pytest -v`
Expected: all pass (config + db + matcher + enroll).

- [ ] **Step 8: Commit**

```bash
git add app/auth.py app/main.py app/server.py tests/conftest.py tests/test_api_enroll.py
git commit -m "feat: app factory, api-key auth, enroll endpoint"
```

---

### Task 6: Recognize endpoint

**Files:**
- Modify: `app/main.py` (add route inside `create_app`, after the `enroll` route)
- Test: `tests/test_api_recognize.py`

**Interfaces:**
- Consumes: fixture `client`, helper `upload`, `FakeEngine` semantics from `tests/conftest.py` (Task 5).
- Produces: `POST /recognize` (multipart `image`) — always 200 except for a corrupt image (422):
  - Match: `{"matched": true, "confidence": <float>, "user": {"id", "name", "email", "phone"}}`
  - No match: `{"matched": false, "reason": "below_threshold"}`
  - No face: `{"matched": false, "reason": "no_face"}`

- [ ] **Step 1: Write failing tests**

`tests/test_api_recognize.py`:
```python
from tests.conftest import upload


def _enroll(client, seed="an", name="An", email="a@x.com", phone="0901"):
    r = client.post("/users", data={"name": name, "email": email, "phone": phone},
                    files=upload(f"face:{seed}".encode()))
    assert r.status_code == 201
    return r.json()["user_id"]


def test_recognize_enrolled_user(client):
    uid = _enroll(client)
    r = client.post("/recognize", files=upload(b"face:an"))
    assert r.status_code == 200
    body = r.json()
    assert body["matched"] is True
    assert body["confidence"] > 0.99
    assert body["user"] == {"id": uid, "name": "An", "email": "a@x.com", "phone": "0901"}


def test_recognize_stranger_below_threshold(client):
    _enroll(client)
    r = client.post("/recognize", files=upload(b"face:stranger"))
    assert r.status_code == 200
    assert r.json() == {"matched": False, "reason": "below_threshold"}


def test_recognize_empty_database(client):
    r = client.post("/recognize", files=upload(b"face:an"))
    assert r.status_code == 200
    assert r.json() == {"matched": False, "reason": "below_threshold"}


def test_recognize_no_face_returns_200(client):
    r = client.post("/recognize", files=upload(b"noface"))
    assert r.status_code == 200
    assert r.json() == {"matched": False, "reason": "no_face"}


def test_recognize_uses_largest_face(client):
    uid = _enroll(client, seed="an")
    _enroll(client, seed="binh", name="Binh", email="b@x.com", phone="0902")
    # the frame has 2 faces, "an" is the largest (first in the list)
    r = client.post("/recognize", files=upload(b"faces:an,binh"))
    assert r.json()["matched"] is True
    assert r.json()["user"]["id"] == uid


def test_recognize_garbage_returns_422(client):
    r = client.post("/recognize", files=upload(b"garbage"))
    assert r.status_code == 422
    assert r.json()["reason"] == "invalid_image"


def test_recognize_requires_auth(client):
    client.headers.pop("X-API-Key")
    r = client.post("/recognize", files=upload(b"face:an"))
    assert r.status_code == 401
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `.venv/bin/pytest tests/test_api_recognize.py -v`
Expected: FAIL — 404 (route does not exist yet).

- [ ] **Step 3: Add route to `create_app` (after `enroll`, before `return app`)**

```python
    @app.post("/recognize", dependencies=[Depends(verify_api_key)])
    def recognize(image: UploadFile = File(...)):
        try:
            _, faces = _extract_faces(image)
        except InvalidImageError:
            return _error(422, "cannot decode image", "invalid_image")
        if not faces:
            return {"matched": False, "reason": "no_face"}
        # faces are already sorted largest first (person closest to the camera)
        result = app.state.matcher.search(faces[0].embedding)
        if result is None:
            return {"matched": False, "reason": "below_threshold"}
        user_id, score = result
        user = db.get_user(app.state.conn, user_id)
        return {"matched": True, "confidence": round(score, 4), "user": user}
```

- [ ] **Step 4: Run tests, confirm they pass**

Run: `.venv/bin/pytest tests/test_api_recognize.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api_recognize.py
git commit -m "feat: recognize endpoint"
```

---

### Task 7: Manage endpoints (update photos, list, delete)

**Files:**
- Modify: `app/main.py` (add 3 routes inside `create_app`)
- Test: `tests/test_api_manage.py`

**Interfaces:**
- Consumes: fixture `client`, helper `upload` (Task 5); `db.delete_faces`, `db.delete_user`, `db.count_faces`, `db.list_users` (Task 2); `matcher.remove_user` (Task 3).
- Produces:
  - `POST /users/{user_id}/faces?mode=replace|append` (default `replace`) → 200 `{"user_id", "face_count"}`; 404 `user_not_found`; 422 same as enroll
  - `GET /users` → 200 list `{"id", "name", "email", "phone", "created_at", "face_count"}`
  - `DELETE /users/{user_id}` → 204; 404 `user_not_found`

- [ ] **Step 1: Write failing tests**

`tests/test_api_manage.py`:
```python
from tests.conftest import upload


def _enroll(client, seed="an", name="An", email="a@x.com", phone="0901"):
    r = client.post("/users", data={"name": name, "email": email, "phone": phone},
                    files=upload(f"face:{seed}".encode()))
    assert r.status_code == 201
    return r.json()["user_id"]


def test_list_users(client):
    _enroll(client)
    _enroll(client, seed="binh", name="Binh", email="b@x.com", phone="0902")
    r = client.get("/users")
    assert r.status_code == 200
    users = r.json()
    assert len(users) == 2
    assert users[0]["face_count"] == 1
    assert set(users[0]) == {"id", "name", "email", "phone", "created_at", "face_count"}


def test_update_faces_append(client):
    uid = _enroll(client)
    r = client.post(f"/users/{uid}/faces?mode=append", files=upload(b"face:an2"))
    assert r.status_code == 200
    assert r.json() == {"user_id": uid, "face_count": 2}
    # both the old and new embeddings recognize this user
    for content in (b"face:an", b"face:an2"):
        rec = client.post("/recognize", files=upload(content))
        assert rec.json()["matched"] is True
        assert rec.json()["user"]["id"] == uid


def test_update_faces_replace_is_default(client):
    uid = _enroll(client)
    r = client.post(f"/users/{uid}/faces", files=upload(b"face:an_new"))
    assert r.status_code == 200
    assert r.json() == {"user_id": uid, "face_count": 1}
    # the old image is no longer recognized, the new one is
    assert client.post("/recognize", files=upload(b"face:an")).json()["matched"] is False
    assert client.post("/recognize", files=upload(b"face:an_new")).json()["matched"] is True


def test_update_faces_unknown_user_404(client):
    r = client.post("/users/999/faces", files=upload(b"face:x"))
    assert r.status_code == 404
    assert r.json()["reason"] == "user_not_found"


def test_update_faces_validates_image(client):
    uid = _enroll(client)
    r = client.post(f"/users/{uid}/faces", files=upload(b"noface"))
    assert r.status_code == 422
    assert r.json()["reason"] == "no_face"


def test_delete_user(client):
    uid = _enroll(client)
    r = client.delete(f"/users/{uid}")
    assert r.status_code == 204
    assert client.get("/users").json() == []
    assert client.post("/recognize", files=upload(b"face:an")).json()["matched"] is False


def test_delete_unknown_user_404(client):
    r = client.delete("/users/999")
    assert r.status_code == 404
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `.venv/bin/pytest tests/test_api_manage.py -v`
Expected: FAIL — 404/405 (route does not exist yet).

- [ ] **Step 3: Add 3 routes to `create_app`**

Add imports at the top of the file if not already present: `from fastapi import Query` and `from fastapi.responses import Response`.

```python
    @app.post("/users/{user_id}/faces", dependencies=[Depends(verify_api_key)])
    def update_faces(
        user_id: int,
        image: UploadFile = File(...),
        mode: str = Query("replace", pattern="^(replace|append)$"),
    ):
        if db.get_user(app.state.conn, user_id) is None:
            return _error(404, "user not found", "user_not_found")
        try:
            image_bytes, faces = _extract_faces(image)
        except InvalidImageError:
            return _error(422, "cannot decode image", "invalid_image")
        reason = _validate_single_face(faces)
        if reason:
            return _error(422, f"image rejected: {reason}", reason)
        with app.state.write_lock:
            if mode == "replace":
                old_paths = db.delete_faces(app.state.conn, user_id)
                app.state.matcher.remove_user(user_id)
                _delete_files(old_paths)
            image_path = _save_image(image_bytes)
            face_id = db.add_face(app.state.conn, user_id, faces[0].embedding, image_path)
            app.state.matcher.add(face_id, user_id, faces[0].embedding)
            face_count = db.count_faces(app.state.conn, user_id)
        return {"user_id": user_id, "face_count": face_count}

    @app.get("/users", dependencies=[Depends(verify_api_key)])
    def list_users():
        return db.list_users(app.state.conn)

    @app.delete("/users/{user_id}", dependencies=[Depends(verify_api_key)])
    def delete_user(user_id: int):
        with app.state.write_lock:
            paths = db.delete_user(app.state.conn, user_id)
            if paths is None:
                return _error(404, "user not found", "user_not_found")
            app.state.matcher.remove_user(user_id)
            _delete_files(paths)
        return Response(status_code=204)
```

- [ ] **Step 4: Run the full test suite, confirm it passes**

Run: `.venv/bin/pytest -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api_manage.py
git commit -m "feat: manage endpoints (update faces, list, delete)"
```

---

### Task 8: Integration test with the real model (marked slow)

**Files:**
- Test: `tests/test_integration.py`

**Interfaces:**
- Consumes: real `FaceEngine` (Task 4), `create_app` (Task 5). Uses the group photo `t1` bundled with the insightface package (`insightface.data.get_image("t1")`) — crop each face as an enroll image, no external image files needed.

- [ ] **Step 1: Write test**

`tests/test_integration.py`:
```python
import cv2
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def real_engine():
    from app.face_engine import FaceEngine

    return FaceEngine("buffalo_l")  # first run downloads the ~300MB model


def _crop_jpeg(img, bbox, margin=0.4) -> bytes:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    crop = img[max(0, y1 - my): y2 + my, max(0, x1 - mx): x2 + mx]
    # upscale so the re-detected face is definitely >= MIN_FACE_SIZE
    if min(crop.shape[:2]) < 220:
        scale = 220 / min(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=scale, fy=scale)
    ok, buf = cv2.imencode(".jpg", crop)
    assert ok
    return buf.tobytes()


def test_end_to_end_enroll_recognize_delete(tmp_path, real_engine):
    from insightface.data import get_image

    img = get_image("t1")  # multi-person group photo, bundled with insightface
    raw = real_engine._app.get(img)
    assert len(raw) >= 2, "t1 image must have at least 2 faces"
    raw.sort(key=lambda f: f.bbox[2] - f.bbox[0], reverse=True)
    person_a = _crop_jpeg(img, raw[0].bbox)
    person_b = _crop_jpeg(img, raw[1].bbox)

    settings = Settings(api_key="itest", data_dir=tmp_path)
    app = create_app(settings=settings, engine=real_engine)
    with TestClient(app) as client:
        client.headers["X-API-Key"] = "itest"

        r = client.post("/users", data={"name": "A", "email": "a@x.com", "phone": "1"},
                        files={"image": ("a.jpg", person_a, "image/jpeg")})
        assert r.status_code == 201, r.text
        uid = r.json()["user_id"]

        # recognizes the enrolled person
        r = client.post("/recognize", files={"image": ("a.jpg", person_a, "image/jpeg")})
        assert r.json()["matched"] is True, r.text
        assert r.json()["user"]["id"] == uid

        # another person in the group photo -> no match
        r = client.post("/recognize", files={"image": ("b.jpg", person_b, "image/jpeg")})
        assert r.json()["matched"] is False, r.text

        # after deletion, no longer recognized
        assert client.delete(f"/users/{uid}").status_code == 204
        r = client.post("/recognize", files={"image": ("a.jpg", person_a, "image/jpeg")})
        assert r.json()["matched"] is False
```

- [ ] **Step 2: Run integration test**

Run: `.venv/bin/pytest tests/test_integration.py -m slow -v`
Expected: 1 passed (slow on the first run because the ~300MB model downloads to `~/.insightface/models/`).

- [ ] **Step 3: Confirm the default test run still skips the slow group**

Run: `.venv/bin/pytest`
Expected: tests pass, the last line shows `deselected` for the slow tests.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration with real insightface model"
```

---

### Task 9: Docker + compose + README

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.env.example`, `README.md`

**Interfaces:**
- Consumes: `app.server:app` (Task 5), env vars from Global Constraints.

- [ ] **Step 1: Write `Dockerfile`**

```dockerfile
FROM python:3.11-slim

# build-essential: insightface builds from source
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model into the image so startup does not have to download ~300MB
ARG MODEL_NAME=buffalo_l
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='${MODEL_NAME}', providers=['CPUExecutionProvider']).prepare(ctx_id=-1)"

COPY app ./app

ENV DATA_DIR=/srv/data
EXPOSE 8000

# Use exactly 1 worker: the matcher keeps state in the process RAM
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

`.dockerignore`:
```
.venv/
data/
tests/
docs/
__pycache__/
.pytest_cache/
.git/
```

- [ ] **Step 2: Write `docker-compose.yml` + `.env.example`**

`docker-compose.yml`:
```yaml
services:
  face-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      API_KEY: ${API_KEY:?set API_KEY in .env}
      MATCH_THRESHOLD: ${MATCH_THRESHOLD:-0.40}
    volumes:
      - ./data:/srv/data
    restart: unless-stopped
```

`.env.example`:
```
API_KEY=change-me-to-a-long-random-string
MATCH_THRESHOLD=0.40
```

- [ ] **Step 3: Build image**

Run: `docker build -t face-recognition-api .`
Expected: build succeeds (the model download step takes a few minutes the first time).

- [ ] **Step 4: Smoke test container**

```bash
API_KEY=smoke-test docker compose up -d
sleep 15
curl -s http://localhost:8000/health
# Expected: {"status":"ok","faces":0}
curl -s -X POST http://localhost:8000/recognize -H "X-API-Key: smoke-test" -F "image=@/dev/null"
# Expected: HTTP 422 invalid_image (empty image)
docker compose down
```

- [ ] **Step 5: Write `README.md`**

Required content (write it in full, no placeholders):
- One-paragraph description: what the service does.
- Env vars table (the 5 variables from Global Constraints + meaning + default).
- How to run: `cp .env.example .env` → edit `API_KEY` → `docker compose up -d --build`.
- Table of the 6 endpoints (method, path, 1-line description) + `curl` examples for enroll and recognize:
```bash
curl -X POST http://localhost:8000/users \
  -H "X-API-Key: $API_KEY" \
  -F "name=Nguyen Van A" -F "email=a@example.com" -F "phone=0901234567" \
  -F "image=@photo.jpg"

curl -X POST http://localhost:8000/recognize \
  -H "X-API-Key: $API_KEY" -F "image=@frame.jpg"
```
- Operational notes: run only 1 worker; tune `MATCH_THRESHOLD` on false matches (raise) / misses (lower); for a weak server switch to `MODEL_NAME=buffalo_s` (rebuild the image with `--build-arg MODEL_NAME=buffalo_s`); server requirement 2 vCPU / 4GB RAM; Swagger docs at `/docs`.
- How to run tests: `pytest` (fast) and `pytest -m slow` (integration with the real model).

- [ ] **Step 6: Commit**

```bash
git add Dockerfile docker-compose.yml .dockerignore .env.example README.md
git commit -m "feat: docker packaging and docs"
```

---

## Self-Review (completed)

- **Spec coverage:** enroll (Task 5), recognize + always-200 + largest face (Task 6), update replace/append + list + delete (Task 7), API key auth (Task 5), 4-reason validation (Tasks 5–7), threshold/model via env (Task 1), in-RAM matcher + lock (Task 3), SQLite UNIQUE trio (Task 2), original image storage (Task 5), Docker + 1 worker + model baked (Task 9), 3 test tiers (Tasks 2–3, 5–7, 8). The health endpoint is a small addition for smoke testing/monitoring.
- **Type consistency:** `Matcher.search` returns `tuple[int, float] | None` — used correctly in Task 6; `db.delete_user` returns `list[str] | None` — used correctly in Task 7; `DetectedFace.min_side` — used correctly in validation.
- **Placeholder scan:** no TBD/TODO left; README has a concrete required-content list.
