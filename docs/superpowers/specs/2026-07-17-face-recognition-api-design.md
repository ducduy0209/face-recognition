# Face Recognition API — Design Spec

**Date:** 2026-07-17
**Status:** Design approved, awaiting implementation planning

## 1. Purpose

Service providing a face recognition API for a check-in system:

- An external project (admin page) enrolls a photo + user info after the admin approves the photo.
- The check-in kiosk continuously captures photos via camera and sends them for recognition; on a match, the user's info (name, email, phone) is returned to display a check-in notification.

## 2. Scope & constraints

- Scale: under 1,000 users.
- Deploy: Docker on a VPS/cloud, **no GPU** (CPU-only).
- Target server: 2 vCPU / 4 GB RAM.
- The info trio `(name, email, phone)` is **unique** — identifies a user.
- Security: API key via the `X-API-Key` header (server-to-server).
- Requirement: fast (~100–300ms/request on CPU) and accurate recognition.

## 3. Technology

| Component | Choice | Rationale |
|---|---|---|
| Web framework | FastAPI + Uvicorn | Async, fast, auto-generates Swagger docs for integrators |
| Face engine | InsightFace `buffalo_l` (ONNX Runtime CPU) | Top open-source ArcFace accuracy, runs well on CPU |
| Matching | Brute-force cosine similarity in RAM (numpy) | < 1,000 users → matching < 1ms, no vector DB needed |
| Storage | SQLite (file in a Docker volume) | Sufficient for the scale, no separate DB server needed |
| Original images | Files in `data/images/` | Enables re-embedding on model upgrades, serves audit |

The model is switchable via env (`MODEL_NAME`, e.g. `buffalo_s` for a weak server).

## 4. Architecture

```
face-recognition/
├── app/
│   ├── main.py          # FastAPI app + routes
│   ├── face_engine.py   # InsightFace wrapper: detect + embedding (loaded once at startup)
│   ├── matcher.py       # In-RAM embedding matrix + cosine search
│   ├── db.py            # SQLite: users + faces tables
│   ├── auth.py          # Dependency checking X-API-Key
│   └── config.py        # Env: API_KEY, MATCH_THRESHOLD, MODEL_NAME, DATA_DIR
├── data/                # Volume: SQLite + original images
├── tests/
├── Dockerfile
└── docker-compose.yml
```

**Data flow:** image → decode → InsightFace detect → 512-d embedding → cosine against the in-RAM matrix → above threshold returns the user.

**In-RAM matcher:** loads all embeddings from SQLite at startup; updates immediately on enroll/update/delete (with a lock preventing races between requests).

## 5. Data model (SQLite)

```sql
CREATE TABLE users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    email      TEXT NOT NULL,
    phone      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (name, email, phone)
);

CREATE TABLE faces (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    embedding  BLOB NOT NULL,      -- 512 float32
    image_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

A user can have multiple photos (multiple embeddings) to improve accuracy across different angles/lighting.

## 6. API contract

All endpoints require the `X-API-Key` header. Wrong/missing → `401`.

### POST `/users` — Enroll a new user
- Multipart form: `image` (file), `name`, `email`, `phone`.
- The image must contain **exactly 1 face**, bbox ≥ 80px.
- Response `201`: `{ "user_id": 1, "name": ..., "email": ..., "phone": ... }`
- `409` if the trio already exists. `422` if the image fails validation (with `reason`: `invalid_image` | `no_face` | `multiple_faces` | `face_too_small`).

### POST `/users/{id}/faces?mode=replace|append` — Update photos
- Multipart form: `image`. `mode` defaults to `replace` (deletes old embeddings + images), `append` adds an extra photo.
- Image validation same as enroll. `404` if the user does not exist.
- Response `200`: `{ "user_id": 1, "face_count": 2 }`

### POST `/recognize` — Recognize
- Multipart form: `image`.
- If the frame has multiple faces → take the **largest face** (person closest to the camera).
- **Always returns `200`** (even on no match / no face) so the camera-side polling loop does not have to handle errors:
  - Match: `{ "matched": true, "confidence": 0.87, "user": { "id", "name", "email", "phone" } }`
  - No match: `{ "matched": false, "reason": "below_threshold" }`
  - No face: `{ "matched": false, "reason": "no_face" }`
  - Corrupt image: `422` (a genuine client-side error).

### GET `/users` — List
- Response `200`: `[{ "id", "name", "email", "phone", "face_count", "created_at" }]`

### DELETE `/users/{id}` — Delete user
- Deletes the user + all embeddings + image files. `404` if not found.
- Response `204`.

## 7. Recognition logic

- Embeddings are L2-normalized; match score = cosine similarity, taking the **max over all faces** (not averaged per user).
- Default threshold `MATCH_THRESHOLD = 0.40` (typical buffalo_l range: 0.35–0.45), adjustable via env for real-world tuning.
- Strict enroll (exactly 1 face, large enough), lenient recognize (take the largest face) — fits the context: enroll is admin-reviewed, recognize comes from a natural camera feed.

## 8. Error handling

| Code | When |
|---|---|
| 401 | Wrong/missing `X-API-Key` |
| 404 | `user_id` does not exist |
| 409 | Trio `(name, email, phone)` already exists |
| 422 | Image cannot be decoded / fails enroll validation |

Consistent error body: `{ "detail": "...", "reason": "..." }`.

## 9. Testing

1. **Unit — matcher**: synthetic vectors; checks threshold, top-1, add/remove, basic thread-safety.
2. **API — mocked engine**: all endpoints with a fake face engine (fast, no model download); checks auth, validation, 409/404, replace/append mode.
3. **Integration (marked `slow`)**: real model + sample images — enroll → recognize the right person → stranger gets `matched: false` → delete → no longer recognized.

## 10. Deploy

- `Dockerfile`: python slim + onnxruntime + insightface + opencv-python-headless; the model is pre-downloaded into the image at build time (avoids downloading at runtime).
- `docker-compose.yml`: mounts the `./data` volume, env `API_KEY`, `MATCH_THRESHOLD`, port 8000.
- Server requirements: 2 vCPU / 4 GB RAM / 20 GB disk, no GPU. On a weaker VPS → switch to `MODEL_NAME=buffalo_s`.

## 11. Out of scope (YAGNI)

- Anti-spoofing / liveness detection (against printed photos, screen photos) — noted as a risk, do later if needed.
- Vector database, horizontal scaling, multiple API keys with permissions.
- Admin UI — the external project handles it.
