# Face Recognition API

Face recognition service for a check-in system. Enroll a photo + user info (the trio `name`, `email`, `phone` is unique); recognize camera images and return the user info on a match. Runs entirely on CPU with InsightFace (ArcFace), FastAPI and SQLite — no GPU, no separate database server.

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `API_KEY` | Auth key — every request must include the `X-API-Key` header | **required** |
| `MATCH_THRESHOLD` | Cosine threshold to count as a match (raise if misidentifying, lower if missing people) | `0.40` |
| `MODEL_NAME` | InsightFace model (`buffalo_l` most accurate, `buffalo_s` for weak servers) | `buffalo_l` |
| `DATA_DIR` | Directory holding SQLite + original images | `data` |
| `MIN_FACE_SIZE` | Minimum face size (px) at enroll time | `80` |

## Running with Docker

```bash
cp .env.example .env
# change API_KEY in .env to a long random string
docker compose up -d --build
```

The server runs at `http://localhost:8000`, Swagger docs at `http://localhost:8000/docs`.

## API

Every endpoint (except `GET /health`) requires the `X-API-Key` header.

| Method | Path | Description |
|---|---|---|
| POST | `/users` | Enroll a new user: multipart `image` + `name` + `email` + `phone` |
| POST | `/users/{id}/faces?mode=replace\|append` | Update photos (default `replace`; `append` adds an extra photo) |
| POST | `/recognize` | Recognize an image — always returns 200, `matched: true/false` |
| GET | `/users` | List users with their photo counts |
| DELETE | `/users/{id}` | Delete a user + all photos/embeddings |
| GET | `/health` | Health check (no auth) |

### Examples

```bash
# Enroll
curl -X POST http://localhost:8000/users \
  -H "X-API-Key: $API_KEY" \
  -F "name=Nguyen Van A" -F "email=a@example.com" -F "phone=0901234567" \
  -F "image=@photo.jpg"

# Recognize (a camera frame)
curl -X POST http://localhost:8000/recognize \
  -H "X-API-Key: $API_KEY" -F "image=@frame.jpg"
# Match:    {"matched": true, "confidence": 0.87, "user": {"id": 1, "name": "...", "email": "...", "phone": "..."}}
# No match: {"matched": false, "reason": "below_threshold"}
```

Enroll errors return 422 with a `reason`: `invalid_image` | `no_face` | `multiple_faces` | `face_too_small`. A duplicate info trio returns 409.

## Operational notes

- **Run exactly 1 worker** (already configured in the Dockerfile) — the face index lives in process RAM; multiple workers would drift out of sync.
- Recommended server: **2 vCPU / 4 GB RAM**, no GPU needed. Each recognition request takes ~100–300ms.
- Weak server: rebuild with the small model — `docker compose build --build-arg MODEL_NAME=buffalo_s` and set `MODEL_NAME=buffalo_s` in the env.
- Resize uploaded images to ~640px to reduce transfer time.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

.venv/bin/pytest              # fast tests (fake engine, no model needed)
.venv/bin/pytest -m slow      # integration test with the real model (~300MB download on first run)

API_KEY=dev-key .venv/bin/uvicorn app.server:app --reload
```
