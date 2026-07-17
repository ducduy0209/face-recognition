import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, Response

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

    @app.post("/recognize", dependencies=[Depends(verify_api_key)])
    def recognize(image: UploadFile = File(...)):
        try:
            _, faces = _extract_faces(image)
        except InvalidImageError:
            return _error(422, "cannot decode image", "invalid_image")
        if not faces:
            return {"matched": False, "reason": "no_face"}
        # faces are sorted largest first (person closest to the camera)
        result = app.state.matcher.search(faces[0].embedding)
        if result is None:
            return {"matched": False, "reason": "below_threshold"}
        user_id, score = result
        user = db.get_user(app.state.conn, user_id)
        if user is None:
            # user was deleted between search and get -> treat as no match
            return {"matched": False, "reason": "below_threshold"}
        return {"matched": True, "confidence": round(score, 4), "user": user}

    @app.post("/users/{user_id}/faces", dependencies=[Depends(verify_api_key)])
    def update_faces(
        user_id: int,
        image: UploadFile = File(...),
        mode: str = Query("replace", pattern="^(replace|append)$"),
    ):
        try:
            image_bytes, faces = _extract_faces(image)
        except InvalidImageError:
            return _error(422, "cannot decode image", "invalid_image")
        reason = _validate_single_face(faces)
        if reason:
            return _error(422, f"image rejected: {reason}", reason)
        with app.state.write_lock:
            # check inside the lock to avoid racing with DELETE /users/{id}
            if db.get_user(app.state.conn, user_id) is None:
                return _error(404, "user not found", "user_not_found")
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

    return app
