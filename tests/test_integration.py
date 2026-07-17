import cv2
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def real_engine():
    from app.face_engine import FaceEngine

    return FaceEngine("buffalo_l")  # first run downloads the model (~300MB)


def _crop_jpeg(img, bbox, margin=0.4) -> bytes:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    crop = img[max(0, y1 - my): y2 + my, max(0, x1 - mx): x2 + mx]
    # upscale so the re-detected face is guaranteed >= MIN_FACE_SIZE
    if min(crop.shape[:2]) < 220:
        scale = 220 / min(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=scale, fy=scale)
    ok, buf = cv2.imencode(".jpg", crop)
    assert ok
    return buf.tobytes()


def test_end_to_end_enroll_recognize_delete(tmp_path, real_engine):
    from insightface.data import get_image

    img = get_image("t1")  # group photo with multiple people, bundled with insightface
    raw = real_engine._app.get(img)
    assert len(raw) >= 2, "image t1 must contain at least 2 faces"
    raw.sort(key=lambda f: f.bbox[2] - f.bbox[0], reverse=True)
    person_a = _crop_jpeg(img, raw[0].bbox)
    person_b = _crop_jpeg(img, raw[1].bbox)

    settings = Settings(api_key="itest", data_dir=tmp_path)
    app = create_app(settings=settings, engine=real_engine)
    with TestClient(app) as client:
        client.headers["X-API-Key"] = "itest"

        # empty image -> 422 invalid_image (must not be a 500)
        r = client.post("/recognize", files={"image": ("e.jpg", b"", "image/jpeg")})
        assert r.status_code == 422, r.text
        assert r.json()["reason"] == "invalid_image"

        r = client.post("/users", data={"name": "A", "email": "a@x.com", "phone": "1"},
                        files={"image": ("a.jpg", person_a, "image/jpeg")})
        assert r.status_code == 201, r.text
        uid = r.json()["user_id"]

        # recognizes the enrolled person
        r = client.post("/recognize", files={"image": ("a.jpg", person_a, "image/jpeg")})
        assert r.json()["matched"] is True, r.text
        assert r.json()["user"]["id"] == uid

        # a different person from the group photo -> no match
        r = client.post("/recognize", files={"image": ("b.jpg", person_b, "image/jpeg")})
        assert r.json()["matched"] is False, r.text

        # after deletion the person is no longer recognized
        assert client.delete(f"/users/{uid}").status_code == 204
        r = client.post("/recognize", files={"image": ("a.jpg", person_a, "image/jpeg")})
        assert r.json()["matched"] is False
