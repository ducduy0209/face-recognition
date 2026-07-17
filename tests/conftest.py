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
    - b"faces:<a>,<b>"      -> multiple faces, first one is the largest
    - b"noface"             -> no face
    - b"garbage"            -> raise InvalidImageError
    Same seed -> same embedding (score ~1.0); different seeds -> nearly orthogonal (score ~0).
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
