import threading

import numpy as np

EMBEDDING_DIM = 512


class Matcher:
    """Keeps all embeddings in RAM, brute-force cosine search (enough for < 1,000 users)."""

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
