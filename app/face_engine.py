from dataclasses import dataclass

import cv2
import numpy as np


# Border added on each side for the retry, as a share of the longer side
PAD_RATIO = 0.25


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

    def extract(self, image_bytes: bytes, pad_retry: bool = False) -> list[DetectedFace]:
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        if buf.size == 0:
            raise InvalidImageError("empty image")
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise InvalidImageError("cannot decode image")
        faces = self._app.get(img)
        if not faces and pad_retry:
            # A selfie held too close fills the frame with the forehead or chin
            # cut off, and the detector — trained on faces with room around
            # them — finds nothing. A border gives it that room back. Padding
            # does not rescale, so box sizes stay in the original's pixels.
            pad = int(max(img.shape[:2]) * PAD_RATIO)
            img = cv2.copyMakeBorder(
                img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
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
