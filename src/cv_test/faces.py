from threading import Lock
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np
from pydantic import BaseModel, Field

from cv_test.config import Settings
from cv_test.face_models import FACE_MODEL_ID, ensure_face_models
from cv_test.images import ImageInputError
from cv_test.schemas import BoundingBox


class FaceEmbedding(BaseModel):
    model: str
    embedding: list[float] = Field(min_length=128, max_length=128)
    bbox: BoundingBox
    inference_ms: float


class FaceExtractor(Protocol):
    model_id: str

    def extract(self, image: np.ndarray) -> FaceEmbedding: ...


class SFaceExtractor:
    """Detect exactly one face, align its landmarks and produce a normalized SFace vector."""

    model_id = FACE_MODEL_ID

    def __init__(self, settings: Settings):
        detection_path, recognition_path = ensure_face_models(settings.face_models_path)
        self._detector = cv2.FaceDetectorYN.create(
            str(detection_path), "", (320, 320), 0.9, 0.3, 5000
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(recognition_path), "")
        self._lock = Lock()
        self._min_size = settings.min_face_size

    def extract(self, image: np.ndarray) -> FaceEmbedding:
        height, width = image.shape[:2]
        scale = min(1.0, 1280 / max(height, width))
        working = (
            cv2.resize(image, (round(width * scale), round(height * scale))) if scale < 1 else image
        )
        with self._lock:
            started = perf_counter()
            self._detector.setInputSize((working.shape[1], working.shape[0]))
            _, faces = self._detector.detect(working)
            if faces is None or len(faces) == 0:
                raise ImageInputError("No face detected. Use a clear front-facing photo.", 422)
            if len(faces) != 1:
                raise ImageInputError("The photo must contain exactly one face.", 422)
            face = faces[0]
            if min(float(face[2]), float(face[3])) < self._min_size:
                raise ImageInputError("The face is too small. Use a closer photo.", 422)
            aligned = self._recognizer.alignCrop(working, face)
            vector = self._recognizer.feature(aligned).reshape(-1).astype(np.float32).copy()
            norm = float(np.linalg.norm(vector))
            if vector.size != 128 or not np.isfinite(vector).all() or norm < 1e-8:
                raise RuntimeError("Invalid face model output")
            vector /= norm
            elapsed = (perf_counter() - started) * 1000
        x, y, w, h = (float(value) / scale for value in face[:4])
        return FaceEmbedding(
            model=self.model_id,
            embedding=vector.tolist(),
            bbox=BoundingBox(
                x1=max(0, x), y1=max(0, y), x2=min(width, x + w), y2=min(height, y + h)
            ),
            inference_ms=round(elapsed, 3),
        )
