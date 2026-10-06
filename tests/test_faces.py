from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cv_test.api import create_app
from cv_test.config import Settings
from cv_test.faces import FaceEmbedding, SFaceExtractor
from cv_test.images import ImageInputError
from cv_test.schemas import BoundingBox, DetectionResult


class EmptyDetector:
    def detect(self, image, confidence):
        return DetectionResult(
            model="stub",
            width=image.shape[1],
            height=image.shape[0],
            inference_ms=0,
            detections=[],
        )


class FakeFaces:
    model_id = "test-face-model"

    def extract(self, image):
        return FaceEmbedding(
            model=self.model_id,
            embedding=[1.0] + [0.0] * 127,
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            inference_ms=1,
        )


def test_internal_face_api_validates_images_and_returns_a_vector(png_bytes):
    with TestClient(
        create_app(Settings(face_enabled=False), EmptyDetector(), FakeFaces())
    ) as client:
        result = client.post("/faces/embedding", files={"image": ("face.png", png_bytes)})
        assert result.status_code == 200
        assert len(result.json()["embedding"]) == 128
        assert result.json()["model"] == "test-face-model"
        assert client.get("/health").json()["face_model"] == "test-face-model"
        assert (
            client.post("/faces/embedding", files={"image": ("bad", b"broken")}).status_code == 400
        )
        assert client.post("/faces/embedding").status_code == 422


def test_disabled_faces_are_explicitly_unavailable(png_bytes):
    with TestClient(create_app(Settings(face_enabled=False), EmptyDetector())) as client:
        assert (
            client.post("/faces/embedding", files={"image": ("a.png", png_bytes)}).status_code
            == 503
        )


@pytest.mark.parametrize("status", [413, 415, 422])
def test_face_validation_status_is_preserved(status, png_bytes):
    class InvalidFace(FakeFaces):
        def extract(self, image):
            raise ImageInputError("Invalid face", status)

    with TestClient(
        create_app(Settings(face_enabled=False), EmptyDetector(), InvalidFace())
    ) as client:
        response = client.post("/faces/embedding", files={"image": ("a.png", png_bytes)})
        assert response.status_code == status


def test_face_model_failure_does_not_leak_details(png_bytes):
    class FailedFace(FakeFaces):
        def extract(self, image):
            raise RuntimeError("Private file/device details")

    with TestClient(
        create_app(Settings(face_enabled=False), EmptyDetector(), FailedFace())
    ) as client:
        response = client.post("/faces/embedding", files={"image": ("a.png", png_bytes)})
        assert response.status_code == 503
        assert response.json() == {"detail": "Face recognition is unavailable."}


@pytest.fixture
def engines(monkeypatch):
    import cv_test.faces as module

    class Detector:
        faces = np.array([[10, 15, 80, 100] + [0] * 10 + [0.95]], dtype=np.float32)

        def setInputSize(self, size):
            self.size = size

        def detect(self, image):
            return 1, self.faces

    class Recognizer:
        vector = np.ones((1, 128), dtype=np.float32)

        def alignCrop(self, image, face):
            return image

        def feature(self, image):
            return self.vector

    detection, recognition = Detector(), Recognizer()
    monkeypatch.setattr(module, "ensure_face_models", lambda _: (Path("a"), Path("b")))
    monkeypatch.setattr(module.cv2.FaceDetectorYN, "create", lambda *args: detection)
    monkeypatch.setattr(module.cv2.FaceRecognizerSF, "create", lambda *args: recognition)
    return SFaceExtractor(Settings()), detection, recognition


def test_alignment_features_are_normalized_and_box_is_rescaled(engines):
    model, detector, _ = engines
    result = model.extract(np.zeros((2560, 2560, 3), dtype=np.uint8))
    assert detector.size == (1280, 1280)
    assert np.linalg.norm(result.embedding) == pytest.approx(1.0)
    assert result.bbox == BoundingBox(x1=20, y1=30, x2=180, y2=230)


@pytest.mark.parametrize("count", [0, 2])
def test_exactly_one_face_is_required(engines, count):
    model, detector, _ = engines
    detector.faces = np.repeat(detector.faces, count, axis=0)
    with pytest.raises(ImageInputError) as error:
        model.extract(np.zeros((256, 256, 3), dtype=np.uint8))
    assert error.value.status_code == 422


def test_small_faces_are_rejected(engines):
    model, detector, _ = engines
    detector.faces[0, 2] = 10
    with pytest.raises(ImageInputError, match="too small"):
        model.extract(np.zeros((256, 256, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    "vector", [np.zeros((1, 128)), np.ones((1, 127)), np.full((1, 128), np.nan)]
)
def test_invalid_embeddings_are_not_returned(engines, vector):
    model, _, recognizer = engines
    recognizer.vector = vector
    with pytest.raises(RuntimeError, match="Invalid face model output"):
        model.extract(np.zeros((256, 256, 3), dtype=np.uint8))
