import numpy as np
import pytest
from fastapi.testclient import TestClient

from cv_test.api import create_app
from cv_test.config import Settings
from cv_test.faces import FaceEmbedding
from cv_test.schemas import BoundingBox


class StubFaces:
    model_id = "sface-fixture"

    def __init__(self):
        self.calls = 0

    def extract(self, image: np.ndarray) -> FaceEmbedding:
        self.calls += 1
        return FaceEmbedding(
            model=self.model_id,
            embedding=[1.0] + [0.0] * 127,
            bbox=BoundingBox(x1=1, y1=2, x2=image.shape[1] - 1, y2=image.shape[0] - 1),
            inference_ms=1.0,
        )


def test_health_reports_face_model():
    faces = StubFaces()
    with TestClient(create_app(Settings(), faces)) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "face_model": "sface-fixture"}


def test_valid_upload_returns_face_embedding(png_bytes):
    faces = StubFaces()
    with TestClient(create_app(Settings(), faces)) as client:
        response = client.post(
            "/faces/embedding",
            files={"image": ("photo.png", png_bytes, "image/png")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "sface-fixture"
    assert len(body["embedding"]) == 128
    assert body["bbox"] == {"x1": 1.0, "y1": 2.0, "x2": 47.0, "y2": 31.0}
    assert faces.calls == 1


def test_requires_upload():
    with TestClient(create_app(Settings(), StubFaces())) as client:
        assert client.post("/faces/embedding").status_code == 422


@pytest.mark.parametrize("data", [b"", b"broken JPEG"])
def test_rejects_bad_image_without_running_model(data):
    faces = StubFaces()
    with TestClient(create_app(Settings(), faces)) as client:
        response = client.post("/faces/embedding", files={"image": ("x.jpg", data)})
    assert response.status_code == 400
    assert faces.calls == 0


def test_rejects_large_upload(png_bytes):
    faces = StubFaces()
    with TestClient(create_app(Settings(max_upload_bytes=8), faces)) as client:
        response = client.post("/faces/embedding", files={"image": ("p.png", png_bytes)})
    assert response.status_code == 413
    assert faces.calls == 0


def test_disabled_face_service_returns_503(png_bytes):
    with TestClient(create_app(Settings(face_enabled=False))) as client:
        response = client.post("/faces/embedding", files={"image": ("p.png", png_bytes)})
    assert response.status_code == 503
    assert response.json() == {"detail": "Face recognition is unavailable."}
