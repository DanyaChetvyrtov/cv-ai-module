import numpy as np
import pytest
from fastapi.testclient import TestClient

from cv_test.api import create_app
from cv_test.config import Settings
from cv_test.schemas import BoundingBox, Detection, DetectionResult


class StubDetector:
    def __init__(self, detections=None):
        self.confidences = []
        self.detections = detections if detections is not None else []

    def detect(self, image: np.ndarray, confidence: float) -> DetectionResult:
        self.confidences.append(confidence)
        return DetectionResult(
            model="test.pt",
            width=image.shape[1],
            height=image.shape[0],
            inference_ms=1.0,
            detections=self.detections,
        )


def test_valid_upload_returns_documented_contract(png_bytes):
    detector = StubDetector(
        [
            Detection(
                class_id=0,
                label="person",
                confidence=0.9,
                bbox=BoundingBox(x1=1, y1=2, x2=20, y2=30),
            )
        ]
    )
    with TestClient(create_app(Settings(), detector)) as client:
        assert client.get("/health").status_code == 200
        response = client.post(
            "/vision/detect?confidence=0.7", files={"image": ("photo.png", png_bytes, "image/png")}
        )
    assert response.status_code == 200
    body = response.json()
    assert (body["width"], body["height"]) == (48, 32)
    assert body["detections"][0]["label"] == "person"
    assert body["detections"][0]["bbox"] == {"x1": 1.0, "y1": 2.0, "x2": 20.0, "y2": 30.0}
    assert detector.confidences == [0.7]


def test_no_objects_is_success_and_uses_default_confidence(png_bytes):
    detector = StubDetector()
    with TestClient(create_app(Settings(confidence=0.4), detector)) as client:
        response = client.post("/vision/detect", files={"image": ("photo.png", png_bytes)})
    assert response.status_code == 200
    assert response.json()["detections"] == []
    assert detector.confidences == [0.4]


@pytest.mark.parametrize("query", ["0", "1.1", "abc"])
def test_rejects_invalid_confidence(query, png_bytes):
    with TestClient(create_app(Settings(), StubDetector())) as client:
        response = client.post(
            f"/vision/detect?confidence={query}", files={"image": ("p.png", png_bytes)}
        )
    assert response.status_code == 422


def test_requires_upload():
    with TestClient(create_app(Settings(), StubDetector())) as client:
        assert client.post("/vision/detect").status_code == 422


@pytest.mark.parametrize("data", [b"", b"broken JPEG"])
def test_rejects_bad_image_without_running_model(data):
    detector = StubDetector()
    with TestClient(create_app(Settings(), detector)) as client:
        response = client.post("/vision/detect", files={"image": ("x.jpg", data)})
    assert response.status_code == 400
    assert detector.confidences == []


def test_rejects_large_upload(png_bytes):
    detector = StubDetector()
    with TestClient(create_app(Settings(max_upload_bytes=8), detector)) as client:
        response = client.post("/vision/detect", files={"image": ("p.png", png_bytes)})
    assert response.status_code == 413
    assert detector.confidences == []


def test_reports_inference_error(png_bytes):
    class FailingDetector:
        def detect(self, image, confidence):
            raise RuntimeError("Internal device details")

    with TestClient(create_app(Settings(), FailingDetector())) as client:
        response = client.post("/vision/detect", files={"image": ("p.png", png_bytes)})
    assert response.status_code == 503
    assert response.json() == {"detail": "Object detection is unavailable."}
