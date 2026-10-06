import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cv_test.api import create_app
from cv_test.config import Settings
from cv_test.schemas import DetectionResult

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("CV_RUN_FACE_INTEGRATION") != "1", reason="Set CV_RUN_FACE_INTEGRATION=1"
    ),
]


class EmptyDetector:
    def detect(self, image, confidence):
        return DetectionResult(
            model="stub",
            width=image.shape[1],
            height=image.shape[0],
            inference_ms=0,
            detections=[],
        )


def encoded(image):
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_real_yunet_sface_with_distinct_queries_and_invalid_faces():
    photo = Path(os.environ["CV_FACE_TEST_IMAGE"]).read_bytes()
    source = Image.open(BytesIO(photo)).convert("RGB")
    multiple = Image.new("RGB", (source.width * 2, source.height))
    multiple.paste(source, (0, 0))
    multiple.paste(source, (source.width, 0))
    with TestClient(create_app(Settings(), EmptyDetector())) as client:
        first = client.post("/faces/embedding", files={"image": ("face.png", photo)})
        second = client.post(
            "/faces/embedding", files={"image": ("query.png", encoded(source.resize((460, 460))))}
        )
        assert first.status_code == second.status_code == 200
        vector1, vector2 = first.json()["embedding"], second.json()["embedding"]
        assert len(vector1) == 128
        assert np.linalg.norm(vector1) == pytest.approx(1, abs=1e-5)
        assert np.dot(vector1, vector2) > 0.8
        for image in (Image.new("RGB", (512, 512)), multiple):
            response = client.post("/faces/embedding", files={"image": ("bad.png", encoded(image))})
            assert response.status_code == 422
