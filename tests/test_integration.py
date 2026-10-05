import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cv_test.api import create_app
from cv_test.config import Settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("CV_RUN_INTEGRATION") != "1", reason="Set CV_RUN_INTEGRATION=1"),
]


def test_real_yolo_api_and_cli(tmp_path):
    import torch
    import ultralytics

    torch.set_num_threads(2)
    photo = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    with TestClient(create_app(Settings(device="cpu"))) as client:
        response = client.post("/vision/detect", files={"image": ("bus.jpg", photo.read_bytes())})
    assert response.status_code == 200
    result = response.json()
    labels = {d["label"] for d in result["detections"]}
    assert {"bus", "person"} <= labels
    for detection in result["detections"]:
        box = detection["bbox"]
        assert 0 <= box["x1"] < box["x2"] <= result["width"]
        assert 0 <= box["y1"] < box["y2"] <= result["height"]
        assert 0.25 <= detection["confidence"] <= 1

    annotated = tmp_path / "result.jpg"
    report = tmp_path / "result.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cv_test.cli",
            str(photo),
            "--output",
            str(annotated),
            "--json",
            str(report),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "CV_DEVICE": "cpu", "OMP_NUM_THREADS": "2"},
    )
    assert completed.returncode == 0, completed.stderr
    assert annotated.stat().st_size > 0
    assert "bus" in {d["label"] for d in json.loads(report.read_text())["detections"]}
