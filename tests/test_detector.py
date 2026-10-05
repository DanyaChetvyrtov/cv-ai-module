from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest

from cv_test.config import Settings
from cv_test.detector import YoloDetector


class TensorStub:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


@pytest.mark.parametrize("has_boxes", [True, False])
def test_converts_yolo_output_to_public_schema(has_boxes):
    boxes = (
        SimpleNamespace(
            xyxy=TensorStub([[1.5, 2.0, 18.0, 9.0]]),
            conf=TensorStub([0.81234567]),
            cls=TensorStub([2.0]),
        )
        if has_boxes
        else None
    )
    calls = []

    class ModelStub:
        def predict(self, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=boxes, names={2: "car"})]

    detector = YoloDetector.__new__(YoloDetector)
    detector.settings = Settings()
    detector._lock = Lock()
    detector._model = ModelStub()
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    result = detector.detect(image, confidence=0.5)
    assert (result.width, result.height) == (20, 10)
    assert calls[0]["source"] is image
    assert calls[0]["conf"] == 0.5
    if has_boxes:
        assert result.detections[0].class_id == 2
        assert result.detections[0].label == "car"
        assert result.detections[0].confidence == 0.812346
        assert result.detections[0].bbox.x1 == 1.5
    else:
        assert result.detections == []
