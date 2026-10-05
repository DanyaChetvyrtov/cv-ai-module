from threading import Lock
from time import perf_counter
from typing import Protocol

import numpy as np

from cv_test.config import Settings
from cv_test.schemas import BoundingBox, Detection, DetectionResult


class Detector(Protocol):
    def detect(self, image: np.ndarray, confidence: float) -> DetectionResult: ...


class YoloDetector:
    """One reusable model; serialize prediction because YOLO has mutable predictor state."""

    def __init__(self, settings: Settings):
        from ultralytics import YOLO

        self.settings = settings
        self._lock = Lock()
        settings.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model = YOLO(str(settings.model_path), task="detect")
        if self._model.task != "detect":
            raise ValueError("CV_MODEL_PATH must point to an object detection model.")
        # Initialize the predictor before accepting requests (also verifies the device).
        self.detect(np.zeros((64, 64, 3), dtype=np.uint8), settings.confidence)

    def detect(self, image: np.ndarray, confidence: float) -> DetectionResult:
        if not 0.01 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.01 and 1.0")
        with self._lock:
            started = perf_counter()
            result = self._model.predict(
                source=image,
                conf=confidence,
                imgsz=self.settings.image_size,
                device=self.settings.device,
                max_det=100,
                verbose=False,
                save=False,
            )[0]
            detections = []
            if result.boxes is not None:
                for xyxy, score, class_id in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                    strict=True,
                ):
                    detections.append(
                        Detection(
                            class_id=int(class_id),
                            label=result.names[int(class_id)],
                            confidence=round(score, 6),
                            bbox=BoundingBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]),
                        )
                    )
            inference_ms = round((perf_counter() - started) * 1000, 3)
        height, width = image.shape[:2]
        return DetectionResult(
            model=self.settings.model_path.name,
            width=width,
            height=height,
            inference_ms=inference_ms,
            detections=detections,
        )
