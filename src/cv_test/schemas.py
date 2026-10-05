from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox


class DetectionResult(BaseModel):
    model: str
    width: int
    height: int
    inference_ms: float
    detections: list[Detection]
