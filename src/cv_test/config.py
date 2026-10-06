from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CV_", env_file=".env", extra="ignore")

    model_path: Path = Path("models/yolo11n.pt")
    device: str = "cpu"
    image_size: int = Field(default=640, ge=32, le=2048, multiple_of=32)
    confidence: float = Field(default=0.25, ge=0.01, le=1.0)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=20_000_000, gt=0)
    face_enabled: bool = True
    face_models_path: Path = Path("models")
    min_face_size: int = Field(default=60, ge=20, le=300)
