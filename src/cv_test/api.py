import logging
from contextlib import asynccontextmanager
from typing import Annotated

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from cv_test.config import Settings
from cv_test.detector import Detector, YoloDetector
from cv_test.faces import FaceEmbedding, FaceExtractor, SFaceExtractor
from cv_test.images import ImageInputError, decode_image
from cv_test.schemas import DetectionResult

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    detector: Detector | None = None,
    face_extractor: FaceExtractor | None = None,
) -> FastAPI:
    config = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.detector = (
            detector if detector is not None else await run_in_threadpool(YoloDetector, config)
        )
        app.state.faces = face_extractor
        if config.face_enabled and app.state.faces is None:
            app.state.faces = await run_in_threadpool(SFaceExtractor, config)
        yield

    app = FastAPI(title="CV Test", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "model": config.model_path.name,
            "device": config.device,
            "face_model": app.state.faces.model_id if app.state.faces else "disabled",
        }

    def infer(data: bytes, confidence: float) -> DetectionResult:
        image = decode_image(
            data, max_bytes=config.max_upload_bytes, max_pixels=config.max_image_pixels
        )
        return app.state.detector.detect(image, confidence)

    @app.post("/vision/detect", response_model=DetectionResult)
    async def detect(
        image: Annotated[UploadFile, File(description="JPEG, PNG or WEBP image")],
        confidence: Annotated[float | None, Query(ge=0.01, le=1.0)] = None,
    ) -> DetectionResult:
        try:
            data = await image.read(config.max_upload_bytes + 1)
            threshold = confidence if confidence is not None else config.confidence
            return await run_in_threadpool(infer, data, threshold)
        except ImageInputError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.exception("YOLO inference failed")
            raise HTTPException(status_code=503, detail="Object detection is unavailable.") from exc
        finally:
            await image.close()

    def extract_face(data: bytes) -> FaceEmbedding:
        if app.state.faces is None:
            raise HTTPException(status_code=503, detail="Face recognition is unavailable.")
        decoded = decode_image(
            data, max_bytes=config.max_upload_bytes, max_pixels=config.max_image_pixels
        )
        return app.state.faces.extract(decoded)

    @app.post("/faces/embedding", response_model=FaceEmbedding)
    async def face_embedding(image: Annotated[UploadFile, File()]) -> FaceEmbedding:
        try:
            data = await image.read(config.max_upload_bytes + 1)
            return await run_in_threadpool(extract_face, data)
        except ImageInputError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except (RuntimeError, cv2.error) as exc:
            logger.exception("Face inference failed")
            raise HTTPException(status_code=503, detail="Face recognition is unavailable.") from exc
        finally:
            await image.close()

    return app
