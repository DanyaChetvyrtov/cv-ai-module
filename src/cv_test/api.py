import logging
from contextlib import asynccontextmanager
from typing import Annotated

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from cv_test.config import Settings
from cv_test.faces import FaceEmbedding, FaceExtractor, SFaceExtractor
from cv_test.images import ImageInputError, decode_image

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    face_extractor: FaceExtractor | None = None,
) -> FastAPI:
    config = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.faces = face_extractor
        if config.face_enabled and app.state.faces is None:
            app.state.faces = await run_in_threadpool(SFaceExtractor, config)
        yield

    app = FastAPI(title="CV Face Service", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "face_model": app.state.faces.model_id if app.state.faces else "disabled",
        }

    def extract_face(data: bytes) -> FaceEmbedding:
        if app.state.faces is None:
            raise HTTPException(status_code=503, detail="Face recognition is unavailable.")
        decoded = decode_image(
            data,
            max_bytes=config.max_upload_bytes,
            max_pixels=config.max_image_pixels,
        )
        return app.state.faces.extract(decoded)

    @app.post("/faces/embedding", response_model=FaceEmbedding)
    async def face_embedding(
        image: Annotated[UploadFile, File(description="JPEG, PNG or WEBP image")],
    ) -> FaceEmbedding:
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
