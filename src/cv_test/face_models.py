"""Pinned OpenCV Zoo models, downloaded once and verified before local inference."""

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

ZOO_REVISION = "47534e27c9851bb1128ccc0102f1145e27f23f98"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
FACE_MODEL_ID = "sface-2021dec:" + SFACE_SHA256
MODELS = (
    ("face_detection_yunet", "face_detection_yunet_2023mar.onnx", YUNET_SHA256, 232589),
    ("face_recognition_sface", "face_recognition_sface_2021dec.onnx", SFACE_SHA256, 38696353),
)


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ensure_face_models(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for folder, filename, digest, size in MODELS:
        target = directory / filename
        if target.is_file() and target.stat().st_size == size and checksum(target) == digest:
            paths.append(target)
            continue
        url = (
            f"https://media.githubusercontent.com/media/opencv/opencv_zoo/"
            f"{ZOO_REVISION}/models/{folder}/{filename}"
        )
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
                temporary = Path(output.name)
                with urllib.request.urlopen(url, timeout=60) as response:
                    received = 0
                    while chunk := response.read(1024 * 1024):
                        received += len(chunk)
                        if received > size:
                            raise RuntimeError("Unexpected face model size")
                        output.write(chunk)
            if temporary.stat().st_size != size or checksum(temporary) != digest:
                raise RuntimeError("Face model checksum verification failed")
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        paths.append(target)
    return paths[0], paths[1]
