from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class ImageInputError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def decode_image(data: bytes, *, max_bytes: int, max_pixels: int) -> np.ndarray:
    """Validate bytes, apply EXIF orientation, and return a BGR uint8 image."""
    if not data:
        raise ImageInputError("The uploaded image is empty.")
    if len(data) > max_bytes:
        raise ImageInputError(f"Image exceeds the {max_bytes}-byte limit.", 413)
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise ImageInputError("Supported image formats: JPEG, PNG, WEBP.", 415)
            if source.width * source.height > max_pixels:
                raise ImageInputError(f"Image exceeds the {max_pixels}-pixel limit.", 413)
            source.load()
            rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    except ImageInputError:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageInputError("Image dimensions are too large.", 413) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageInputError("Cannot decode image. Upload a valid JPEG, PNG or WEBP.") from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
