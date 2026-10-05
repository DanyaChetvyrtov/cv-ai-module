import argparse
import sys
from contextlib import redirect_stdout
from pathlib import Path

import cv2

from cv_test.config import Settings
from cv_test.detector import YoloDetector
from cv_test.images import annotate_image, decode_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect objects in a local image using YOLO.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument(
        "--output", type=Path, help="Optional annotated image, e.g. outputs/result.jpg"
    )
    parser.add_argument("--json", type=Path, help="Write JSON here instead of stdout")
    args = parser.parse_args()
    try:
        settings = Settings()
        confidence = args.confidence if args.confidence is not None else settings.confidence
        if not 0.01 <= confidence <= 1.0:
            parser.error("--confidence must be between 0.01 and 1.0")
        with args.image.open("rb") as source:
            data = source.read(settings.max_upload_bytes + 1)
        image = decode_image(
            data,
            max_bytes=settings.max_upload_bytes,
            max_pixels=settings.max_image_pixels,
        )
        # Keep stdout machine-readable even on the first model download.
        with redirect_stdout(sys.stderr):
            detector = YoloDetector(settings)
            result = detector.detect(image, confidence)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.output), annotate_image(image, result)):
                raise OSError(f"Could not write {args.output}")
        payload = result.model_dump_json(indent=2)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
