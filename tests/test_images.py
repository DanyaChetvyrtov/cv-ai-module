from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from cv_test.images import ImageInputError, decode_image


def test_decodes_bgr_without_changing_dimensions(png_bytes):
    image = decode_image(png_bytes, max_bytes=10000, max_pixels=2000)
    assert image.shape == (32, 48, 3)
    assert image.dtype == np.uint8
    assert image[0, 0].tolist() == [20, 10, 250]


@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP"])
def test_supported_formats_and_grayscale(image_format):
    output = BytesIO()
    Image.new("L", (20, 10), color=100).save(output, format=image_format)
    image = decode_image(output.getvalue(), max_bytes=10000, max_pixels=1000)
    assert image.shape == (10, 20, 3)


def test_applies_exif_orientation():
    output = BytesIO()
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise.
    Image.new("RGB", (20, 10)).save(output, format="JPEG", exif=exif)
    image = decode_image(output.getvalue(), max_bytes=10000, max_pixels=1000)
    assert image.shape == (20, 10, 3)


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\n"])
def test_rejects_invalid_image(data):
    with pytest.raises(ImageInputError) as error:
        decode_image(data, max_bytes=10000, max_pixels=2000)
    assert error.value.status_code == 400


def test_rejects_excessive_file_size(png_bytes):
    with pytest.raises(ImageInputError) as error:
        decode_image(png_bytes, max_bytes=len(png_bytes) - 1, max_pixels=2000)
    assert error.value.status_code == 413


def test_rejects_excessive_dimensions(png_bytes):
    with pytest.raises(ImageInputError) as error:
        decode_image(png_bytes, max_bytes=10000, max_pixels=100)
    assert error.value.status_code == 413


def test_rejects_unsupported_format():
    output = BytesIO()
    Image.new("RGB", (10, 10)).save(output, format="GIF")
    with pytest.raises(ImageInputError) as error:
        decode_image(output.getvalue(), max_bytes=10000, max_pixels=1000)
    assert error.value.status_code == 415
