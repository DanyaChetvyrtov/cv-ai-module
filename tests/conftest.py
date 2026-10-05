from io import BytesIO

import pytest
from PIL import Image


@pytest.fixture
def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (48, 32), color=(250, 10, 20)).save(output, format="PNG")
    return output.getvalue()
