#!/usr/bin/env python3
"""Download a checksum-pinned public-domain NASA photo for CV integration tests only."""

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/astronaut.png"
)
SHA256 = "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5"


def main():
    with urllib.request.urlopen(URL, timeout=30) as response:
        data = response.read(1024 * 1024)
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise SystemExit("Invalid integration photo checksum")
    Path(sys.argv[1]).write_bytes(data)


if __name__ == "__main__":
    main()
