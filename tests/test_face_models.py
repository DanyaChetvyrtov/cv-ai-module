import hashlib
from io import BytesIO

import pytest

import cv_test.face_models as module


@pytest.fixture
def downloads(monkeypatch):
    digest = hashlib.sha256(b"abc").hexdigest()
    monkeypatch.setattr(module, "MODELS", (("a", "a.onnx", digest, 3), ("b", "b.onnx", digest, 3)))
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        return BytesIO(b"abc")

    monkeypatch.setattr(module.urllib.request, "urlopen", fetch)
    return calls


def test_cached_verified_models_work_offline_and_corrupt_cache_is_repaired(tmp_path, downloads):
    first = module.ensure_face_models(tmp_path)
    assert len(downloads) == 2
    assert module.ensure_face_models(tmp_path) == first
    assert len(downloads) == 2
    first[0].write_bytes(b"bad")
    module.ensure_face_models(tmp_path)
    assert len(downloads) == 3
    assert first[0].read_bytes() == b"abc"


def test_untrusted_model_bytes_are_not_installed(tmp_path, downloads, monkeypatch):
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: BytesIO(b"bad"))
    with pytest.raises(RuntimeError, match="checksum"):
        module.ensure_face_models(tmp_path)
    assert list(tmp_path.iterdir()) == []
