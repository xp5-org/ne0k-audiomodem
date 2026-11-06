import pytest

from modemgui import native


def _built():
    return all(r["ok"] for r in native.inventory())


@pytest.mark.skipif(not _built(), reason="run `make` first")
def test_native_smoke():
    assert native.smoke() == []


def test_missing_library_message_says_how_to_fix(tmp_path, monkeypatch):
    monkeypatch.setenv(native.LIB_DIR_ENV, str(tmp_path))
    native._cache.clear()
    with pytest.raises(native.NativeError, match="run `make`"):
        native.load("libofdm")
    native._cache.clear()
