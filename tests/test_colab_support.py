import sys
import types
from pathlib import Path

from jura_hypersumm import colab_support


def test_uploaded_documents_are_always_deleted(monkeypatch) -> None:
    fake_files = types.SimpleNamespace()

    def upload():
        Path("decision.docx").write_bytes(b"test")
        return {"decision.docx": b"test"}

    fake_files.upload = upload
    fake_colab = types.ModuleType("google.colab")
    fake_colab.files = fake_files
    fake_google = types.ModuleType("google")
    fake_google.colab = fake_colab
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)
    monkeypatch.setattr(colab_support, "require_colab", lambda: None)

    with colab_support.uploaded_docx_files() as paths:
        assert len(paths) == 1
        uploaded_path = paths[0]
        assert uploaded_path.exists()

    assert not uploaded_path.exists()
    assert not uploaded_path.parent.exists()
