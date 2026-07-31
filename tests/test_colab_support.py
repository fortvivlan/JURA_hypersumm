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


def test_explicit_local_documents_are_preserved(monkeypatch, tmp_path: Path) -> None:
    first = tmp_path / "b.docx"
    second = tmp_path / "a.DOCX"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    monkeypatch.setattr(colab_support, "is_colab", lambda: False)

    with colab_support.selected_docx_files([first, second]) as paths:
        assert [path.name for path in paths] == ["a.DOCX", "b.docx"]

    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"


def test_local_artifact_root_is_created(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(colab_support, "is_colab", lambda: False)
    target = tmp_path / "artifacts"

    resolved = colab_support.prepare_artifact_root(target)

    assert resolved == target.resolve()
    assert resolved.is_dir()
