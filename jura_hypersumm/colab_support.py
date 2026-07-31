"""Thin, lazily imported Google Colab integration helpers."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence


def is_colab() -> bool:
    """Return whether the current interpreter is running in Google Colab."""
    try:
        import google.colab  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


def require_colab() -> None:
    """Raise a clear error when a Colab-only workflow runs elsewhere."""
    if not is_colab():
        raise RuntimeError(
            "This workflow must run in Google Colab because it mounts Drive "
            "and asks the user to upload DOCX files."
        )


def get_huggingface_token() -> str | None:
    """Read an HF token from the environment or Colab secrets without printing it."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from google.colab import userdata

        return userdata.get("HF_TOKEN")
    except Exception:
        # Colab uses dedicated exception classes when a secret is missing or
        # the notebook has not been granted access. Non-gated models can still
        # run without a token.
        return None


def mount_drive(drive_root: str | Path) -> Path:
    """Mount Google Drive and return the configured existing project root."""
    require_colab()
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
    root = Path(drive_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Expected Drive project folder does not exist: {root}"
        )
    return root


def prepare_artifact_root(root: str | Path) -> Path:
    """Mount Drive in Colab or create and return an explicit local root."""
    if is_colab():
        return mount_drive(root)
    local_root = Path(root).expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    return local_root


@contextmanager
def uploaded_docx_files() -> Iterator[list[Path]]:
    """Upload DOCX files into a temporary directory and always delete them."""
    require_colab()
    from google.colab import files

    temp_parent = Path("/content") if Path("/content").is_dir() else None
    temporary = Path(
        tempfile.mkdtemp(prefix="jura_docx_", dir=str(temp_parent) if temp_parent else None)
    )
    previous_directory = Path.cwd()
    try:
        os.chdir(temporary)
        uploaded = files.upload()
        paths = []
        for uploaded_name in uploaded:
            safe_name = Path(uploaded_name).name
            path = temporary / safe_name
            if path.suffix.lower() == ".docx" and path.is_file():
                paths.append(path)
        yield sorted(paths, key=lambda item: item.name.lower())
    finally:
        os.chdir(previous_directory)
        shutil.rmtree(temporary, ignore_errors=True)


@contextmanager
def selected_docx_files(
    document_paths: Sequence[str | Path] | None = None,
) -> Iterator[list[Path]]:
    """Use explicit local DOCX paths or Colab's temporary upload interface.

    Explicit files belong to the caller and are never removed. Files uploaded
    through Colab retain the existing temporary-file cleanup behavior.
    """
    if document_paths is None:
        if is_colab():
            with uploaded_docx_files() as uploaded:
                yield uploaded
        else:
            yield []
        return
    paths = [Path(path).expanduser().resolve() for path in document_paths]
    invalid = [
        path
        for path in paths
        if path.suffix.lower() != ".docx" or not path.is_file()
    ]
    if invalid:
        raise FileNotFoundError(
            "Invalid or missing DOCX input(s): " + ", ".join(map(str, invalid))
        )
    yield sorted(paths, key=lambda item: item.name.lower())


def download_file(path: str | Path) -> None:
    """Download one generated artifact from Colab."""
    require_colab()
    from google.colab import files

    files.download(str(path))


def deliver_file(path: str | Path) -> None:
    """Download an artifact in Colab or report its resolved local path."""
    if is_colab():
        download_file(path)
    else:
        print(f"[JURA][results][LOCAL] Saved artifact: {Path(path).resolve()}", flush=True)
