"""Path-safe access to the files the Nonlinear and Test data pages show: run outputs in the run store and
committed media in the repository. Used by ``feacore.ui.nonlinear_api`` and ``feacore.ui.testdata_api``.

A *ref* names a file or a directory without handing out arbitrary paths:

    @runs/<path>    inside the run store, under one of STORE_DIRS
    <path>          inside the repository, under one of COMMITTED_DIRS

A ref never contains ``..`` or a backslash, is never absolute, and must still resolve (symlinks followed)
inside its root. Files are served only with a known media type (MEDIA_TYPES), never as HTML or SVG, so a
file in a run directory cannot run script in the dashboard's origin.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# committed directories whose media the dashboard may show (figures, videos, test data, printable parts)
COMMITTED_DIRS = ("docs/gallery", "talk_plots/act3_real_data", "physical_tests", "results/real_data_finetune",
                  "experiments/mesh_systems", "experiments/material", "experiments/physical")
# run-store subdirectories with nonlinear runs, meshes and uploaded test data
STORE_DIRS = ("nonlinear", "ccx_ramp", "testdata")

TEXT = "text/plain; charset=utf-8"
MEDIA_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp",
    ".mp4": "video/mp4", ".webm": "video/webm", ".pdf": "application/pdf",
    ".csv": TEXT, ".txt": TEXT, ".log": TEXT, ".md": TEXT, ".inp": TEXT, ".tsv": TEXT, ".dat": TEXT,
    ".json": "application/json",
    ".step": "application/step", ".stp": "application/step", ".stl": "model/stl",
    ".msh": "application/octet-stream", ".vtu": "application/octet-stream",
}
KINDS = {".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image", ".webp": "image",
         ".mp4": "video", ".webm": "video", ".pdf": "pdf", ".step": "cad", ".stp": "cad", ".stl": "cad"}


class RefError(ValueError):
    """A ref that is malformed, outside the allowed directories, or missing."""


def kind_of(path: str | Path) -> str:
    """image | video | pdf | cad | data, from the file extension."""
    return KINDS.get(Path(path).suffix.lower(), "data")


def _inside(p: Path, root: Path) -> bool:
    root = root.resolve()
    return p == root or root in p.parents


def resolve_ref(ref: str, store_root: Path, want: str = "any") -> Path:
    """The path a ref names, checked; ``want``: file | dir | any. Raises RefError."""
    if not isinstance(ref, str) or not ref or len(ref) > 1000:
        raise RefError("missing or too long ref")
    if "\\" in ref or "\x00" in ref or ref.startswith(("/", "~")) or ".." in ref.split("/"):
        raise RefError(f"invalid ref {ref!r}")
    if ref.startswith("@runs/"):
        rel = ref[len("@runs/"):].strip("/")
        root, allowed = Path(store_root), STORE_DIRS
    else:
        rel = ref.strip("/")
        root, allowed = REPO_ROOT, COMMITTED_DIRS
    if not any(rel == d or rel.startswith(d + "/") for d in allowed):
        raise RefError(f"{ref!r} is outside the directories the dashboard may read")
    p = (root / rel).resolve()
    if not _inside(p, root) or not any(_inside(p, (root / d).resolve()) for d in allowed if (root / d).exists()):
        raise RefError(f"{ref!r} resolves outside its directory")
    if not p.exists():
        raise RefError(f"{ref!r} does not exist")
    if want == "file" and not p.is_file():
        raise RefError(f"{ref!r} is not a file")
    if want == "dir" and not p.is_dir():
        raise RefError(f"{ref!r} is not a directory")
    return p


def to_ref(path: str | Path, store_root: Path) -> str | None:
    """The ref of a path (None when the dashboard may not show it)."""
    p = Path(path).resolve()
    sr = Path(store_root).resolve()
    if _inside(p, sr):
        rel = p.relative_to(sr).as_posix()
        return f"@runs/{rel}" if any(rel == d or rel.startswith(d + "/") for d in STORE_DIRS) else None
    if _inside(p, REPO_ROOT):
        rel = p.relative_to(REPO_ROOT).as_posix()
        return rel if any(rel == d or rel.startswith(d + "/") for d in COMMITTED_DIRS) else None
    return None


def file_response(ref: str, store_root: Path, download: bool = False):
    """A FileResponse for a ref (Range requests work, so videos seek). Raises RefError."""
    from fastapi.responses import FileResponse

    p = resolve_ref(ref, store_root, "file")
    mt = MEDIA_TYPES.get(p.suffix.lower())
    if mt is None:
        raise RefError(f"{p.suffix or 'this'} files are not served")
    headers = {"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"}
    disp = "attachment" if download else "inline"
    return FileResponse(p, media_type=mt, headers=headers, content_disposition_type=disp, filename=p.name)


def media_files(d: Path, store_root: Path, limit: int = 200, skip_dirs=("ccx_work", "series", "harness")) -> list[dict]:
    """Images, videos and PDFs under a directory (two levels down), newest first; render frame folders
    (``*_frames``) and solver scratch are skipped."""
    out = []
    d = Path(d)
    for root, dirs, files in os.walk(d):
        depth = len(Path(root).relative_to(d).parts)
        dirs[:] = [x for x in dirs if depth < 2 and x not in skip_dirs and not x.endswith("_frames")
                   and not x.startswith(".")]
        for f in files:
            p = Path(root) / f
            k = kind_of(p)
            if k not in ("image", "video", "pdf"):
                continue
            ref = to_ref(p, store_root)
            if ref is None:
                continue
            st = p.stat()
            out.append({"ref": ref, "name": p.relative_to(d).as_posix(), "kind": k, "bytes": st.st_size,
                        "mtime": st.st_mtime})
    out.sort(key=lambda m: -m["mtime"])
    return out[:limit]
