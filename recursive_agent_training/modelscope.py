"""ModelScope snapshot inspection and reproducibility manifests."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from recursive_agent_training.schemas import SchemaModel, utc_now_iso, write_json


class ModelScopeSnapshotManifest(SchemaModel):
    model_id: str
    requested_revision: str
    resolved_revision: str = ""
    local_path: str
    snapshot_hash: str
    file_hashes: dict[str, str] = Field(default_factory=dict)
    modelscope_version: str = ""
    download_seconds: float | None = None
    inspected_at: str = Field(default_factory=utc_now_iso)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_modelscope_snapshot(
    path: str | Path,
    *,
    model_id: str,
    revision: str = "master",
    modelscope_version: str = "",
) -> ModelScopeSnapshotManifest:
    if not modelscope_version:
        try:
            import modelscope
        except ImportError:
            pass
        else:
            modelscope_version = str(getattr(modelscope, "__version__", ""))
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    required = {"config.json", "tokenizer_config.json"}
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise ValueError(f"invalid model snapshot; missing: {', '.join(missing)}")
    files = sorted(item for item in root.rglob("*") if item.is_file() and item.name != ".msc")
    file_hashes = {item.relative_to(root).as_posix(): _sha256_file(item) for item in files}
    digest = hashlib.sha256()
    for relative, checksum in file_hashes.items():
        digest.update(relative.encode("utf-8"))
        digest.update(checksum.encode("ascii"))
    resolved_revision = revision
    marker = root / ".mv"
    if marker.exists():
        for part in marker.read_text(encoding="utf-8", errors="replace").split(","):
            key, separator, value = part.partition(":")
            if separator and key.strip() == "Revision" and value.strip():
                resolved_revision = value.strip()
    return ModelScopeSnapshotManifest(
        model_id=model_id,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        local_path=str(root),
        snapshot_hash=digest.hexdigest(),
        file_hashes=file_hashes,
        modelscope_version=modelscope_version,
    )


def download_modelscope_snapshot(
    model_id: str,
    *,
    revision: str = "master",
    cache_dir: str | Path | None = None,
) -> ModelScopeSnapshotManifest:
    try:
        import modelscope
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError("ModelScope download requires the 'modelscope' package") from exc
    kwargs: dict[str, Any] = {"revision": revision}
    if cache_dir:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser())
    started = time.monotonic()
    local_path = snapshot_download(model_id, **kwargs)
    manifest = inspect_modelscope_snapshot(
        local_path,
        model_id=model_id,
        revision=revision,
        modelscope_version=str(getattr(modelscope, "__version__", "")),
    )
    payload = manifest.model_dump(mode="json")
    payload["download_seconds"] = time.monotonic() - started
    return ModelScopeSnapshotManifest.model_validate(payload)


def write_modelscope_manifest(
    path: str | Path,
    manifest: ModelScopeSnapshotManifest,
) -> None:
    write_json(path, manifest)
