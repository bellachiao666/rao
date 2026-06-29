"""Runtime environment manifests for reproducible experiments."""

from __future__ import annotations

import hashlib
import importlib
import platform
from pathlib import Path
import sys
from typing import Any


def _version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return ""
    return str(getattr(module, "__version__", ""))


def collect_environment_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "packages": {
            name: _version(name)
            for name in (
                "torch",
                "transformers",
                "accelerate",
                "peft",
                "modelscope",
                "pydantic",
                "yaml",
            )
        },
        "gpus": [],
    }
    try:
        import torch
    except ImportError:
        return manifest
    manifest["cuda_available"] = torch.cuda.is_available()
    manifest["cuda_version"] = str(torch.version.cuda or "")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            manifest["gpus"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_mib": round(properties.total_memory / 1024 / 1024, 1),
                    "capability": [properties.major, properties.minor],
                }
            )
    return manifest


def collect_source_manifest(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    candidates = [
        root / "pyproject.toml",
        root / "requirements.txt",
        *sorted((root / "recursive_agent_harness").rglob("*.py")),
        *sorted((root / "recursive_agent_training").rglob("*.py")),
        *sorted((root / "configs/train").glob("*.yaml")),
        *sorted((root / "examples").glob("*.py")),
    ]
    files: dict[str, str] = {}
    for path in candidates:
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for relative, checksum in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(checksum.encode("ascii"))
    return {
        "project_root": str(root),
        "source_hash": digest.hexdigest(),
        "file_count": len(files),
        "file_hashes": files,
    }
