"""Select the smallest ModelScope policy that passes the reproduction gates."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from recursive_agent_training.admission import audit_training_round, latest_round_dir
from recursive_agent_training.config import TrainingConfig


MODEL_IDENTITY_FIELDS = (
    "name",
    "path",
    "source",
    "model_id",
    "revision",
    "tokenizer_revision",
    "dtype",
    "use_lora",
    "lora_rank",
    "lora_alpha",
)


def model_slug(config: TrainingConfig) -> str:
    value = config.model.model_id or config.model.name
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def inspect_candidate_run(
    config_path: str | Path,
    run_dir: str | Path,
    *,
    max_gpu_memory_mib: float,
) -> dict[str, Any]:
    config = TrainingConfig.from_yaml(config_path)
    directory = Path(run_dir).expanduser().resolve()
    result: dict[str, Any] = {
        "config": str(Path(config_path).expanduser().resolve()),
        "run_dir": str(directory),
        "model_id": config.model.model_id,
        "model_name": config.model.name,
        "accepted": False,
        "failures": [],
    }
    try:
        round_dir = latest_round_dir(directory)
        admission = audit_training_round(round_dir)
        round_manifest = json.loads(
            (round_dir / "round_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result["failures"].append(f"run evidence unavailable: {exc}")
        return result

    peak = float(
        round_manifest.get("metrics", {}).get("gpu_max_memory_mib") or 0.0
    )
    result.update(
        {
            "round_dir": str(round_dir),
            "policy_version": round_manifest.get("policy_version_after"),
            "gpu_max_memory_mib": peak,
            "admission": admission,
        }
    )
    result["failures"].extend(admission["failures"])
    if peak <= 0:
        result["failures"].append("GPU peak memory metric is missing")
    elif peak > max_gpu_memory_mib:
        result["failures"].append(
            f"GPU peak memory {peak:.1f} MiB exceeds {max_gpu_memory_mib:.1f} MiB"
        )
    result["accepted"] = not result["failures"]
    return result


def select_first_accepted(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((item for item in results if item.get("accepted")), None)


def freeze_selected_s1_config(
    candidate_config_path: str | Path,
    s1_template_path: str | Path,
    output_path: str | Path,
    *,
    checkpoint_output: str | Path,
    action_profile: str = "codeact",
) -> TrainingConfig:
    candidate = TrainingConfig.from_yaml(candidate_config_path)
    template = TrainingConfig.from_yaml(s1_template_path)
    payload = template.model_dump(mode="json", by_alias=True)
    candidate_model = candidate.model.model_dump(mode="json")
    for field in MODEL_IDENTITY_FIELDS:
        payload["model"][field] = candidate_model[field]
    payload["experiment"]["name"] = f"{model_slug(candidate)}_selected_s1"
    payload["experiment"]["action_profile"] = action_profile
    payload["checkpoint"]["output_dir"] = str(
        Path(checkpoint_output).expanduser().resolve()
    )
    payload["parameter_sources"]["model"] = "modelscope_admission_selected"
    selected = TrainingConfig.model_validate(payload)

    import yaml

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(
            selected.model_dump(mode="json", by_alias=True),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return selected


def freeze_selected_codeact_s0_config(
    candidate_config_path: str | Path,
    output_path: str | Path,
    *,
    checkpoint_output: str | Path,
) -> TrainingConfig:
    candidate = TrainingConfig.from_yaml(candidate_config_path)
    payload = candidate.model_dump(mode="json", by_alias=True)
    payload["experiment"]["name"] = f"{model_slug(candidate)}_selected_codeact_s0"
    payload["experiment"]["action_profile"] = "codeact"
    payload["checkpoint"]["output_dir"] = str(
        Path(checkpoint_output).expanduser().resolve()
    )
    payload["parameter_sources"]["model"] = "modelscope_admission_selected"
    payload["parameter_sources"]["rollout"] = "hardware_adapted_codeact_s0"
    selected = TrainingConfig.model_validate(payload)

    import yaml

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(
            selected.model_dump(mode="json", by_alias=True),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return selected
