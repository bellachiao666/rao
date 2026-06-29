#!/usr/bin/env python3
"""Fast static/environment audit for the RAO paper reproduction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parent
EXPECTED_DEEPDIVE_HASH = "126047370f723bd0d665a0ffbe4ff285069a3cabb220afce5657665c02eed2aa"
DEFAULT_ADAPTED_CONFIG = ROOT / "configs/train/deepdive_qwen06b_s1.yaml"


def source(path: str) -> str:
    target = ROOT / path
    return target.read_text(encoding="utf-8") if target.exists() else ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def env_value(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def gpu_count() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return 0
    return len([line for line in result.stdout.splitlines() if line.strip()])


def check(
    check_id: str,
    status: str,
    category: str,
    detail: str,
    next_action: str = "",
) -> dict[str, str]:
    return {
        "id": check_id,
        "status": status,
        "category": category,
        "detail": detail,
        "next_action": "" if status == "READY" else next_action,
    }


def inspect_repository() -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    adapted_config_path = Path(env_value("RAO_CONFIG") or DEFAULT_ADAPTED_CONFIG)
    try:
        import yaml

        adapted_config = yaml.safe_load(adapted_config_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, ImportError):
        adapted_config = {}

    algorithm_files = [
        "recursive_agent_training/rewards.py",
        "recursive_agent_training/advantages.py",
        "recursive_agent_training/weighting.py",
        "recursive_agent_training/objectives/cispo.py",
        "recursive_agent_training/compilation.py",
    ]
    missing = [path for path in algorithm_files if not (ROOT / path).exists()]
    checks.append(
        check(
            "algorithm_core",
            "READY" if not missing else "BLOCKED",
            "implementation",
            "RAO equation components are present." if not missing else f"Missing: {', '.join(missing)}",
            "Restore and test all equation components." if missing else "",
        )
    )

    flywheel_text = source("recursive_agent_training/flywheel.py")
    flywheel_ready = all(
        marker in flywheel_text
        for marker in ("RoundManifest", "LatestPublication", "atomic_write_directory", "resume")
    )
    checks.append(
        check(
            "synthetic_flywheel",
            "READY" if flywheel_ready else "BLOCKED",
            "implementation",
            "Atomic round publication and resume code are present."
            if flywheel_ready
            else "Round publication or resume primitives are missing.",
        )
    )

    admission_sources = "\n".join(
        [
            source("recursive_agent_training/admission.py"),
            source("recursive_agent_training/model_admission.py"),
            source("recursive_agent_training/flywheel.py"),
            source("examples/run_model_admission.py"),
        ]
    )
    admission_ready = all(
        marker in admission_sources
        for marker in (
            "audit_training_round",
            "UntrustedRewardBatchError",
            "select_first_accepted",
            "freeze_selected_s1_config",
        )
    )
    checks.append(
        check(
            "model_and_reward_admission",
            "READY" if admission_ready else "BLOCKED",
            "implementation",
            "Model selection and pre-optimizer reward admission are enforced."
            if admission_ready
            else "Model admission or pre-optimizer reward auditing is missing.",
            "Add candidate selection, reward-source auditing, and frozen S1 output."
            if not admission_ready
            else "",
        )
    )

    data_path = Path(env_value("RAO_DATA_PATH") or ROOT / "data/deepdive_qa_rl.csv")
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists():
        data_status = "BLOCKED"
        data_detail = f"Dataset not found: {data_path}"
    else:
        actual_hash = sha256(data_path)
        data_status = "READY" if actual_hash == EXPECTED_DEEPDIVE_HASH else "BLOCKED"
        data_detail = (
            f"DeepDive dataset hash matches: {actual_hash}"
            if data_status == "READY"
            else f"Dataset hash mismatch: {actual_hash}"
        )
    checks.append(
        check(
            "deepdive_dataset",
            data_status,
            "environment",
            data_detail,
            "Provide the paper dataset with the recorded SHA-256." if data_status != "READY" else "",
        )
    )

    split_path = Path(
        env_value("RAO_SPLIT_MANIFEST")
        or ROOT / "output/train/deepdive_split.json"
    )
    if not split_path.is_absolute():
        split_path = ROOT / split_path
    split_ready = False
    split_detail = f"Split manifest not found: {split_path}"
    if split_path.exists():
        try:
            split = json.loads(split_path.read_text(encoding="utf-8"))
            split_ready = (
                split.get("dataset_hash") == EXPECTED_DEEPDIVE_HASH
                and len(split.get("eval_task_ids", [])) == 50
                and bool(split.get("train_task_ids"))
            )
            split_detail = (
                "Fixed 50-task split manifest is valid."
                if split_ready
                else "Split exists but dataset hash, train IDs, or 50-task eval IDs are invalid."
            )
        except (OSError, json.JSONDecodeError):
            split_detail = "Split manifest is unreadable or invalid JSON."
    checks.append(
        check(
            "deepdive_split",
            "READY" if split_ready else "BLOCKED",
            "environment",
            split_detail,
            "Run examples/prepare_deepdive_split.py and preserve the manifest."
            if not split_ready
            else "",
        )
    )

    deepdive_runner = source("examples/run_deepdive_rao_training.py")
    runner_markers = (
        "build_deepdive_tools",
        "build_judge_verifier",
        "SyncFlywheelRunner",
        "HuggingFaceRuntimeFactory",
    )
    runner_ready = all(marker in deepdive_runner for marker in runner_markers)
    runner_blocked = "raise RuntimeError" in deepdive_runner
    checks.append(
        check(
            "deepdive_training_entrypoint",
            "READY" if runner_ready and not runner_blocked else "BLOCKED",
            "implementation",
            "DeepDive has a real training loop."
            if runner_ready and not runner_blocked
            else "DeepDive command is still preflight-only.",
            "Wire dataset, tools, judge, rollout, compilation, optimizer, and publication.",
        )
    )

    runtime_sources = "\n".join(
        [
            source("recursive_agent_training/codeact/environment.py"),
            source("recursive_agent_training/codeact/policy.py"),
            source("recursive_agent_training/hf_flywheel.py"),
        ]
    )
    codeact_ready = (
        "HuggingFaceCodeActTrainingPolicy" in runtime_sources
        and "CodeActEnvironment" in runtime_sources
        and "launch_subagent" in runtime_sources
        and "asyncio" in runtime_sources
    )
    checks.append(
        check(
            "hardware_adapted_codeact_runtime",
            "READY" if codeact_ready else "BLOCKED",
            "implementation",
            "Hardware-adapted restricted CodeAct runtime is wired."
            if codeact_ready
            else "CodeAct prototypes are not connected to the DeepDive rollout path.",
            "Implement the persistent restricted Python REPL and subagent action loop.",
        )
    )

    deepdive_tools = source("recursive_agent_training/domains/deepdive.py")
    judge_verifier = source("recursive_agent_training/verifiers/llm.py")
    service_basics = (
        "DeepDiveWebServices" in deepdive_tools
        and "LocalTransformersJudgeVerifier" in judge_verifier
    )
    service_controls = all(
        marker in (deepdive_tools + judge_verifier)
        for marker in ("max_retries", "RateLimiter", "JsonFileCache", "ServiceMetrics")
    )
    service_status = "READY" if service_basics and service_controls else (
        "PARTIAL" if service_basics else "BLOCKED"
    )
    checks.append(
        check(
            "search_judge_implementation",
            service_status,
            "implementation",
            "Search and judge adapters exist, but production retry/rate/cache controls are incomplete."
            if service_status == "PARTIAL"
            else "Search and judge service controls are present."
            if service_status == "READY"
            else "Search or judge adapter is missing.",
            "Add bounded retries, rate limits, caching, strict schemas, and metrics."
            if service_status != "READY"
            else "",
        )
    )

    search_config = adapted_config.get("services", {}).get("search", {})
    judge_config = adapted_config.get("services", {}).get("judge", {})
    local_judge_path = Path(str(judge_config.get("model_path", "")))
    credentialless_services = (
        search_config.get("provider") == "duckduckgo"
        and judge_config.get("provider") == "local_transformers"
        and local_judge_path.exists()
    )
    credentials_ready = credentialless_services or all(
        [
            env_value("RAO_SEARCH_ENDPOINT", "TAVILY_SEARCH_ENDPOINT"),
            env_value("TAVILY_API_KEY"),
            env_value("RAO_JUDGE_BASE_URL", "JUDGE_BASE_URL", "OPENAI_BASE_URL"),
            env_value("RAO_JUDGE_API_KEY", "JUDGE_API_KEY", "OPENAI_API_KEY"),
            env_value("RAO_JUDGE_MODEL", "JUDGE_MODEL"),
        ]
    )
    checks.append(
        check(
            "search_judge_credentials",
            "READY" if credentials_ready else "BLOCKED",
            "environment",
            "Credentialless DuckDuckGo search and local ModelScope judge are available."
            if credentialless_services
            else "Search and judge endpoints, keys, and model are configured."
            if credentials_ready
            else "The configured local judge snapshot is unavailable in this environment.",
            "Provide the configured local judge snapshot or service credentials.",
        )
    )

    areal_adapter = source("recursive_agent_training/optimizers/areal_optimizer.py")
    required_areal_methods = (
        "compute_logprobs",
        "load_checkpoint",
        "publish_snapshot",
    )
    areal_impl_ready = all(method in areal_adapter for method in required_areal_methods)
    checks.append(
        check(
            "distributed_areal_implementation",
            "READY" if areal_impl_ready else "BLOCKED",
            "paper_fidelity",
            "AReaL backend exposes the required distributed lifecycle."
            if areal_impl_ready
            else "AReaL adapter is still a thin optimizer wrapper.",
            "Implement workers, transport, logprobs, load/resume, versioning, and publication.",
        )
    )
    checks.append(
        check(
            "areal_installation",
            "READY" if module_available("areal") else "BLOCKED",
            "paper_fidelity",
            "AReaL is importable." if module_available("areal") else "AReaL is not importable.",
            "Install the pinned AReaL cluster environment.",
        )
    )

    hf_policy = source("recursive_agent_training/hf_policy.py")
    hf_runtime = source("recursive_agent_training/hf_flywheel.py")
    long_context_ready = all(
        marker in (hf_policy + hf_runtime)
        for marker in ("max_input_tokens", "prompt_truncated", "action_truncated")
    )
    checks.append(
        check(
            "long_context_runtime",
            "READY" if long_context_ready else "BLOCKED",
            "implementation",
            "Hardware-adapted 8K/16K context limits and action truncation rejection are enforced."
            if long_context_ready
            else "Configured context limits are not enforced by the runtime.",
            "Implement and test truncation plus exact action-token preservation.",
        )
    )

    evaluator_path = ROOT / "examples/run_deepdive_evaluation.py"
    evaluator = (
        evaluator_path.read_text(encoding="utf-8")
        + source("recursive_agent_training/evaluation.py")
        if evaluator_path.exists()
        else ""
    )
    evaluator_ready = all(
        marker in evaluator.lower()
        for marker in ("pass_at_8", "bootstrap", "per_task", "split")
    )
    checks.append(
        check(
            "heldout_evaluation",
            "READY" if evaluator_ready else "BLOCKED",
            "implementation",
            "Held-out evaluator produces pass@8 and confidence intervals."
            if evaluator_ready
            else "No complete DeepDive held-out rollout evaluator exists.",
            "Add examples/run_deepdive_evaluation.py with fixed-split per-task outputs.",
        )
    )

    matrix_path = ROOT / "examples/run_reproduction_matrix.py"
    matrix = matrix_path.read_text(encoding="utf-8") if matrix_path.exists() else ""
    matrix_ready = all(
        marker in matrix.lower()
        for marker in ("base", "trained", "single", "recursive", "ablation")
    )
    checks.append(
        check(
            "comparison_and_ablation_matrix",
            "READY" if matrix_ready else "BLOCKED",
            "implementation",
            "Four-way comparison and ablations are orchestrated."
            if matrix_ready
            else "No executable comparison/ablation matrix exists.",
            "Add examples/run_reproduction_matrix.py with strict control matching.",
        )
    )

    model_path_value = env_value("RAO_MODEL_PATH") or str(
        adapted_config.get("model", {}).get("path", "")
    )
    model_ready = bool(model_path_value and Path(model_path_value).exists())
    checks.append(
        check(
            "modelscope_qwen_model",
            "READY" if model_ready else "BLOCKED",
            "environment",
            f"Model path exists: {model_path_value}"
            if model_ready
            else "The configured ModelScope Qwen snapshot is unavailable.",
            "Set RAO_MODEL_PATH or model.path to the pinned ModelScope snapshot.",
        )
    )

    detected_gpus = gpu_count()
    checks.append(
        check(
            "eight_gpu_environment",
            "READY" if detected_gpus >= 8 else "BLOCKED",
            "environment",
            f"Detected {detected_gpus} NVIDIA GPU(s).",
            "Run the audit on the eight-GPU training host.",
        )
    )

    output_value = env_value("RAO_REPRO_OUTPUT")
    output_path = Path(output_value) if output_value else Path()
    evidence_ready = False
    if output_value and output_path.exists():
        try:
            run_manifest = json.loads(
                (output_path / "training/dense_weighted/run_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            evidence_ready = (
                int(run_manifest.get("completed_rounds", 0)) >= 75
                and (output_path / "report/joined_results.json").exists()
                and (output_path / "report/implementation_differences.json").exists()
            )
        except (OSError, ValueError, json.JSONDecodeError):
            evidence_ready = False
    checks.append(
        check(
            "hardware_adapted_run_evidence",
            "READY" if evidence_ready else "BLOCKED",
            "evidence",
            f"Evidence package found at {output_path}."
            if evidence_ready
            else "No complete hardware-adapted 75-step evidence package was found.",
            "Complete training, evaluation, comparisons, and archive the required manifests.",
        )
    )
    return checks


def summarize(checks: list[dict[str, str]]) -> dict[str, Any]:
    counts = {
        status: sum(item["status"] == status for item in checks)
        for status in ("READY", "PARTIAL", "BLOCKED")
    }
    implementation_blockers = [
        item["id"]
        for item in checks
        if item["category"] == "implementation" and item["status"] != "READY"
    ]
    required_checks = [
        item
        for item in checks
        if item["category"] in {"implementation", "environment", "evidence"}
    ]
    if implementation_blockers:
        claim_level = "L1_SYNTHETIC_FLYWHEEL"
        overall = "BLOCKED"
    elif any(item["status"] != "READY" for item in required_checks):
        claim_level = "L2_DEEPDIVE_SHORT_RUN"
        overall = "ENVIRONMENT_BLOCKED"
    else:
        claim_level = "L4_HARDWARE_ADAPTED_METHOD"
        overall = "READY"
    return {
        "overall": overall,
        "current_claim_level": claim_level,
        "counts": counts,
        "implementation_blockers": implementation_blockers,
        "checks": checks,
    }


def print_text(report: dict[str, Any]) -> None:
    print(f"Overall: {report['overall']}")
    print(f"Claim level: {report['current_claim_level']}")
    counts = report["counts"]
    print(
        f"Checks: {counts['READY']} ready, {counts['PARTIAL']} partial, "
        f"{counts['BLOCKED']} blocked"
    )
    print()
    for category in ("implementation", "environment", "evidence", "paper_fidelity"):
        print(f"[{category}]")
        for item in report["checks"]:
            if item["category"] != category:
                continue
            print(f"{item['status']:7} {item['id']}: {item['detail']}")
            if item["status"] != "READY" and item["next_action"]:
                print(f"        next: {item['next_action']}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless every implementation, environment, and evidence check is ready.",
    )
    args = parser.parse_args()
    report = summarize(inspect_repository())
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_text(report)
    if args.strict and report["overall"] != "READY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
