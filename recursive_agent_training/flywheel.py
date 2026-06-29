"""Synchronous, round-based rollout and training orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from recursive_agent_harness.tools import ToolRegistry
from recursive_agent_training.admission import audit_training_round
from recursive_agent_training.compilation import compile_optimizer_batch, verify_and_score_group
from recursive_agent_training.config import TrainingConfig
from recursive_agent_training.grouping import PolicyFactory, collect_rollout_group
from recursive_agent_training.optimizers.base import PolicyOptimizer
from recursive_agent_training.schemas import (
    CompiledBatch,
    RolloutGroupBundle,
    RootTaskRecord,
    SchemaModel,
    utc_now_iso,
    write_json,
)
from recursive_agent_training.snapshots import PolicySnapshot
from recursive_agent_training.verifiers.base import SuccessSignalProvider


class RoundStatus(str, Enum):
    PENDING = "pending"
    COLLECTING = "collecting"
    FROZEN = "frozen"
    TRAINING = "training"
    PUBLISHED = "published"
    FAILED = "failed"


class ZeroAdvantageBatchError(ValueError):
    pass


class UntrustedRewardBatchError(ZeroAdvantageBatchError):
    pass


def _is_missing_reward_error(exc: ValueError) -> bool:
    message = str(exc)
    return message.startswith("root reward missing for ") or message.startswith(
        "node reward missing for "
    )


class RoundManifest(SchemaModel):
    run_id: str
    round_index: int
    status: RoundStatus = RoundStatus.PENDING
    task_ids: list[str] = Field(default_factory=list)
    policy_version_before: str
    policy_checkpoint_before: str = ""
    policy_checkpoint_hash_before: str = ""
    policy_version_after: str = ""
    raw_group_paths: list[str] = Field(default_factory=list)
    scored_group_paths: list[str] = Field(default_factory=list)
    optimizer_batch_path: str = ""
    optimizer_batch_hash: str = ""
    adapter_path: str = ""
    adapter_hash: str = ""
    optimizer_path: str = ""
    optimizer_hash: str = ""
    config_hash: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class LatestPublication(SchemaModel):
    run_id: str
    round_index: int
    policy_version: str
    adapter_path: str
    adapter_hash: str
    optimizer_path: str
    optimizer_hash: str
    config_hash: str
    published_at: str = Field(default_factory=utc_now_iso)


class RunManifest(SchemaModel):
    run_id: str
    model_name: str
    initial_checkpoint_path: str
    config_hash: str
    target_rounds: int
    completed_rounds: int = 0
    status: str = "running"
    latest_policy_version: str = "policy_000000"
    failure_reason: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


@dataclass
class RoundRuntime:
    policy_factory: PolicyFactory
    optimizer: PolicyOptimizer
    save_adapter: Callable[[Path], None]
    save_optimizer: Callable[[Path], None]
    runtime_metrics: Callable[[], dict[str, Any]] = lambda: {}
    close: Callable[[], None] = lambda: None


RuntimeFactory = Callable[[PolicySnapshot, LatestPublication | None], RoundRuntime]
VerifierFactory = Callable[[], SuccessSignalProvider]
ToolFactory = Callable[[], ToolRegistry]


def sha256_path(path: str | Path) -> str:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    if source.is_file():
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    for item in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = item.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def atomic_write_model(path: str | Path, value: BaseModel | dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, target)


def atomic_write_directory(path: str | Path, writer: Callable[[Path], None]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    if target.exists():
        raise FileExistsError(f"refusing to replace published directory: {target}")
    temporary.mkdir(parents=True)
    try:
        writer(temporary)
        if not any(temporary.iterdir()):
            raise ValueError(f"artifact writer produced an empty directory: {temporary}")
        os.replace(temporary, target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def parse_policy_version(version: str) -> int:
    prefix, separator, raw = version.rpartition("_")
    if not separator or prefix != "policy":
        raise ValueError(f"invalid policy version: {version}")
    return int(raw)


class SyncFlywheelRunner:
    def __init__(
        self,
        *,
        config: TrainingConfig,
        tasks: list[RootTaskRecord],
        output_dir: str | Path,
        model_name: str,
        initial_checkpoint_path: str,
        runtime_factory: RuntimeFactory,
        verifier_factory: VerifierFactory,
        tools_factory: ToolFactory | None = None,
    ):
        self.config = config
        self.tasks = tasks
        self.output_dir = Path(output_dir).resolve()
        self.model_name = model_name
        self.initial_checkpoint_path = str(Path(initial_checkpoint_path).resolve())
        self.runtime_factory = runtime_factory
        self.verifier_factory = verifier_factory
        self.tools_factory = tools_factory
        required = config.sync_flywheel.root_tasks_per_round
        if len(tasks) < required:
            raise ValueError(f"sync flywheel requires at least {required} distinct root tasks")
        if len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("sync flywheel root task ids must be unique")

    async def run(self, target_rounds: int, *, resume: bool = False) -> list[RoundManifest]:
        if target_rounds <= 0:
            raise ValueError("target_rounds must be positive")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        latest = self._load_latest()
        latest = self._recover_unindexed_publications(latest)
        run_manifest = self._load_or_create_run_manifest(target_rounds, resume, latest)
        start_round = latest.round_index + 1 if latest else 0
        if target_rounds < start_round:
            raise ValueError(
                f"target round count {target_rounds} is behind published round count {start_round}"
            )
        if target_rounds == start_round:
            run_manifest.completed_rounds = start_round
            run_manifest.latest_policy_version = (
                latest.policy_version if latest else "policy_000000"
            )
            run_manifest.status = "completed"
            run_manifest.failure_reason = None
            run_manifest.updated_at = utc_now_iso()
            atomic_write_model(self.output_dir / "run_manifest.json", run_manifest)
            return []
        completed: list[RoundManifest] = []
        for round_index in range(start_round, target_rounds):
            discarded_attempt_paths: list[str] = []
            manifest = None
            snapshot = self._snapshot_for_round(round_index, latest)
            runtime = self.runtime_factory(snapshot, latest)
            verifier: SuccessSignalProvider | None = None
            try:
                verifier = self.verifier_factory()
                tools = self.tools_factory() if self.tools_factory else None
                for collection_attempt in range(
                    self.config.sync_flywheel.max_collection_attempts
                ):
                    try:
                        manifest = await self._run_round(
                            run_manifest,
                            round_index,
                            latest,
                            runtime=runtime,
                            verifier=verifier,
                            tools=tools,
                            collection_attempt=collection_attempt,
                            discarded_attempt_paths=discarded_attempt_paths,
                        )
                        break
                    except ZeroAdvantageBatchError:
                        if (
                            collection_attempt + 1
                            >= self.config.sync_flywheel.max_collection_attempts
                        ):
                            raise
            finally:
                try:
                    close_verifier = getattr(verifier, "close", None)
                    if callable(close_verifier):
                        close_verifier()
                finally:
                    runtime.close()
            if manifest is None:
                raise RuntimeError("round collection ended without a manifest")
            completed.append(manifest)
            latest = self._load_latest(required=True)
            run_manifest.completed_rounds = round_index + 1
            run_manifest.latest_policy_version = latest.policy_version
            run_manifest.status = "completed" if round_index + 1 == target_rounds else "running"
            run_manifest.failure_reason = None
            run_manifest.updated_at = utc_now_iso()
            atomic_write_model(self.output_dir / "run_manifest.json", run_manifest)
        return completed

    async def _run_round(
        self,
        run_manifest: RunManifest,
        round_index: int,
        latest: LatestPublication | None,
        *,
        runtime: RoundRuntime,
        verifier: SuccessSignalProvider,
        tools: ToolRegistry | None,
        collection_attempt: int = 0,
        discarded_attempt_paths: list[str] | None = None,
    ) -> RoundManifest:
        round_dir = self.output_dir / "rounds" / f"round_{round_index:04d}"
        archived = self._archive_incomplete_round(round_dir)
        if discarded_attempt_paths is None:
            discarded_attempt_paths = []
        if archived is not None:
            discarded_attempt_paths.append(str(archived))
        raw_dir = round_dir / "raw_rollouts"
        scored_dir = round_dir / "scored_groups"
        batch_dir = round_dir / "optimizer_batch"
        raw_dir.mkdir(parents=True)
        scored_dir.mkdir()
        batch_dir.mkdir()

        snapshot = self._snapshot_for_round(round_index, latest)
        tasks = self._tasks_for_round(round_index, collection_attempt)
        manifest = RoundManifest(
            run_id=run_manifest.run_id,
            round_index=round_index,
            status=RoundStatus.COLLECTING,
            task_ids=[task.task_id for task in tasks],
            policy_version_before=snapshot.version,
            policy_checkpoint_before=snapshot.checkpoint_path,
            policy_checkpoint_hash_before=snapshot.checkpoint_hash,
            config_hash=self.config.stable_hash(),
            metrics={
                "collection_attempt_count": collection_attempt + 1,
                "zero_advantage_retry_count": collection_attempt,
                "discarded_attempt_paths": discarded_attempt_paths,
            },
        )
        manifest_path = round_dir / "round_manifest.json"
        atomic_write_model(manifest_path, manifest)
        tool_metrics_before = self._service_metrics(tools)
        judge_metrics_before = self._service_metrics(verifier)
        rollout_started = time.monotonic()
        try:
            group_limit = min(
                self.config.rollout.max_concurrent_groups,
                len(tasks),
            )
            if group_limit <= 0:
                raise ValueError("max_concurrent_groups must be positive")
            group_semaphore = asyncio.Semaphore(group_limit)

            async def collect_task(
                task_offset: int,
                task: RootTaskRecord,
            ) -> tuple[int, RolloutGroupBundle, Path]:
                group_sequence = (
                    round_index
                    * self.config.sync_flywheel.max_collection_attempts
                    * len(tasks)
                    + collection_attempt * len(tasks)
                    + task_offset
                )

                def group_policy_factory(index: int, seed: int):
                    global_index = (
                        task_offset * self.config.rollout.group_size + index
                    )
                    return runtime.policy_factory(global_index, seed)

                async with group_semaphore:
                    bundle = await collect_rollout_group(
                        task,
                        snapshot,
                        self.config.rollout.group_size,
                        group_policy_factory,
                        self._harness_config(round_dir),
                        tools=tools,
                        group_sequence=group_sequence,
                        output_dir=raw_dir / "traces" / f"task_{task_offset:04d}",
                    )
                raw_path = raw_dir / f"group_{task_offset:04d}.json"
                write_json(raw_path, bundle)
                return task_offset, bundle, raw_path

            collection_tasks = [
                asyncio.create_task(collect_task(task_offset, task))
                for task_offset, task in enumerate(tasks)
            ]
            try:
                collected = await asyncio.gather(*collection_tasks)
            except BaseException:
                for collection_task in collection_tasks:
                    collection_task.cancel()
                await asyncio.gather(*collection_tasks, return_exceptions=True)
                raise
            collected.sort(key=lambda item: item[0])
            bundles = [item[1] for item in collected]
            manifest.raw_group_paths.extend(str(item[2]) for item in collected)

            manifest.status = RoundStatus.FROZEN
            manifest.metrics["rollout_seconds"] = time.monotonic() - rollout_started
            tool_metrics_after = self._service_metrics(tools)
            if tool_metrics_after:
                manifest.metrics["tool_service"] = self._metric_delta(
                    tool_metrics_after,
                    tool_metrics_before,
                )
            manifest.updated_at = utc_now_iso()
            atomic_write_model(manifest_path, manifest)

            for task_offset, bundle in enumerate(bundles):
                await verify_and_score_group(bundle, verifier, self.config.reward.delegation_lambda)
                scored_path = scored_dir / f"group_{task_offset:04d}.json"
                write_json(scored_path, bundle)
                manifest.scored_group_paths.append(str(scored_path))
            judge_metrics_after = self._service_metrics(verifier)
            if judge_metrics_after:
                manifest.metrics["judge_service"] = self._metric_delta(
                    judge_metrics_after,
                    judge_metrics_before,
                )

            try:
                batch = compile_optimizer_batch(
                    bundles,
                    self.config,
                    optimizer_step=round_index,
                    policy_version_before=snapshot.version,
                )
            except ValueError as exc:
                if "at least one trainable trajectory is required" in str(exc):
                    raise ZeroAdvantageBatchError(
                        "optimizer batch has no trainable trajectories"
                    ) from exc
                if _is_missing_reward_error(exc):
                    raise ZeroAdvantageBatchError(
                        f"optimizer batch has missing verifier rewards: {exc}"
                    ) from exc
                raise
            batch_path = batch_dir / "batch.json"
            write_json(batch_path, batch)
            manifest.optimizer_batch_path = str(batch_path)
            manifest.optimizer_batch_hash = sha256_path(batch_path)
            manifest.metrics.update(self._data_metrics(bundles, batch))
            epsilon = self.config.sync_flywheel.nonzero_advantage_epsilon
            if not any(abs(sample.advantage) > epsilon for sample in batch.samples):
                raise ZeroAdvantageBatchError(
                    "optimizer batch has no non-zero advantages"
                )
            manifest.updated_at = utc_now_iso()
            atomic_write_model(manifest_path, manifest)
            admission = audit_training_round(
                round_dir,
                advantage_epsilon=epsilon,
                require_published=False,
            )
            admission_path = round_dir / "admission_audit.json"
            atomic_write_model(admission_path, admission)
            manifest.metrics["admission_accepted"] = admission["accepted"]
            manifest.metrics["admission_audit_path"] = str(admission_path)
            manifest.metrics["trusted_positive_reward_count"] = admission[
                "trusted_positive_reward_count"
            ]
            if not admission["accepted"]:
                raise UntrustedRewardBatchError(
                    "optimizer batch failed admission: "
                    + "; ".join(admission["failures"])
                )

            manifest.status = RoundStatus.TRAINING
            manifest.updated_at = utc_now_iso()
            atomic_write_model(manifest_path, manifest)
            training_started = time.monotonic()
            optimizer_metrics = runtime.optimizer.step(batch)
            manifest.metrics.update(optimizer_metrics)
            manifest.metrics["training_seconds"] = time.monotonic() - training_started
            self._validate_optimizer_metrics(optimizer_metrics)

            adapter_path = round_dir / "adapter"
            optimizer_path = round_dir / "optimizer.pt"
            atomic_write_directory(adapter_path, runtime.save_adapter)
            runtime.save_optimizer(optimizer_path)
            manifest.adapter_path = str(adapter_path)
            manifest.adapter_hash = sha256_path(adapter_path)
            manifest.optimizer_path = str(optimizer_path)
            manifest.optimizer_hash = sha256_path(optimizer_path)
            manifest.policy_version_after = runtime.optimizer.current_version()
            expected_after = f"policy_{round_index + 1:06d}"
            if manifest.policy_version_after != expected_after:
                raise ValueError(
                    f"optimizer published {manifest.policy_version_after}, expected {expected_after}"
                )
            manifest.metrics.update(runtime.runtime_metrics())
            manifest.status = RoundStatus.PUBLISHED
            manifest.updated_at = utc_now_iso()
            atomic_write_model(manifest_path, manifest)

            publication = LatestPublication(
                run_id=run_manifest.run_id,
                round_index=round_index,
                policy_version=manifest.policy_version_after,
                adapter_path=manifest.adapter_path,
                adapter_hash=manifest.adapter_hash,
                optimizer_path=manifest.optimizer_path,
                optimizer_hash=manifest.optimizer_hash,
                config_hash=manifest.config_hash,
            )
            atomic_write_model(self.output_dir / "latest.json", publication)
            return manifest
        except Exception as exc:
            manifest.status = RoundStatus.FAILED
            manifest.failure_reason = f"{type(exc).__name__}: {exc}"
            manifest.updated_at = utc_now_iso()
            atomic_write_model(manifest_path, manifest)
            run_manifest.status = "failed"
            run_manifest.failure_reason = manifest.failure_reason
            run_manifest.updated_at = utc_now_iso()
            atomic_write_model(self.output_dir / "run_manifest.json", run_manifest)
            raise

    def _snapshot_for_round(
        self,
        round_index: int,
        latest: LatestPublication | None,
    ) -> PolicySnapshot:
        expected = f"policy_{round_index:06d}"
        if latest is None:
            if round_index != 0:
                raise ValueError("missing publication for non-zero round")
            checkpoint = Path(self.initial_checkpoint_path)
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            return PolicySnapshot(
                version=expected,
                model_name=self.model_name,
                checkpoint_path=str(checkpoint),
                template_version="hf_chat_template_v1",
            )
        self._validate_publication(latest)
        if latest.policy_version != expected or latest.round_index != round_index - 1:
            raise ValueError(
                f"latest publication {latest.policy_version} cannot seed round {round_index}"
            )
        return PolicySnapshot(
            version=latest.policy_version,
            model_name=self.model_name,
            checkpoint_path=latest.adapter_path,
            checkpoint_hash=latest.adapter_hash,
            template_version="hf_chat_template_v1",
        )

    def _tasks_for_round(
        self,
        round_index: int,
        collection_attempt: int = 0,
    ) -> list[RootTaskRecord]:
        count = self.config.sync_flywheel.root_tasks_per_round
        start = (
            (
                round_index * self.config.sync_flywheel.max_collection_attempts
                + collection_attempt
            )
            * count
        ) % len(self.tasks)
        return [self.tasks[(start + offset) % len(self.tasks)] for offset in range(count)]

    def _harness_config(self, round_dir: Path):
        from recursive_agent_training.mini import make_harness_config

        return make_harness_config(self.config, str(round_dir))

    def _load_or_create_run_manifest(
        self,
        target_rounds: int,
        resume: bool,
        latest: LatestPublication | None,
    ) -> RunManifest:
        path = self.output_dir / "run_manifest.json"
        if path.exists():
            if not resume:
                raise FileExistsError(f"run already exists; pass --resume: {path}")
            manifest = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
            if not self.config.matches_hash_except_runtime_retry_fields(
                manifest.config_hash
            ):
                raise ValueError("resume config hash does not match the existing run")
            if manifest.initial_checkpoint_path != self.initial_checkpoint_path:
                raise ValueError("resume model path does not match the existing run")
            if latest and latest.run_id != manifest.run_id:
                raise ValueError("latest publication belongs to another run")
            manifest.target_rounds = target_rounds
            manifest.status = "running"
            manifest.failure_reason = None
            manifest.updated_at = utc_now_iso()
            atomic_write_model(path, manifest)
            return manifest
        if resume and latest is not None:
            raise ValueError("latest publication exists without a run manifest")
        manifest = RunManifest(
            run_id=f"sync_{uuid.uuid4().hex[:16]}",
            model_name=self.model_name,
            initial_checkpoint_path=self.initial_checkpoint_path,
            config_hash=self.config.stable_hash(),
            target_rounds=target_rounds,
        )
        atomic_write_model(path, manifest)
        return manifest

    def _load_latest(self, required: bool = False) -> LatestPublication | None:
        path = self.output_dir / "latest.json"
        if not path.exists():
            if required:
                raise FileNotFoundError(path)
            return None
        publication = LatestPublication.model_validate_json(path.read_text(encoding="utf-8"))
        self._validate_publication(publication)
        return publication

    def _recover_unindexed_publications(
        self,
        latest: LatestPublication | None,
    ) -> LatestPublication | None:
        while True:
            candidate_index = latest.round_index + 1 if latest else 0
            manifest_path = (
                self.output_dir
                / "rounds"
                / f"round_{candidate_index:04d}"
                / "round_manifest.json"
            )
            if not manifest_path.exists():
                return latest
            manifest = RoundManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            if manifest.status != RoundStatus.PUBLISHED:
                return latest
            if not self.config.matches_hash_except_runtime_retry_fields(
                manifest.config_hash
            ):
                raise ValueError("unindexed published round config hash does not match")
            expected_before = latest.policy_version if latest else "policy_000000"
            expected_hash = latest.adapter_hash if latest else ""
            if manifest.policy_version_before != expected_before:
                raise ValueError("unindexed published round has an invalid policy predecessor")
            if manifest.policy_checkpoint_hash_before != expected_hash:
                raise ValueError("unindexed published round has an invalid checkpoint predecessor")
            if sha256_path(manifest.adapter_path) != manifest.adapter_hash:
                raise ValueError("unindexed published adapter checksum does not match")
            if sha256_path(manifest.optimizer_path) != manifest.optimizer_hash:
                raise ValueError("unindexed published optimizer checksum does not match")
            if sha256_path(manifest.optimizer_batch_path) != manifest.optimizer_batch_hash:
                raise ValueError("unindexed published batch checksum does not match")
            latest = LatestPublication(
                run_id=manifest.run_id,
                round_index=manifest.round_index,
                policy_version=manifest.policy_version_after,
                adapter_path=manifest.adapter_path,
                adapter_hash=manifest.adapter_hash,
                optimizer_path=manifest.optimizer_path,
                optimizer_hash=manifest.optimizer_hash,
                config_hash=manifest.config_hash,
            )
            self._validate_publication(latest)
            atomic_write_model(self.output_dir / "latest.json", latest)

    def _validate_publication(self, publication: LatestPublication) -> None:
        if not self.config.matches_hash_except_runtime_retry_fields(
            publication.config_hash
        ):
            raise ValueError("published checkpoint config hash does not match")
        expected = f"policy_{publication.round_index + 1:06d}"
        if publication.policy_version != expected:
            raise ValueError(
                f"published version {publication.policy_version} does not match round {publication.round_index}"
            )
        if sha256_path(publication.adapter_path) != publication.adapter_hash:
            raise ValueError("published adapter checksum does not match")
        if sha256_path(publication.optimizer_path) != publication.optimizer_hash:
            raise ValueError("published optimizer checksum does not match")

    def _archive_incomplete_round(self, round_dir: Path) -> Path | None:
        if not round_dir.exists():
            return None
        manifest_path = round_dir / "round_manifest.json"
        if manifest_path.exists():
            try:
                manifest = RoundManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = None
            if manifest is not None and manifest.status == RoundStatus.PUBLISHED:
                raise FileExistsError(f"published round already exists: {round_dir}")
        suffix = time.strftime("%Y%m%d_%H%M%S")
        archived = round_dir.with_name(f"{round_dir.name}.incomplete_{suffix}")
        counter = 1
        while archived.exists():
            archived = round_dir.with_name(f"{round_dir.name}.incomplete_{suffix}_{counter}")
            counter += 1
        os.replace(round_dir, archived)
        return archived

    def _data_metrics(
        self,
        bundles: list[RolloutGroupBundle],
        batch: CompiledBatch,
    ) -> dict[str, Any]:
        rewards = [
            node.credit.reward.reward
            for bundle in bundles
            for tree in bundle.rollouts
            for node in tree.nodes.values()
            if node.credit.reward is not None
        ]
        by_group: dict[str, list[float]] = {}
        for sample in batch.samples:
            by_group.setdefault(sample.group_id, []).append(sample.advantage)
        epsilon = self.config.sync_flywheel.nonzero_advantage_epsilon
        zero_groups = sum(
            not any(abs(advantage) > epsilon for advantage in advantages)
            for advantages in by_group.values()
        )
        root_successes = [
            tree.nodes[tree.root_node_id].evaluation.value
            for bundle in bundles
            for tree in bundle.rollouts
            if tree.nodes[tree.root_node_id].evaluation is not None
            and tree.nodes[tree.root_node_id].evaluation.value is not None
        ]
        return {
            "reward_count": len(rewards),
            "reward_min": min(rewards) if rewards else None,
            "reward_max": max(rewards) if rewards else None,
            "reward_mean": sum(rewards) / len(rewards) if rewards else None,
            "root_success_rate": sum(root_successes) / len(root_successes) if root_successes else None,
            "zero_advantage_group_count": zero_groups,
            "optimizer_sample_count": len(batch.samples),
        }

    def _validate_optimizer_metrics(self, metrics: dict[str, Any]) -> None:
        for key in ("loss", "ratio_mean", "grad_norm"):
            value = float(metrics[key])
            if not math.isfinite(value):
                raise ValueError(f"optimizer metric {key} is not finite")
        ratio = float(metrics["ratio_mean"])
        lower = self.config.sync_flywheel.ratio_min
        upper = self.config.sync_flywheel.ratio_max
        if not lower <= ratio <= upper:
            raise ValueError(
                f"importance ratio {ratio:.6f} is outside required range [{lower}, {upper}]"
            )

    @staticmethod
    def _service_metrics(service: Any) -> dict[str, Any]:
        for name in ("service_metrics", "metrics"):
            method = getattr(service, name, None)
            if callable(method):
                value = method()
                return dict(value) if isinstance(value, dict) else {}
        return {}

    @staticmethod
    def _metric_delta(
        current: dict[str, Any],
        previous: dict[str, Any],
    ) -> dict[str, Any]:
        delta: dict[str, Any] = {}
        for key, value in current.items():
            before = previous.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isinstance(before, (int, float))
                and not isinstance(before, bool)
            ):
                delta[key] = value - before
            else:
                delta[key] = value
        return delta
