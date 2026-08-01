"""Target-aware model artifact, runtime-provider, and benchmark gating."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

from .model_registry import (
    DetectionModelVariant,
    ModelBenchmark,
    ModelVariant,
    load_detection_model_variants,
    load_model_benchmarks,
    load_model_variants,
)
from .resource_paths import configuration_resource


DEPLOYMENT_POLICY_SCHEMA_VERSION = 1


def _default_deployment_policy_path() -> Path:
    return configuration_resource("hardware/jetson_deployment.json")


DEFAULT_DEPLOYMENT_POLICY_PATH = _default_deployment_policy_path()


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    system: str
    machine: str
    onnxruntime_version: str | None
    onnx_execution_providers: tuple[str, ...]
    tensorrt_version: str | None
    probe_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DeploymentVariantStatus:
    task: str
    model_id: str
    adapter_kind: str
    selectable: bool
    checks: tuple[dict[str, Any], ...]

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(
            str(check["id"]) for check in self.checks if not check["passed"]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "model_id": self.model_id,
            "adapter_kind": self.adapter_kind,
            "selectable": self.selectable,
            "reasons": list(self.reasons),
            "checks": [dict(check) for check in self.checks],
        }


@dataclass(frozen=True, slots=True)
class DeploymentReport:
    target_id: str
    target_match: bool
    ready: bool
    capabilities: RuntimeCapabilities
    variants: tuple[DeploymentVariantStatus, ...]

    @property
    def selectable_model_ids(self) -> tuple[str, ...]:
        return tuple(
            variant.model_id for variant in self.variants if variant.selectable
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_match": self.target_match,
            "ready": self.ready,
            "selectable_model_ids": list(self.selectable_model_ids),
            "capabilities": {
                "system": self.capabilities.system,
                "machine": self.capabilities.machine,
                "onnxruntime_version": self.capabilities.onnxruntime_version,
                "onnx_execution_providers": list(
                    self.capabilities.onnx_execution_providers
                ),
                "tensorrt_version": self.capabilities.tensorrt_version,
                "probe_errors": list(self.capabilities.probe_errors),
            },
            "variants": [variant.to_dict() for variant in self.variants],
        }


def load_deployment_policy(
    path: str | Path = DEFAULT_DEPLOYMENT_POLICY_PATH,
) -> dict[str, Any]:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load deployment policy: {source}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("unsupported deployment-policy schema")
    for name in ("target_host", "tasks", "policy"):
        if not isinstance(document.get(name), dict):
            raise ValueError(f"deployment policy requires {name}")
    policy = document["policy"]
    if not isinstance(policy.get("allowed_adapter_kinds"), list):
        raise ValueError("deployment policy requires allowed adapter kinds")
    if not isinstance(policy.get("allowed_execution_providers"), list):
        raise ValueError("deployment policy requires execution providers")
    if int(policy.get("sha256_chunk_bytes", 0)) <= 0:
        raise ValueError("deployment hash chunk size must be positive")
    if float(policy.get("runtime_probe_timeout_s", 0.0)) <= 0.0:
        raise ValueError("deployment runtime probe timeout must be positive")
    return document


def artifact_sha256(path: str | Path, *, chunk_bytes: int) -> str:
    if chunk_bytes <= 0:
        raise ValueError("artifact hash chunk size must be positive")
    digest = sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def collect_runtime_capabilities(
    policy: Mapping[str, Any],
) -> RuntimeCapabilities:
    settings = policy["policy"]
    timeout_s = float(settings["runtime_probe_timeout_s"])
    output_limit = int(settings["runtime_probe_output_limit_characters"])
    probe_source = (
        "import json\n"
        "result = {'onnxruntime_version': None, 'onnx_execution_providers': [], "
        "'tensorrt_version': None, 'errors': []}\n"
        "try:\n"
        " import onnxruntime as ort\n"
        " result['onnxruntime_version'] = ort.__version__\n"
        " result['onnx_execution_providers'] = ort.get_available_providers()\n"
        "except BaseException as error:\n"
        " result['errors'].append('onnxruntime: ' + type(error).__name__ + ': ' + str(error))\n"
        "try:\n"
        " import tensorrt as trt\n"
        " result['tensorrt_version'] = trt.__version__\n"
        "except BaseException as error:\n"
        " result['errors'].append('tensorrt: ' + type(error).__name__ + ': ' + str(error))\n"
        "print(json.dumps(result))\n"
    )
    errors: list[str] = []
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", probe_source],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if completed.returncode == 0:
            payload = json.loads(completed.stdout)
        else:
            payload = {}
            errors.append(
                f"runtime probe exited {completed.returncode}: "
                f"{completed.stderr[:output_limit]}"
            )
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        payload = {}
        errors.append(f"runtime probe failed: {error}")
    errors.extend(str(value) for value in payload.get("errors", []))
    return RuntimeCapabilities(
        system=platform.system(),
        machine=platform.machine(),
        onnxruntime_version=payload.get("onnxruntime_version"),
        onnx_execution_providers=tuple(
            str(value) for value in payload.get("onnx_execution_providers", [])
        ),
        tensorrt_version=payload.get("tensorrt_version"),
        probe_errors=tuple(errors),
    )


def evaluate_deployment(
    model_configuration_path: str | Path,
    benchmark_path: str | Path,
    policy: Mapping[str, Any],
    capabilities: RuntimeCapabilities,
) -> DeploymentReport:
    settings = policy["policy"]
    target = policy["target_host"]
    target_match = (
        capabilities.system == target["system"]
        and capabilities.machine in target["machines"]
    )
    benchmarks = load_model_benchmarks(benchmark_path)
    segmentation = load_model_variants(
        model_configuration_path, benchmark_path
    )
    detectors = load_detection_model_variants(model_configuration_path)
    statuses = tuple(
        [
            _evaluate_variant(
                "segmentation",
                variant,
                variant.benchmark,
                settings,
                capabilities,
                target_match,
            )
            for variant in segmentation
        ]
        + [
            _evaluate_variant(
                "object_detection",
                variant,
                benchmarks.get(variant.model_id),
                settings,
                capabilities,
                target_match,
            )
            for variant in detectors
        ]
    )
    task_ready = []
    for task_name, task_policy in policy["tasks"].items():
        available = sum(
            status.selectable for status in statuses if status.task == task_name
        )
        task_ready.append(
            available >= int(task_policy["minimum_selectable_variants"])
        )
    return DeploymentReport(
        target_id=str(policy["target_id"]),
        target_match=target_match,
        ready=target_match and all(task_ready),
        capabilities=capabilities,
        variants=statuses,
    )


def _evaluate_variant(
    task: str,
    variant: ModelVariant | DetectionModelVariant,
    benchmark: ModelBenchmark | None,
    settings: Mapping[str, Any],
    capabilities: RuntimeCapabilities,
    target_match: bool,
) -> DeploymentVariantStatus:
    options = variant.adapter_options
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, observed: Any, required: Any) -> None:
        checks.append(
            {
                "id": check_id,
                "passed": passed,
                "observed": observed,
                "required": required,
            }
        )

    check("target_host", target_match, [capabilities.system, capabilities.machine], "target match")
    allowed_kinds = settings["allowed_adapter_kinds"]
    check(
        "adapter_kind",
        variant.adapter_kind in allowed_kinds,
        variant.adapter_kind,
        allowed_kinds,
    )
    disabled_reason = options.get("runtime_disabled_reason")
    check(
        "runtime_enabled",
        disabled_reason is None,
        disabled_reason,
        None,
    )
    model_path_value = options.get("model_path")
    model_path = None if model_path_value is None else Path(str(model_path_value))
    check(
        "artifact_exists",
        model_path is not None and model_path.is_file(),
        None if model_path is None else str(model_path),
        "existing file",
    )
    expected_digest = options.get("artifact_sha256")
    require_digest = bool(settings["require_artifact_sha256"])
    digest_matches = False
    actual_digest = None
    if model_path is not None and model_path.is_file() and expected_digest:
        actual_digest = artifact_sha256(
            model_path, chunk_bytes=int(settings["sha256_chunk_bytes"])
        )
        digest_matches = actual_digest == expected_digest
    check(
        "artifact_sha256",
        digest_matches if require_digest else expected_digest is None or digest_matches,
        actual_digest,
        expected_digest if expected_digest is not None else "recorded SHA-256",
    )
    available_providers = set(capabilities.onnx_execution_providers)
    allowed_providers = set(settings["allowed_execution_providers"])
    required_provider = options.get("required_execution_provider")
    provider_ready = (
        str(required_provider) in available_providers
        if required_provider is not None
        else bool(available_providers & allowed_providers)
    )
    runtime_ready = (
        capabilities.onnxruntime_version is not None
        if variant.adapter_kind in {"onnx", "yolo_onnx"}
        else capabilities.tensorrt_version is not None
    )
    check(
        "runtime",
        runtime_ready,
        {
            "onnxruntime": capabilities.onnxruntime_version,
            "tensorrt": capabilities.tensorrt_version,
        },
        "runtime installed",
    )
    check(
        "execution_provider",
        provider_ready,
        sorted(available_providers),
        (
            str(required_provider)
            if required_provider is not None
            else sorted(allowed_providers)
        ),
    )
    benchmark_required = bool(settings["require_target_benchmark"])
    environment_patterns = [
        str(value) for value in settings["benchmark_environment_substrings"]
    ]
    benchmark_matches = (
        benchmark is not None
        and benchmark.iterations >= int(settings["minimum_benchmark_iterations"])
        and all(value in benchmark.environment for value in environment_patterns)
    )
    check(
        "target_benchmark",
        benchmark_matches if benchmark_required else True,
        None if benchmark is None else benchmark.environment,
        {
            "minimum_iterations": settings["minimum_benchmark_iterations"],
            "environment_substrings": environment_patterns,
        },
    )
    return DeploymentVariantStatus(
        task=task,
        model_id=variant.model_id,
        adapter_kind=variant.adapter_kind,
        selectable=all(check_value["passed"] for check_value in checks),
        checks=tuple(checks),
    )


def filter_deployable_model_variants(
    variants: Sequence[ModelVariant], report: DeploymentReport
) -> tuple[ModelVariant, ...]:
    selectable = set(report.selectable_model_ids)
    return tuple(variant for variant in variants if variant.model_id in selectable)
