"""Optional Hugging Face semantic-segmentation adapters."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any, Mapping

import numpy as np

from .configuration import runtime_config_section
from .inference import (
    ModelMetadata,
    SegmentationAdapter,
    SegmentationPrediction,
    _validate_image,
)


_DEFAULTS = runtime_config_section("pretrained_segmentation")


DEFAULT_ROAD_SEGMENTATION_MODEL = str(_DEFAULTS["model_name"])


@dataclass(frozen=True, slots=True)
class HuggingFaceSegmentationConfig:
    """Loading and class-mapping options for a pretrained model."""

    model_name: str = DEFAULT_ROAD_SEGMENTATION_MODEL
    revision: str | None = None
    source_road_labels: tuple[str, ...] = tuple(
        _DEFAULTS["source_road_labels"]
    )
    output_road_class_id: int = int(_DEFAULTS["output_road_class_id"])
    device: str = str(_DEFAULTS["device"])
    precision: str = str(_DEFAULTS["precision"])
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("pretrained model name must not be empty")
        if not self.source_road_labels or any(
            not label.strip() for label in self.source_road_labels
        ):
            raise ValueError("at least one source road label is required")
        if not 1 <= self.output_road_class_id <= 255:
            raise ValueError("output road class ID must be in [1, 255]")
        if self.device not in {"auto", "cpu", "mps", "cuda"} and not (
            self.device.startswith("cuda:")
            and self.device.removeprefix("cuda:").isdigit()
        ):
            raise ValueError("device must be auto, cpu, mps, cuda, or cuda:N")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")


class HuggingFaceSegmentationAdapter(SegmentationAdapter):
    """Map an off-the-shelf semantic model's road labels to a binary mask.

    The default is NVIDIA SegFormer-B0 fine-tuned on ADE20K. PyTorch and
    Transformers are loaded only when this adapter is constructed.
    """

    def __init__(
        self,
        config: HuggingFaceSegmentationConfig | None = None,
        *,
        model_id: str | None = None,
        display_name: str = "SegFormer-B0 ADE20K road",
        compression: str = "none",
    ) -> None:
        self.config = config or HuggingFaceSegmentationConfig()
        torch, processor_class, model_class = _load_dependencies()
        self._torch = torch
        self.device = _select_device(torch, self.config.device)
        self._dtype = _torch_dtype(torch, self.config.precision)
        if self.device == "cpu" and self.config.precision == "fp16":
            raise ValueError("fp16 pretrained inference is not supported on CPU")

        load_options: dict[str, Any] = {
            "local_files_only": self.config.local_files_only,
            "trust_remote_code": False,
        }
        if self.config.revision is not None:
            load_options["revision"] = self.config.revision
        self._processor = processor_class.from_pretrained(
            self.config.model_name, **load_options
        )
        self._model = model_class.from_pretrained(
            self.config.model_name, **load_options
        )
        self._model.to(device=self.device, dtype=self._dtype)
        self._model.eval()
        self.source_road_class_ids = resolve_source_class_ids(
            self._model.config.id2label,
            self.config.source_road_labels,
        )

        input_width, input_height = _processor_dimensions(self._processor)
        resolved_id = model_id or f"hf:{self.config.model_name}"
        if self.config.revision is not None and model_id is None:
            resolved_id += f"@{self.config.revision}"
        self._metadata = ModelMetadata(
            model_id=resolved_id,
            display_name=display_name,
            backend=f"pytorch-{self.device}",
            precision=self.config.precision,
            compression=compression,
            input_width=input_width,
            input_height=input_height,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def infer(self, image_bgr: np.ndarray) -> SegmentationPrediction:
        _validate_image(image_bgr)
        image_rgb = np.ascontiguousarray(image_bgr[..., ::-1])
        inputs = self._processor(images=image_rgb, return_tensors="pt")
        model_inputs: dict[str, Any] = {}
        for name, value in inputs.items():
            value = value.to(self.device)
            if value.is_floating_point():
                value = value.to(self._dtype)
            model_inputs[name] = value

        with self._torch.inference_mode():
            output = self._model(**model_inputs)
            logits = self._torch.nn.functional.interpolate(
                output.logits,
                size=image_bgr.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            source_labels = logits.argmax(dim=1)[0]
            road = self._torch.zeros_like(source_labels, dtype=self._torch.bool)
            for source_id in self.source_road_class_ids:
                road |= source_labels == source_id
            labels = road.to(self._torch.uint8) * self.config.output_road_class_id
            labels = labels.cpu().numpy()
        _synchronise(self._torch, self.device)
        return SegmentationPrediction(
            labels=labels,
            road_class_id=self.config.output_road_class_id,
        )


def resolve_source_class_ids(
    id_to_label: Mapping[int | str, str],
    requested_labels: tuple[str, ...],
) -> tuple[int, ...]:
    """Resolve human-readable class names without hardcoding ADE20K IDs."""

    available = {
        _normalise_label(str(label)): int(class_id)
        for class_id, label in id_to_label.items()
    }
    requested = tuple(_normalise_label(label) for label in requested_labels)
    missing = [label for label in requested if label not in available]
    if missing:
        names = ", ".join(sorted(available))
        raise ValueError(
            f"source labels not found: {', '.join(missing)}; "
            f"model labels are: {names}"
        )
    return tuple(available[label] for label in requested)


def _normalise_label(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", " ").split())


def _load_dependencies() -> tuple[Any, Any, Any]:
    _validate_pytorch_runtime_safety()
    try:
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForSemanticSegmentation,
        )
    except ImportError as error:
        raise RuntimeError(
            "pretrained segmentation requires the optional dependencies; "
            "install jetracer-sim[pretrained]"
        ) from error
    return torch, AutoImageProcessor, AutoModelForSemanticSegmentation


def _validate_pytorch_runtime_safety(
    *,
    platform_name: str | None = None,
) -> None:
    """Reject in-process PyTorch from the native package on macOS."""

    resolved_platform = sys.platform if platform_name is None else platform_name
    if resolved_platform == "darwin":
        raise RuntimeError(
            "in-process PyTorch segmentation is disabled on macOS because the "
            "native simulator and PyTorch can load conflicting libomp runtimes "
            "and abort Python; export the model with "
            "tools/export_segformer_onnx.py and use the ONNX adapter"
        )


def _select_device(torch: Any, requested: str) -> str:
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(
        mps_backend is not None and mps_backend.is_available()
    )
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not mps_available:
            raise RuntimeError("MPS was requested but is unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def _torch_dtype(torch: Any, precision: str) -> Any:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]


def _processor_dimensions(processor: Any) -> tuple[int | None, int | None]:
    size = getattr(processor, "size", None)
    if not isinstance(size, dict):
        return None, None
    width = size.get("width")
    height = size.get("height")
    if isinstance(width, int) and isinstance(height, int):
        return width, height
    return None, None


def _synchronise(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elif device == "mps":
        torch.mps.synchronize()
