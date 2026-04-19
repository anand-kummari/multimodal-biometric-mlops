"""Model export utilities for production deployment.

Converts a trained MultimodalFusionNet to ONNX format so it can be
served by runtime-agnostic inference engines (Triton, ONNX Runtime,
TensorRT).  The exported graph uses dynamic batch axes, allowing the
same artifact to handle both single-sample and batched requests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shapes: dict[str, tuple[int, ...]] | None = None,
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export a multimodal model to the ONNX format.

    Args:
        model: Trained PyTorch model in eval mode.
        output_path: Destination ``.onnx`` file path.
        input_shapes: Per-modality shapes **including** the batch dim,
            e.g. ``{"iris_left": (1, 3, 224, 224), ...}``.
        opset_version: ONNX opset to target.
        dynamic_batch: When True the batch dimension is marked dynamic
            so the graph accepts variable-size batches at inference.

    Returns:
        Resolved path to the written ``.onnx`` file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shapes = input_shapes or {
        "iris_left": (1, 3, 224, 224),
        "iris_right": (1, 3, 224, 224),
        "fingerprint": (1, 1, 224, 224),
    }

    dummy_inputs = {k: torch.randn(*s) for k, s in shapes.items()}
    input_names = list(shapes.keys())

    model.eval()
    wrapper = _DictInputWrapper(model, input_names)

    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch:
        dynamic_axes = {name: {0: "batch"} for name in [*input_names, "logits"]}

    positional = tuple(dummy_inputs[n] for n in input_names)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            positional,
            str(output_path),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            dynamo=False,
            verbose=False,
        )
    logger.info("ONNX model exported to %s (opset %d)", output_path, opset_version)
    return output_path.resolve()


class _DictInputWrapper(nn.Module):
    """Thin wrapper that converts positional tensor args back into the dict the model expects."""

    def __init__(self, model: nn.Module, input_names: list[str]) -> None:
        super().__init__()
        self._model = model
        self._names = input_names

    def forward(self, *args: Any) -> torch.Tensor:
        """Map positional args to a dict and delegate to the real model."""
        features = dict(zip(self._names, args, strict=False))
        result: torch.Tensor = self._model(features)
        return result
