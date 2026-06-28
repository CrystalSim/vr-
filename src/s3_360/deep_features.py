from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from PIL import Image


def extract_googlenet_pool5(
    frames: np.ndarray,
    batch_size: int = 16,
    device: str | None = None,
) -> np.ndarray:
    """Extract ImageNet-pretrained GoogleNet pool5 features for paper-style experiments."""
    try:
        import torch
        from torchvision.models import GoogLeNet_Weights, googlenet
        from torchvision.models.feature_extraction import create_feature_extractor
    except ImportError as exc:
        raise RuntimeError(
            "Deep feature extraction requires torch and torchvision. "
            "Install the optional deep-learning stack before running this step."
        ) from exc

    weights = GoogLeNet_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    model = googlenet(weights=weights, aux_logits=True).eval()
    extractor = create_feature_extractor(model, return_nodes={"avgpool": "pool5"}).eval()
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    extractor.to(run_device)

    features: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in _batches(frames, batch_size):
            tensors = [
                transform(Image.fromarray(frame.astype(np.uint8)).convert("RGB"))
                for frame in batch
            ]
            stacked = torch.stack(tensors).to(run_device)
            pooled = extractor(stacked)["pool5"].flatten(1).cpu().numpy()
            features.append(pooled.astype(np.float32))
    return np.concatenate(features, axis=0)


def _batches(frames: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(frames), batch_size):
        yield frames[start : start + batch_size]
