from dataclasses import dataclass
import time

import pandas as pd
import torch
import torch.nn.functional as torch_f

from vision_demo.image_utils import overlay_class_mask
from vision_demo.models import get_device, load_fcn


@dataclass
class SegmentationResult:
    overlay: object
    class_stats: list[dict]
    elapsed_ms: float


def run_fcn_segmentation(image):
    model, weights = load_fcn()
    device = get_device()
    preprocess = weights.transforms()

    started = time.perf_counter()
    batch = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(batch)["out"][0]
        output = torch_f.interpolate(
            output.unsqueeze(0),
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )[0]
        probabilities = torch.softmax(output, dim=0)
        confidence, mask = probabilities.max(dim=0)

    mask_np = mask.byte().cpu().numpy()
    confidence_np = confidence.cpu().numpy()
    overlay = overlay_class_mask(image, mask_np)
    elapsed_ms = (time.perf_counter() - started) * 1000

    categories = weights.meta.get("categories", [])
    total = mask_np.size
    stats = []
    for class_id in sorted(set(mask_np.flatten().tolist())):
        if class_id == 0:
            continue
        active = mask_np == class_id
        pixels = int(active.sum())
        if pixels == 0:
            continue
        label = categories[class_id] if class_id < len(categories) else f"class_{class_id}"
        stats.append(
            {
                "class": label,
                "pixel_percent": round(pixels / total * 100, 2),
                "mean_confidence": round(float(confidence_np[active].mean()), 3),
            }
        )
    return SegmentationResult(overlay=overlay, class_stats=stats, elapsed_ms=elapsed_ms)


def class_stats_to_frame(stats):
    if not stats:
        return pd.DataFrame(columns=["类别", "像素占比", "平均置信度"])
    return (
        pd.DataFrame(stats)
        .rename(
            columns={
                "class": "类别",
                "pixel_percent": "像素占比",
                "mean_confidence": "平均置信度",
            }
        )
        .sort_values("像素占比", ascending=False)
    )
