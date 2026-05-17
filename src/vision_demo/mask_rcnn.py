from dataclasses import dataclass
import time

import pandas as pd
import torch
from torchvision.transforms import functional as F

from vision_demo.image_utils import Detection, overlay_instance_masks
from vision_demo.models import get_device, load_mask_rcnn


@dataclass
class InstanceResult:
    visualization: object
    instances: list[Detection]
    elapsed_ms: float


def run_mask_rcnn(image, score_threshold=0.5, mask_threshold=0.5, max_instances=16):
    model, weights = load_mask_rcnn()
    device = get_device()
    categories = weights.meta.get("categories", [])

    started = time.perf_counter()
    tensor = F.to_tensor(image).to(device)
    with torch.no_grad():
        output = model([tensor])[0]

    boxes = output["boxes"].detach().cpu().tolist()
    labels = output["labels"].detach().cpu().tolist()
    scores = output["scores"].detach().cpu().tolist()
    masks = output["masks"].detach().cpu().numpy()[:, 0]

    selected_boxes = []
    selected_labels = []
    selected_scores = []
    selected_masks = []
    instances = []
    for box, label_id, score, mask in zip(boxes, labels, scores, masks):
        if score < score_threshold:
            continue
        label = categories[label_id] if label_id < len(categories) else f"class_{label_id}"
        selected_boxes.append(box)
        selected_labels.append(label)
        selected_scores.append(score)
        selected_masks.append(mask)
        instances.append(Detection(label=label, score=float(score), box=tuple(map(float, box))))
        if len(instances) >= max_instances:
            break

    visualization = overlay_instance_masks(
        image,
        selected_masks,
        selected_boxes,
        selected_labels,
        selected_scores,
        mask_threshold=mask_threshold,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return InstanceResult(visualization=visualization, instances=instances, elapsed_ms=elapsed_ms)


def instances_to_frame(instances):
    rows = []
    for instance in instances:
        x1, y1, x2, y2 = instance.box
        rows.append(
            {
                "实例类别": instance.label,
                "置信度": round(instance.score, 3),
                "bbox": f"({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})",
            }
        )
    return pd.DataFrame(rows, columns=["实例类别", "置信度", "bbox"])
