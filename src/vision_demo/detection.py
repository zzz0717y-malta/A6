from collections import OrderedDict
from dataclasses import dataclass
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.ops import nms
from torchvision.transforms import functional as F

from vision_demo.image_utils import Detection, draw_detections
from vision_demo.models import get_device, load_classifier, load_faster_rcnn


@dataclass
class DetectionResult:
    visualization: object
    detections: list[Detection]
    elapsed_ms: float


def _clip_box(box, width, height):
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, float(x1))),
        max(0, min(height - 1, float(y1))),
        max(1, min(width, float(x2))),
        max(1, min(height, float(y2))),
    )


def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _nms_numpy(boxes, scores, threshold=0.45, max_boxes=80):
    if not boxes:
        return []
    box_tensor = torch.tensor(boxes, dtype=torch.float32)
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    keep = nms(box_tensor, score_tensor, threshold).cpu().tolist()
    return keep[:max_boxes]


def generate_region_proposals(image, max_proposals=80):
    width, height = image.size
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 160)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    scores = []
    image_area = width * height
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < image_area * 0.004 or area > image_area * 0.85:
            continue
        pad_x = max(4, int(w * 0.08))
        pad_y = max(4, int(h * 0.08))
        box = _clip_box((x - pad_x, y - pad_y, x + w + pad_x, y + h + pad_y), width, height)
        if _box_area(box) <= 0:
            continue
        boxes.append(box)
        scores.append(float(area) / image_area)

    for scale in (0.32, 0.48, 0.64, 0.82):
        win_w = int(width * scale)
        win_h = int(height * scale)
        step_x = max(1, win_w // 2)
        step_y = max(1, win_h // 2)
        for y in range(0, max(1, height - win_h + 1), step_y):
            for x in range(0, max(1, width - win_w + 1), step_x):
                box = _clip_box((x, y, x + win_w, y + win_h), width, height)
                boxes.append(box)
                scores.append(0.02 + _box_area(box) / image_area * 0.1)

    boxes.append((0.0, 0.0, float(width), float(height)))
    scores.append(0.01)
    keep = _nms_numpy(boxes, scores, max_boxes=max_proposals)
    return [boxes[i] for i in keep]


def _filter_detector_output(output, categories, score_threshold=0.45, max_detections=20):
    boxes = output["boxes"].detach().cpu()
    labels = output["labels"].detach().cpu().tolist()
    scores = output["scores"].detach().cpu().tolist()

    detections = []
    for box, label_id, score in zip(boxes.tolist(), labels, scores):
        if score < score_threshold:
            continue
        label = categories[label_id] if label_id < len(categories) else f"class_{label_id}"
        detections.append(Detection(label=label, score=float(score), box=tuple(map(float, box))))
        if len(detections) >= max_detections:
            break
    return detections


def run_rcnn_style(image, score_threshold=0.25, max_regions=14):
    classifier, weights = load_classifier()
    device = get_device()
    preprocess = weights.transforms()
    categories = weights.meta.get("categories", [])

    started = time.perf_counter()
    proposals = generate_region_proposals(image, max_proposals=max_regions * 3)
    crops = []
    crop_boxes = []
    for box in proposals:
        if len(crops) >= max_regions:
            break
        x1, y1, x2, y2 = box
        if (x2 - x1) < 24 or (y2 - y1) < 24:
            continue
        crops.append(preprocess(image.crop((x1, y1, x2, y2))))
        crop_boxes.append(box)

    detections = []
    if crops:
        batch = torch.stack(crops).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(classifier(batch), dim=1)
        scores, label_ids = probabilities.max(dim=1)
        for box, score, label_id in zip(crop_boxes, scores.cpu().tolist(), label_ids.cpu().tolist()):
            if score < score_threshold:
                continue
            label = categories[label_id] if label_id < len(categories) else f"class_{label_id}"
            detections.append(Detection(label=label, score=float(score), box=box))

    if detections:
        keep = _nms_numpy([d.box for d in detections], [d.score for d in detections], threshold=0.35, max_boxes=12)
        detections = [detections[i] for i in keep]
    visualization = draw_detections(image, detections)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return DetectionResult(visualization=visualization, detections=detections, elapsed_ms=elapsed_ms)


def _resize_boxes(boxes, original_size, new_size):
    original_h, original_w = original_size
    new_h, new_w = new_size
    ratios = torch.tensor(
        [new_w / original_w, new_h / original_h, new_w / original_w, new_h / original_h],
        dtype=boxes.dtype,
        device=boxes.device,
    )
    return boxes * ratios


def run_fast_rcnn_style(image, score_threshold=0.45, max_detections=20):
    model, weights = load_faster_rcnn()
    device = get_device()
    categories = weights.meta.get("categories", [])

    started = time.perf_counter()
    tensor = F.to_tensor(image).to(device)
    original_size = (tensor.shape[-2], tensor.shape[-1])
    proposals_np = generate_region_proposals(image, max_proposals=96)
    proposals = torch.tensor(proposals_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        images, _ = model.transform([tensor], None)
        features = model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        transformed_proposals = [_resize_boxes(proposals, original_size, images.image_sizes[0])]
        detections, _ = model.roi_heads(features, transformed_proposals, images.image_sizes, None)
        detections = model.transform.postprocess(detections, images.image_sizes, [original_size])

    selected = _filter_detector_output(detections[0], categories, score_threshold, max_detections)
    visualization = draw_detections(image, selected)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return DetectionResult(visualization=visualization, detections=selected, elapsed_ms=elapsed_ms)


def run_faster_rcnn(image, score_threshold=0.45, max_detections=20):
    model, weights = load_faster_rcnn()
    device = get_device()
    categories = weights.meta.get("categories", [])

    started = time.perf_counter()
    tensor = F.to_tensor(image).to(device)
    with torch.no_grad():
        output = model([tensor])[0]
    selected = _filter_detector_output(output, categories, score_threshold, max_detections)
    visualization = draw_detections(image, selected)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return DetectionResult(visualization=visualization, detections=selected, elapsed_ms=elapsed_ms)


def detections_to_frame(detections):
    rows = []
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        rows.append(
            {
                "类别": detection.label,
                "置信度": round(detection.score, 3),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            }
        )
    return pd.DataFrame(rows, columns=["类别", "置信度", "x1", "y1", "x2", "y2"])
