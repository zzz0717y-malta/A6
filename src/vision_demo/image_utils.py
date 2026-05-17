from dataclasses import dataclass
from io import BytesIO
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]


def resize_for_inference(image_or_file, max_side=640):
    if isinstance(image_or_file, Image.Image):
        image = image_or_file.convert("RGB")
    else:
        image = Image.open(image_or_file).convert("RGB")

    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)


def load_image_from_url(url, max_bytes=10_000_000):
    request = Request(url, headers={"User-Agent": "streamlit-vision-demo/1.0"})
    with urlopen(request, timeout=15) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("图片过大，请选择 10MB 以内的图片")
    return Image.open(BytesIO(data)).convert("RGB")


def make_sample_image(width=960, height=620):
    image = Image.new("RGB", (width, height), (218, 230, 235))
    draw = ImageDraw.Draw(image)

    draw.rectangle([0, int(height * 0.58), width, height], fill=(98, 111, 118))
    draw.polygon(
        [(0, int(height * 0.58)), (width, int(height * 0.45)), (width, int(height * 0.62)), (0, int(height * 0.78))],
        fill=(122, 132, 136),
    )
    draw.rectangle([0, int(height * 0.46), width, int(height * 0.58)], fill=(82, 145, 106))
    for x in range(50, width, 160):
        draw.rectangle([x, 125, x + 82, 285], fill=(179, 191, 197))
        draw.rectangle([x + 12, 145, x + 32, 180], fill=(94, 125, 145))
        draw.rectangle([x + 48, 145, x + 68, 180], fill=(94, 125, 145))
    draw.rectangle([340, 365, 685, 465], fill=(202, 72, 73))
    draw.polygon([(400, 365), (475, 300), (610, 300), (675, 365)], fill=(185, 58, 66))
    draw.rectangle([455, 318, 535, 360], fill=(88, 133, 160))
    draw.rectangle([552, 318, 628, 360], fill=(88, 133, 160))
    draw.ellipse([385, 440, 455, 510], fill=(30, 34, 37))
    draw.ellipse([575, 440, 645, 510], fill=(30, 34, 37))
    draw.ellipse([407, 462, 433, 488], fill=(188, 196, 200))
    draw.ellipse([597, 462, 623, 488], fill=(188, 196, 200))
    draw.ellipse([735, 260, 785, 310], fill=(222, 177, 132))
    draw.rectangle([750, 310, 774, 405], fill=(46, 97, 166))
    draw.line([750, 330, 710, 370], fill=(222, 177, 132), width=9)
    draw.line([774, 330, 815, 370], fill=(222, 177, 132), width=9)
    draw.line([756, 405, 742, 480], fill=(36, 49, 71), width=10)
    draw.line([770, 405, 795, 480], fill=(36, 49, 71), width=10)
    draw.text((24, 24), "Upload a real photo for best pretrained-model results", fill=(35, 44, 48))
    return image


def color_for_index(index):
    if index == 0:
        return (0, 0, 0)
    return (
        int((37 * index + 91) % 255),
        int((73 * index + 47) % 255),
        int((109 * index + 19) % 255),
    )


def overlay_class_mask(image, mask, alpha=0.48):
    base = np.array(image).astype(np.float32)
    color = np.zeros_like(base)
    for class_id in np.unique(mask):
        if class_id == 0:
            continue
        color[mask == class_id] = color_for_index(int(class_id))
    active = mask > 0
    base[active] = base[active] * (1 - alpha) + color[active] * alpha
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def overlay_instance_masks(image, masks, boxes, labels, scores, mask_threshold=0.5, alpha=0.46):
    canvas = np.array(image).astype(np.float32)
    for idx, mask in enumerate(masks):
        active = mask > mask_threshold
        if not np.any(active):
            continue
        color = np.array(color_for_index(idx + 1), dtype=np.float32)
        canvas[active] = canvas[active] * (1 - alpha) + color * alpha
    output = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    detections = [
        Detection(label=label, score=float(score), box=tuple(map(float, box)))
        for box, label, score in zip(boxes, labels, scores)
    ]
    return draw_detections(output, detections)


def draw_detections(image, detections, width=3):
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    for idx, detection in enumerate(detections):
        color = color_for_index(idx + 1)
        x1, y1, x2, y2 = detection.box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        label = f"{detection.label} {detection.score:.2f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        y_text = max(0, y1 - text_h - 5)
        draw.rectangle([x1, y_text, x1 + text_w + 8, y_text + text_h + 5], fill=color)
        draw.text((x1 + 4, y_text + 2), label, fill=(255, 255, 255), font=font)
    return output
