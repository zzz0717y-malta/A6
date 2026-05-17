from pathlib import Path
import sys
import time

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from vision_demo.detection import (  # noqa: E402
    detections_to_frame,
    run_fast_rcnn_style,
    run_faster_rcnn,
    run_rcnn_style,
)
from vision_demo.image_utils import (  # noqa: E402
    load_image_from_url,
    make_sample_image,
    resize_for_inference,
)
from vision_demo.mask_rcnn import instances_to_frame, run_mask_rcnn  # noqa: E402
from vision_demo.performance import METHOD_COMPARISON, cloud_tips  # noqa: E402
from vision_demo.segmentation import class_stats_to_frame, run_fcn_segmentation  # noqa: E402


st.set_page_config(
    page_title="FCN / R-CNN / Mask R-CNN Demo",
    layout="wide",
)


METHODS = [
    "FCN 语义分割",
    "R-CNN 目标检测（简化示例）",
    "Fast R-CNN 目标检测（RoI Head 示例）",
    "Faster R-CNN 目标检测",
    "Mask R-CNN 实例分割",
    "性能对比",
]


def get_image():
    source = st.sidebar.radio("图片来源", ["上传图片", "图片 URL", "内置示例"], index=2)
    if source == "上传图片":
        uploaded = st.sidebar.file_uploader("选择 JPG / PNG", type=["jpg", "jpeg", "png"])
        if uploaded is None:
            return make_sample_image(), "内置示例"
        return resize_for_inference(uploaded), uploaded.name
    if source == "图片 URL":
        url = st.sidebar.text_input("图片 URL", placeholder="https://example.com/photo.jpg")
        if not url:
            return make_sample_image(), "内置示例"
        try:
            return load_image_from_url(url), url
        except Exception as exc:
            st.sidebar.error(f"图片读取失败：{exc}")
            return make_sample_image(), "内置示例"
    return make_sample_image(), "内置示例"


def show_detection_table(detections):
    frame = detections_to_frame(detections)
    if frame.empty:
        st.info("当前阈值下没有检测结果。可以降低置信度阈值，或换一张真实照片。")
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)


def show_instance_table(instances):
    frame = instances_to_frame(instances)
    if frame.empty:
        st.info("当前阈值下没有实例结果。可以降低置信度阈值，或换一张真实照片。")
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)


def run_benchmark(image, methods, score_threshold, mask_threshold):
    rows = []
    for method in methods:
        started = time.perf_counter()
        if method == "FCN":
            result = run_fcn_segmentation(image)
            count = len(result.class_stats)
        elif method == "R-CNN":
            result = run_rcnn_style(image, score_threshold=score_threshold)
            count = len(result.detections)
        elif method == "Fast R-CNN":
            result = run_fast_rcnn_style(image, score_threshold=score_threshold)
            count = len(result.detections)
        elif method == "Faster R-CNN":
            result = run_faster_rcnn(image, score_threshold=score_threshold)
            count = len(result.detections)
        else:
            result = run_mask_rcnn(
                image,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
            )
            count = len(result.instances)
        rows.append(
            {
                "方法": method,
                "本次耗时(ms)": round((time.perf_counter() - started) * 1000, 1),
                "输出数量": count,
                "说明": "包含模型加载" if len(rows) == 0 else "模型已缓存后更稳定",
            }
        )
    return pd.DataFrame(rows)


st.title("FCN / R-CNN 系列 / Mask R-CNN 视觉任务演示")

method = st.sidebar.radio("演示方法", METHODS)
max_side = st.sidebar.slider("推理图像最长边", 384, 1024, 640, step=64)
image, image_name = get_image()
image = resize_for_inference(image, max_side=max_side)

st.sidebar.caption("首次运行会下载 torchvision 预训练权重；Streamlit Cloud 会缓存到会话磁盘。")

left, right = st.columns([0.92, 1.08])
with left:
    st.subheader("输入图片")
    st.image(image, caption=f"{image_name} | {image.width} x {image.height}", use_column_width=True)

score_threshold = 0.45
mask_threshold = 0.5

if method == "FCN 语义分割":
    with right:
        st.subheader("FCN 语义分割结果")
        with st.spinner("正在运行 FCN-ResNet50..."):
            result = run_fcn_segmentation(image)
        st.image(result.overlay, caption=f"耗时 {result.elapsed_ms:.1f} ms", use_column_width=True)
        stats = class_stats_to_frame(result.class_stats)
        if stats.empty:
            st.info("没有识别到非背景语义类别。")
        else:
            st.dataframe(stats, use_container_width=True, hide_index=True)

elif method == "R-CNN 目标检测（简化示例）":
    score_threshold = st.sidebar.slider("分类置信度阈值", 0.05, 0.90, 0.25, step=0.05)
    max_regions = st.sidebar.slider("候选区域数量", 4, 32, 14, step=2)
    with right:
        st.subheader("R-CNN 简化结果")
        st.caption("候选区域由传统图像处理产生，每个区域单独送入 ImageNet 分类器。")
        with st.spinner("正在逐区域分类..."):
            result = run_rcnn_style(
                image,
                score_threshold=score_threshold,
                max_regions=max_regions,
            )
        st.image(result.visualization, caption=f"耗时 {result.elapsed_ms:.1f} ms", use_column_width=True)
        show_detection_table(result.detections)

elif method == "Fast R-CNN 目标检测（RoI Head 示例）":
    score_threshold = st.sidebar.slider("检测置信度阈值", 0.05, 0.90, 0.45, step=0.05)
    with right:
        st.subheader("Fast R-CNN 风格结果")
        st.caption("外部候选区域 + 共享 backbone 特征 + Faster R-CNN 的 RoI Head，不启用 RPN。")
        with st.spinner("正在运行共享特征与 RoI Head..."):
            result = run_fast_rcnn_style(image, score_threshold=score_threshold)
        st.image(result.visualization, caption=f"耗时 {result.elapsed_ms:.1f} ms", use_column_width=True)
        show_detection_table(result.detections)

elif method == "Faster R-CNN 目标检测":
    score_threshold = st.sidebar.slider("检测置信度阈值", 0.05, 0.90, 0.45, step=0.05)
    with right:
        st.subheader("Faster R-CNN 结果")
        with st.spinner("正在运行 RPN + RoI Head..."):
            result = run_faster_rcnn(image, score_threshold=score_threshold)
        st.image(result.visualization, caption=f"耗时 {result.elapsed_ms:.1f} ms", use_column_width=True)
        show_detection_table(result.detections)

elif method == "Mask R-CNN 实例分割":
    score_threshold = st.sidebar.slider("实例置信度阈值", 0.05, 0.90, 0.50, step=0.05)
    mask_threshold = st.sidebar.slider("Mask 阈值", 0.25, 0.80, 0.50, step=0.05)
    with right:
        st.subheader("Mask R-CNN 结果")
        with st.spinner("正在运行检测框与实例 mask..."):
            result = run_mask_rcnn(
                image,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
            )
        st.image(result.visualization, caption=f"耗时 {result.elapsed_ms:.1f} ms", use_column_width=True)
        show_instance_table(result.instances)

else:
    with right:
        st.subheader("方法性能对比")
        st.dataframe(pd.DataFrame(METHOD_COMPARISON), use_container_width=True, hide_index=True)

        st.subheader("当前图片快速 benchmark")
        selected = st.multiselect(
            "选择要实际运行的方法",
            ["FCN", "R-CNN", "Fast R-CNN", "Faster R-CNN", "Mask R-CNN"],
            default=["FCN", "Faster R-CNN"],
        )
        score_threshold = st.slider("benchmark 置信度阈值", 0.05, 0.90, 0.45, step=0.05)
        mask_threshold = st.slider("benchmark mask 阈值", 0.25, 0.80, 0.50, step=0.05)
        if st.button("运行 benchmark", type="primary", disabled=not selected):
            with st.spinner("正在依次运行所选模型..."):
                bench = run_benchmark(image, selected, score_threshold, mask_threshold)
            st.dataframe(bench, use_container_width=True, hide_index=True)

        with st.expander("Streamlit Cloud 部署建议", expanded=True):
            for tip in cloud_tips():
                st.markdown(f"- {tip}")
