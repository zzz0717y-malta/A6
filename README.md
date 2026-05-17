# FCN / R-CNN / Mask R-CNN Streamlit Demo

这是一个可以部署到 Streamlit Cloud 的计算机视觉演示项目，包含：

- FCN 语义分割示例：`FCN-ResNet50`
- R-CNN 目标检测简化示例：候选区域 + 逐区域分类
- Fast R-CNN 风格示例：外部候选区域 + 共享 backbone + RoI Head
- Faster R-CNN 目标检测示例：`Faster R-CNN MobileNetV3 FPN`
- Mask R-CNN 实例分割示例：`Mask R-CNN ResNet50 FPN`
- 方法性能对比表，以及当前图片的快速 benchmark

## 本地运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

首次运行会自动下载 torchvision 预训练权重。建议上传真实照片，内置示例主要用于确认页面和推理流程可运行。

## Streamlit Cloud 部署

1. 将本项目上传到 GitHub。
2. 在 Streamlit Cloud 中选择该仓库。
3. App file 选择 `app.py`。
4. Python 版本建议使用 `3.11`。
5. 点击 Deploy。

如果 Streamlit Cloud 冷启动较慢，这是 PyTorch 和预训练权重下载导致的正常现象。页面采用延迟加载：只有选择某个方法时才会加载对应模型。

## 项目结构

```text
.
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── src/
    └── vision_demo/
        ├── detection.py
        ├── image_utils.py
        ├── mask_rcnn.py
        ├── models.py
        ├── performance.py
        └── segmentation.py
```

## 说明

R-CNN 与 Fast R-CNN 的经典论文实现通常需要候选区域算法、训练好的检测头和专门数据集。本项目为了适配课堂展示与 Streamlit Cloud，采用可运行、轻量、便于解释的教学式实现；Faster R-CNN 与 Mask R-CNN 使用 torchvision 的真实 COCO 预训练模型。
