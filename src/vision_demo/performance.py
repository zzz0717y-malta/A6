METHOD_COMPARISON = [
    {
        "方法": "FCN",
        "任务": "语义分割",
        "核心思想": "全卷积网络输出每个像素的类别",
        "本项目示例": "FCN-ResNet50 / VOC 类别",
        "速度": "中等",
        "输出": "类别 mask",
        "适合": "道路、人体、背景等整体区域理解",
    },
    {
        "方法": "R-CNN",
        "任务": "目标检测",
        "核心思想": "候选区域逐个裁剪并分类",
        "本项目示例": "传统候选框 + MobileNetV3 分类器",
        "速度": "慢",
        "输出": "bbox + 分类",
        "适合": "理解两阶段检测的原始流程",
    },
    {
        "方法": "Fast R-CNN",
        "任务": "目标检测",
        "核心思想": "整图只提一次特征，RoI Pooling/Align 后分类回归",
        "本项目示例": "外部候选框 + Faster R-CNN RoI Head",
        "速度": "较快",
        "输出": "bbox + 分类",
        "适合": "展示共享 backbone 的加速效果",
    },
    {
        "方法": "Faster R-CNN",
        "任务": "目标检测",
        "核心思想": "RPN 自动学习候选框，再接 RoI Head",
        "本项目示例": "Faster R-CNN MobileNetV3 FPN",
        "速度": "快",
        "输出": "bbox + 分类",
        "适合": "通用目标检测",
    },
    {
        "方法": "Mask R-CNN",
        "任务": "实例分割",
        "核心思想": "在 Faster R-CNN 上增加 mask 分支",
        "本项目示例": "Mask R-CNN ResNet50 FPN",
        "速度": "较慢",
        "输出": "bbox + 分类 + instance mask",
        "适合": "区分同类物体的不同实例",
    },
]


def cloud_tips():
    return [
        "页面只加载当前选择的方法，减少 Streamlit Cloud 的内存压力。",
        "推荐上传最长边 640 左右的图片；更大图片会显著增加 CPU 推理时间。",
        "第一次打开会下载 PyTorch 权重，冷启动较慢；后续同一会话会走缓存。",
        "若 Cloud 内存不足，可优先展示 FCN、R-CNN 简化示例和 Faster R-CNN MobileNetV3。",
    ]
