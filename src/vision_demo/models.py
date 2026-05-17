import torch
import streamlit as st

from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    MaskRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_torch():
    torch.set_grad_enabled(False)
    try:
        torch.set_num_threads(2)
    except RuntimeError:
        pass


@st.cache_resource(show_spinner=False)
def load_fcn():
    configure_torch()
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights, progress=False)
    model.to(get_device()).eval()
    return model, weights


@st.cache_resource(show_spinner=False)
def load_faster_rcnn():
    configure_torch()
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, progress=False)
    model.to(get_device()).eval()
    return model, weights


@st.cache_resource(show_spinner=False)
def load_mask_rcnn():
    configure_torch()
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights, progress=False)
    model.to(get_device()).eval()
    return model, weights


@st.cache_resource(show_spinner=False)
def load_classifier():
    configure_torch()
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights, progress=False)
    model.to(get_device()).eval()
    return model, weights
