import pytest
import numpy as np
import torch
from src.models.model import get_model
from src.models.inference import grad_cam

def test_grad_cam_hook():
    model = get_model(pretrained=False)
    dummy_input = torch.randn(1, 3, 224, 224)
    dummy_class = 1
    cam = grad_cam(model, dummy_input, dummy_class)
    assert cam.shape == (224, 224)
