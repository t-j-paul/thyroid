import pytest
import torch
from src.models.model import get_model

def test_model_forward():
    model = get_model(pretrained=False)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (1, 2)
