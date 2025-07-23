"""
PyTorch ResNet18 CNN model for binary thyroid nodule classification.
"""

import torch
import torch.nn as nn
from torchvision import models

class ThyroidNoduleResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.base = models.resnet18(pretrained=pretrained)
        # Change final fc for binary classification
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.base(x)

def get_model(pretrained=True):
    return ThyroidNoduleResNet18(pretrained=pretrained)

if __name__ == '__main__':
    # Test stub
    model = get_model()
    print(model)
