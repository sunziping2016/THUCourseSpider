import torch
import torch.nn as nn

from .feature_extractor_40x40 import CaptchFeatureExtractor40x40


class CaptchaClassifierCNN40x40(nn.Module):
    def __init__(self, n_classes=25):
        super(CaptchaClassifierCNN40x40, self).__init__()
        self.features = CaptchFeatureExtractor40x40()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
