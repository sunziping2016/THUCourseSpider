import torch.nn as nn


class CaptchFeatureExtractor40x40(nn.Sequential):
    def __init__(self):
        super(CaptchFeatureExtractor40x40, self).__init__(
            nn.Conv2d(1, 32, kernel_size=5),   # 32 x 36 x 36
            nn.MaxPool2d(2),                   # 32 x 18 x 18
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),  # 32 x 16 x 16
            nn.MaxPool2d(2),                   # 32 x 8 x 8
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3),  # 64 x 6 x 6
            nn.MaxPool2d(2),                   # 64 x 3 x 3
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=3), # 128 x 1 x 1
            nn.ReLU(inplace=True),
        )
