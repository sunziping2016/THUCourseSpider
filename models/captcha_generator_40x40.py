import torch
import torch.nn as nn

from .captcha_decoder import CaptchaDecoder
from .feature_extractor_40x40 import CaptchFeatureExtractor40x40


class CaptchaGenerator_40x40(nn.Module):
    def __init__(self, encoder_len=256, hidden_size=2048, slide_x=5, total_width=180, lock_classifier=True):
        super(CaptchaGenerator_40x40, self).__init__()
        self.slide_width = 40
        self.slide_x = slide_x
        self.encoder_len = encoder_len
        self.slide_num = (total_width - self.slide_width) // slide_x + 1
        self.features = CaptchFeatureExtractor40x40()
        if lock_classifier:
            for parameter in self.features.children():
                parameter.requires_grad = False
        self.mapper = nn.Sequential(
            nn.Dropout(),
            nn.Linear(encoder_len * self.slide_num, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.decoder = CaptchaDecoder(hidden_size=hidden_size)

    def forward(self, x, device):  # n x 1 x 180 x 40
        chunks = []
        for i in range(self.slide_num):
            left = self.slide_x * i
            chunks.append(x[:, :, :, left:left + self.slide_width])
        characters = torch.stack(chunks).transpose(0, 1).reshape(-1, 1, x.size(-2), self.slide_width)
        features = torch.stack(self.features(characters).chunk(x.size(0))).view(x.size(0), -1)
        features = self.mapper(features)
        return self.decoder(features, device)
