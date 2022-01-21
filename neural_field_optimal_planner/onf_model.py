import torch
from torch import nn


class ONF(nn.Module):
    def __init__(self, mean, sigma, use_cos=False, use_normal_init=False, bias=True):
        super().__init__()
        feature_dim = 200 if use_cos else 100
        self._use_cos = use_cos
        self.mlp = nn.Sequential(*[
            nn.Linear(feature_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        ])
        self.mlp2 = nn.Sequential(*[
            nn.Linear(100 + feature_dim, 1)
        ])
        self._mean = mean
        self._sigma = sigma
        self.encoding_layer = nn.Linear(2, feature_dim, bias=bias)
        if use_normal_init:
            torch.nn.init.normal_(self.encoding_layer.weight)

    def forward(self, x):
        x = (x - self._mean) / self._sigma
        x = self.encoding_layer(x)
        if self._use_cos:
            x = torch.cat([torch.sin(x[:, :100]), torch.cos(x[:, 100:])], dim=1)
        else:
            x = torch.sin(x)
        input_x = x
        x = self.mlp(input_x)
        x = torch.cat([x, input_x], dim=1)
        x = self.mlp2(x)
        return x
