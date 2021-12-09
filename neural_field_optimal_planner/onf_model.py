import torch
from torch import nn


class ONF(nn.Module):
    def __init__(self, mean, sigma):
        super().__init__()
        self.mlp = nn.Sequential(*[
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        ])
        self._mean = mean
        self._sigma = sigma
        self.encoding_layer = nn.Linear(2, 100, bias=True)
        torch.nn.init.normal_(self.encoding_layer.weight)

    def forward(self, x):
        x = (x - self._mean) / self._sigma
        x = self.encoding_layer(x)
        x = torch.sin_(x)
        x = self.mlp(x)
        return x
