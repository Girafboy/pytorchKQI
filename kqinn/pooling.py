import torch
import numpy as np

from .kqi import KQI


class MaxPool2d(torch.nn.MaxPool2d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        return super().KQIforward(x)
    

    def KQIbackward(self, volumes: torch.Tensor) -> torch.Tensor:
        return super().KQIbackward(volumes)
