import torch
import numpy as np

from .kqi import KQI


class Dropout(torch.nn.Dropout, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        return super().KQIforward(x)
    

    def KQIbackward(self, volumes: torch.Tensor) -> torch.Tensor:
        return super().KQIbackward(volumes)