import torch
import numpy as np

from .kqi import KQI


class Dropout(torch.nn.Dropout, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        return super().KQIforward(x)
    

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        return super().KQIbackward(volume, volume_backward)