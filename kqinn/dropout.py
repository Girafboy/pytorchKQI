import torch
import numpy as np

from .kqi import KQI


class Dropout(torch.nn.Dropout, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W *= 1-self.p
        return self.forward(x)
    

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        return super().KQIbackward(volume, volume_backward)