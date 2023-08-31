import torch
import numpy as np

from .kqi import KQI


class Sequential(torch.nn.Sequential, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self:
            x = module.KQIforward(x)

        return x


    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        modules = list(reversed(self))
        for module in modules[:-1]:
            volume = module.KQIbackward(volume)
        volume_backward = modules[-1].KQIbackward(volume, volume_backward)

        return volume_backward
