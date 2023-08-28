import torch
import numpy as np

from .kqi import KQI


class Sequential(torch.nn.Sequential, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self:
            x = module.KQIforward(x)

        return x
        
    def KQIbackward(self, volumes: torch.Tensor) -> torch.Tensor:
        for module in reversed(self):
            volumes = module.KQIbackward(volumes)

        return volumes
