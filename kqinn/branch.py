import torch
import numpy as np
import logging
from typing import Tuple

from .kqi import KQI


class SimplePass(torch.nn.Module, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return x

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'SimplePass: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Branch(KQI):
    def __init__(self, *modules):
        self.modules = modules

    def KQIforward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return (module.KQIforward(x) for module in self.modules)

    def KQIbackward(self, *volumes: Tuple[torch.Tensor]) -> torch.Tensor:
        kqi_snapshot = KQI.kqi.clone()
        volume_forward = 0
        for module, volume in zip(self.modules, volumes):
            volume_forward = module.KQIbackward(volume) + volume_forward
        KQI.kqi = kqi_snapshot
        for module, volume in zip(self.modules, volumes):
            module.KQIbackward(volume, volume_forward)
        return volume_forward
