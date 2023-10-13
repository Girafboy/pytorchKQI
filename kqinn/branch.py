import torch
import numpy as np
import logging
from typing import Tuple

from .kqi import KQI


class EmptyModule(KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume
        return volume_backward


class DefaultMerge(KQI):
    def KQIforward(self, xs: Tuple[torch.Tensor]) -> torch.Tensor:
        self.args_num = len(xs)
        shape = xs[0].shape
        for x in xs:
            if shape != x.shape:
                raise ValueError('DefaultMerge only supports xs with the same shape. Otherwise, please implement a custom merge module.')

        for x in xs:
            KQI.W += np.prod(x.shape)
        return sum(xs)

    def KQIbackward(self, volume: torch.Tensor, volume_backwards: Tuple[torch.Tensor] = None) -> Tuple[torch.Tensor]:
        if volume_backwards is None:
            volume_backwards = (volume/self.args_num+1, ) * self.args_num
        else:
            volume_backwards = [volume/self.args_num+1 if volume_backward is None else volume_backward for volume_backward in volume_backwards]

        for volume_backward in volume_backwards:
            KQI.kqi += self.KQI_formula(volume/self.args_num, volume_backward)
        logging.debug(f'DefaultMerge: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backwards


class Branch(KQI):
    def __init__(self, *modules, merge=DefaultMerge()):
        self.modules = modules
        self.merge = merge

    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        return self.merge.KQIforward(tuple(module.KQIforward(x) for module in self.modules))

    def KQIbackward(self, volume: torch.Tensor) -> torch.Tensor:
        kqi_snapshot = KQI.kqi.clone()
        volumes = self.merge.KQIbackward(volume)
        volume_forward = sum(module.KQIbackward(volume) for module, volume in zip(self.modules, volumes))

        KQI.kqi = kqi_snapshot
        volumes = self.merge.KQIbackward(volume, (volume_forward if isinstance(module, EmptyModule) else None for module in self.modules))
        for module, volume in zip(self.modules, volumes):
            module.KQIbackward(volume, volume_forward)

        return volume_forward
