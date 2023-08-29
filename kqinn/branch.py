import torch
import numpy as np
import logging
from typing import Tuple, Callable
import inspect

from .kqi import KQI


class SimplePass(torch.nn.Module, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return x


    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'SimplePass: KQI={KQI.kqi}, node={np.product(volume.shape)}, volume={volume.sum()}')
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


class Merge(KQI):
    def __init__(self, func: Callable):
        self.func = func


    def KQIforward(self, *xs: Tuple[torch.Tensor]) -> torch.Tensor:
        KQI.W += sum(map(lambda x: np.prod(x.shape), xs))
        return self.func(*xs)


    def KQIbackward(self, volume: torch.Tensor) -> Tuple[torch.Tensor]:
        num_args = len(inspect.signature(self.func).parameters)
        volume_backward = 1 + volume / num_args
        return (volume_backward,) * num_args


# class Add(KQI):
#     def __init__(self, left, right):
#         if left.shape != right.shape:
#             raise ValueError(f'Shape of left {left.shape} is incompatible with right {right.shape}')
#         self.left = left
#         self.right = right
        

#     def KQIforward(self) -> torch.Tensor:
#         KQI.W += np.prod(self.left.shape) + np.prod(self.right.shape)
#         return self.left + self.right


#     def KQIbackward(self, volumes: torch.Tensor) -> (torch.Tensor, torch.Tensor):
#         volumes_forward_left = 1 + volumes / 2
#         volumes_forward_right = 1 + volumes / 2
#         KQI.kqi += self.KQI_formula(volumes/2, volumes_forward_left) + self.KQI_formula(volumes/2, volumes_forward_right) 
#         logging.debug(f'Linear: KQI={KQI.kqi}, node={np.product(volumes.shape)}, volume={volumes.sum()}')
#         return volumes_forward_left, volumes_forward_right
