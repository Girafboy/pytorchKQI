import torch
import numpy as np
import itertools
import logging

from .kqi import KQI


class kqi_add(KQI):
    def KQIforward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x0.shape) + np.prod(x1.shape)
        return x0 + x1
    

    def KQI_formula(self, volumes: torch.Tensor, volumes_forward: torch.Tensor) -> float:
        pos = torch.where(volumes != 0)
        return (2 * (-volumes[pos] / KQI.W * np.log2(volumes[pos] / volumes_forward[pos]))).sum()


    def KQIbackward(self, volumes: torch.Tensor, kqi: float, volumes_forward=torch.tensor(-1)) -> (torch.Tensor, float):
        if volumes_forward.dim()==0:
            volumes_forward = volumes + torch.ones_like(volumes)
        kqi += self.KQI_formula(volumes/2, volumes_forward)
        logging.debug(f'kqi_add: KQI={kqi}, node={np.product(volumes.shape)}, volume={volumes.sum()}')
        return volumes_forward, kqi


def Combine(backward_func1, backward_func2, volumes1: torch.Tensor, volumes2: torch.Tensor, kqi: float) -> (torch.Tensor, float):
    volumes_forward1, _ = backward_func1(volumes1, kqi)
    volumes_forward2, _ = backward_func2(volumes1, kqi)
    if volumes_forward1.dim() >= volumes_forward2.dim():
        volumes_forward = volumes_forward1 + volumes_forward2.reshape(np.shape(volumes_forward1))
    else:
        volumes_forward = volumes_forward2 + volumes_forward1.reshape(np.shape(volumes_forward2))

    # 需要对所有的KQIbackward()函数增加一个参数，当传入了volumes_forward时，直接计算kqi
    _, kqi = backward_func1(volumes1, kqi, volumes_forward)
    _, kqi = backward_func2(volumes2, kqi, volumes_forward)

    return volumes_forward, kqi
