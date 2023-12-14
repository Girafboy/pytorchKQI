import torch
import numpy as np
import itertools
import logging

from .kqi import KQI

class BatchNorm2d(torch.nn.BatchNorm2d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape[-2:]) ** 2 * x.shape[-3]
        return self.forward(x)
    
    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        C, H, W = volume.shape[-3:]
        
        if volume_backward is None:
            volume_backward = torch.zeros(volume.shape)
            for i, j, k in itertools.product(range(H), range(W), range(C)):
                volume_backward[0,k,i,j] += H * W + (volume[0,k,:,:] / H / W).sum()

        for k in range(C):
            for vol in volume_backward[0,k,:,:].flatten():
                KQI.kqi += self.KQI_formula(volume[0,k,:,:] / H / W, vol)
        
        logging.debug(f'BatchNorm2d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward
