import torch
import numpy as np
import logging

from .kqi import KQI


class ReLU(torch.nn.ReLU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)
    

    def KQIbackward(self, volumes: torch.Tensor, kqi: float) -> (torch.Tensor, float):
        volumes_forward = volumes + 1
        kqi += self.KQI_formula(volumes, volumes_forward)
        logging.debug(f'ReLU: KQI={kqi}, node={np.product(volumes.shape)}, volume={volumes.sum()}')
        return volumes_forward, kqi
