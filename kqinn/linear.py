import torch
import numpy as np
import logging

from .kqi import KQI


class Linear(torch.nn.Linear, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += self.in_features * self.out_features 
        return self.forward(x)


    def KQIbackward(self, volumes: torch.Tensor) -> torch.Tensor:
        volumes_forward = self.out_features + (volumes / self.in_features).sum()
        KQI.kqi += self.KQI_formula(volumes/self.in_features, volumes_forward) * self.in_features
        logging.debug(f'Linear: KQI={KQI.kqi}, node={np.product(volumes.shape)}, volume={volumes.sum()}')
        return torch.ones(self.in_features) * volumes_forward
