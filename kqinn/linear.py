import torch
import numpy as np
import logging

from .kqi import KQI


class Linear(torch.nn.Linear, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += self.in_features * self.out_features
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = torch.ones(self.in_features) * (self.out_features + (volume / self.in_features).sum())

        for vol in volume_backward:
            KQI.kqi += self.KQI_formula(volume/self.in_features, vol)

        logging.debug(f'Linear: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward
