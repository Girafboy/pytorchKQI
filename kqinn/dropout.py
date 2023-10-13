import torch
import numpy as np
import logging

from .kqi import KQI


class Dropout(torch.nn.Dropout, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume

        logging.debug(f'Dropout: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward
