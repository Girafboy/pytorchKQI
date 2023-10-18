import torch
import numpy as np
import logging


class KQI:
    W = 0
    kqi = 0

    def KQI(self, x: torch.Tensor) -> float:
        KQI.W = torch.tensor(np.prod(x.shape), dtype=float)
        KQI.kqi = 0

        x = self.KQIforward(x)
        volume = self.KQIbackward(torch.zeros_like(x))
        KQI.kqi += self.KQI_formula(volume, KQI.W)

        logging.debug(f'Root: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        logging.debug(f'Total volume = {KQI.W}')
        return KQI.kqi

    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required KQIforward function')

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required KQIbackward function')

    def KQI_formula(self, volume: torch.Tensor, volume_backward: torch.Tensor) -> float:
        if volume.shape != volume_backward.shape and volume_backward.dim():
            raise ValueError(f'Shape of volume {volume.shape} is incompatible with volume_backward {volume_backward.shape}')
        if volume.dim():
            pos = torch.where(volume != 0)
            if volume_backward.dim():
                return (- volume[pos] / KQI.W * np.log2(volume[pos] / volume_backward[pos])).sum()
            else:
                return (- volume[pos] / KQI.W * np.log2(volume[pos] / volume_backward)).sum()
        else:
            return - volume / KQI.W * np.log2(volume / volume_backward) if volume else 0
