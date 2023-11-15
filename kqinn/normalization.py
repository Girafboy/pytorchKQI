import logging

import numpy as np
import torch

from .kqi import KQI


class LayerNorm(torch.nn.LayerNorm, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        normalized_shape = self.normalized_shape
        input_shape = tuple(x.shape)

        index = None
        for i in range(len(input_shape) - len(normalized_shape) + 1):
            if input_shape[i:i + len(normalized_shape)] == normalized_shape:
                index = i
                break

        if index is None:
            raise ValueError("Normalized shape is not a subsequence of input shape.")

        if index == 0:
            KQI.W += np.prod(input_shape) ** 2
        else:
            KQI.W += np.prod(input_shape[:index]) * (np.prod(normalized_shape) ** 2)

        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        normalized_shape = self.normalized_shape
        input_shape = tuple(volume.shape)

        index = None
        for i in range(len(input_shape) - len(normalized_shape) + 1):
            if input_shape[i:i + len(normalized_shape)] == normalized_shape:
                index = i
                break

        if index is None:
            raise ValueError("Normalized shape is not a subsequence of input shape.")

        if index == 0:
            if volume_backward is None:
                volume_backward = torch.ones(input_shape) * (np.prod(input_shape) + (volume / np.prod(input_shape)).sum())
            for vol in volume_backward.flatten():
                KQI.kqi += self.KQI_formula(volume / np.prod(input_shape), vol)
        else:
            if volume_backward is None:
                volume_backward = torch.ones(input_shape) * (np.prod(normalized_shape) + (volume / np.prod(input_shape)).sum())
            for vol in volume_backward.flatten():
                KQI.kqi += self.KQI_formula(volume / np.prod(input_shape), vol)

        logging.debug(f'Threshold: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward
