import logging

import numpy as np
import torch

from .kqi import KQI


class LayerNorm(torch.nn.LayerNorm, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        normalized_shape = self.normalized_shape
        input_shape = tuple(x.shape)

        index = None
        if input_shape[len(input_shape) - len(normalized_shape):len(input_shape)] == normalized_shape:
            index = len(input_shape) - len(normalized_shape)

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
        if input_shape[len(input_shape) - len(normalized_shape):len(input_shape)] == normalized_shape:
            index = len(input_shape) - len(normalized_shape)

        if index is None:
            raise ValueError("Normalized shape is not a subsequence of input shape.")

        if index == 0:
            if volume_backward is None:
                volume_backward = torch.ones(input_shape) * (
                        np.prod(input_shape) + (volume / np.prod(input_shape)).sum())
            for vol in volume_backward.flatten():
                KQI.kqi += self.KQI_formula(volume / np.prod(input_shape), vol)
        else:
            if volume_backward is None:
                volume_backward = torch.ones(input_shape) * (
                        np.prod(normalized_shape) + (volume / np.prod(input_shape)).sum())
            tmp = volume
            for i in range(index):
                tmp = tmp[0, :]
            for vol in volume_backward.flatten():
                KQI.kqi += self.KQI_formula(tmp / np.prod(normalized_shape), vol)

        logging.debug(f'LayerNorm: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward
    

class GroupNorm(torch.nn.GroupNorm, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape[-3:]) ** 2  / self.num_groups
        return self.forward(x)
    
    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        num = np.prod(volume.shape[-3:]) / self.num_groups
        stride = int(self.num_channels / self.num_groups)
        if volume_backward is None:
            volume_backward = torch.zeros(volume.shape)
            for i in range(0,self.num_channels,stride):
                volume_backward[:,i:i+stride,:,:] += (num + (volume[:,i:i+stride,:,:] / num).sum())

        for i in range(0,self.num_channels,stride):
            for vol in volume_backward[:,i:i+stride,:,:].flatten():
                KQI.kqi += self.KQI_formula(volume[:,i:i+stride,:,:] / num, vol)
        logging.debug(f'GroupNorm: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward