import torch
import numpy as np
import itertools
import logging

from .kqi import KQI


class Conv2d(torch.nn.Conv2d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_size = x.shape

        x_new = self.forward(x)
        assert x_new.shape[-3] == self.out_channels
        if self.padding[0] or self.padding[1]:
            raise NotImplementedError(f"padding is not supported")
        else:
            KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size) * self.in_channels

        return x_new


    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if self.padding[0] or self.padding[1]:
            raise NotImplementedError(f"padding is not supported")
        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                _, H, W = volume.shape
                for c,i,j in itertools.product(range(self.in_channels), range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1])):
                    volume_backward[c, i:H*self.stride[0]+i:self.stride[0], j:W*self.stride[1]+j:self.stride[1]] += self.out_channels + (volume / np.prod(self.kernel_size) / self.in_channels).sum(dim=0)
            
            for c,i,j in itertools.product(range(self.in_channels), range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1])):
                KQI.kqi += self.KQI_formula((volume[0] / np.prod(self.kernel_size) / self.in_channels), volume_backward[c, i:H*self.stride[0]+i:self.stride[0], j:W*self.stride[1]+j:self.stride[1]]) * self.out_channels
        
        logging.debug(f'Conv2d: KQI={KQI.kqi}, node={np.product(volume.shape)}, volume={volume.sum()}')
        return volume_backward