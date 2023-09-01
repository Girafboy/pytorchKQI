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
            padding_values = (self.padding[1],self.padding[1],self.padding[0],self.padding[0])
            x_padded = torch.nn.functional.pad(x, padding_values)
            x_new = self.forward(x_padded)
        
            KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size) * self.in_channels
            _,H_new,W_new = x_new.shape

            connections_to_padding = (4 * (2*self.kernel_size[0]-1)+ (2*H_new+2*W_new-4)*self.kernel_size[0]) * self.in_channels
            KQI.W -= connections_to_padding

        else:
            x_new = self.forward(x)
            KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size) * self.in_channels

        return x_new


    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if self.padding[0] or self.padding[1]:
            if volume_backward is None:
                C, H, W = volume.shape
                
                volume_backward = torch.zeros_like(volume)
                
                _,H_back, W_back = volume_backward.shape
                
            
                for c, i, j in itertools.product(
                range(C),
                range(-self.padding[0], H_back * self.stride[0] + self.padding[0], self.dilation[0]),
                range(-self.padding[1], W_back * self.stride[1] + self.padding[1], self.dilation[1])):
                
                
                    output_h_start = max(0, -i)
                    output_h_end = min(H_back, H_back - (i + self.kernel_size[0] - 1))
                    output_w_start = max(0, -j)
                    output_w_end = min(W_back, W_back - (j + self.kernel_size[1] - 1))

                
                    volume_backward[c, output_h_start:output_h_end:self.stride[0], output_w_start:output_w_end:self.stride[1]] += (
                    self.out_channels + (volume / (self.kernel_size[0] * self.kernel_size[1]) / self.in_channels).sum(dim=0))
            
           
                    input_h_start = max(0, i)
                    input_h_end = min(H, i + self.kernel_size[0] * self.dilation[0])
                    input_w_start = max(0, j)
                    input_w_end = min(W, j + self.kernel_size[1] * self.dilation[1])

                    KQI.kqi += self.KQI_formula((volume[c, input_h_start:input_h_end, input_w_start:input_w_end] /
                (self.kernel_size[0] * self.kernel_size[1]) / self.in_channels),
                volume_backward[c, output_h_start:output_h_end:self.stride[0], output_w_start:output_w_end:self.stride[1]]) * self.out_channels

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