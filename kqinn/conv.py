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
            
            _, H, W = x_new.shape
            x_padding_size = (x.shape[0], x.shape[1]+2*self.padding[0], x.shape[2]+2*self.padding[1])
            degree = torch.zeros(x_padding_size)
            
            for c,i,j in itertools.product(range(self.in_channels), range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1])):
                degree[c, i:H*self.stride[0]+i:self.stride[0], j:W*self.stride[1]+j:self.stride[1]] += 1
            
            degree = degree[:,self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]]
            
            KQI.W += degree.sum()*self.out_channels
            
        else:
        
            KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size) * self.in_channels
            
        return x_new


    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if self.padding[0] or self.padding[1]:
            
            
            if volume_backward is None:
                _, H, W = volume.shape

                size_backward = (self.input_size[0], self.input_size[1]+2*self.padding[0], self.input_size[2]+2*self.padding[1])
                volume_backward = torch.zeros(size_backward)
                

                degree = torch.zeros_like(volume)

                for c,i,j in itertools.product(range(self.in_channels), range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1])):
                    degree[c, max(0,i-self.padding[0]):min(H,H+i-self.padding[0]),max(0,j-self.padding[1]):min(W,W+j-self.padding[1])] += 1
                
                
                for c,i,j in itertools.product(range(self.in_channels), range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1])):
                    volume_backward[c, i:H*self.stride[0]+i:self.stride[0], j:W*self.stride[1]+j:self.stride[1]] += self.out_channels + (volume / degree[0] / self.in_channels).sum(dim=0)
                
                
                volume_backward = volume_backward[:,self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]]
                

            for c,i,j in itertools.product(range(self.in_channels), range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1])):
                volume_back_virtual = torch.zeros(size_backward)
            
                volume_back_virtual[c, i:H*self.stride[0]+i:self.stride[0], j:W*self.stride[1]+j:self.stride[1]] += (volume[0] / degree[0] / self.in_channels)
                
                volume_back_virtual[c, i:H*self.stride[0]+i:self.stride[0], j:W*self.stride[1]+j:self.stride[1]][max(0,self.padding[0]-i):min(H,H-i+self.padding[0]), max(0,self.padding[0]-j):min(W,W-j+self.padding[1])]= volume_backward[c, max(0,i-self.padding[0]):min(H,H+i-self.padding[0]),max(0,j-self.padding[1]):min(W,W+j-self.padding[1])]
               
                KQI.kqi += self.KQI_formula((volume[0] / degree[0] / self.in_channels), volume_back_virtual[c, i:H*self.stride[0]+i:self.stride[0], j:W*self.stride[1]+j:self.stride[1]]) * self.out_channels
                
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