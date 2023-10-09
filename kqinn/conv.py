import torch
import numpy as np
import itertools
import logging
import math

from .kqi import KQI

class Conv1d(torch.nn.Conv1d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_size = x.shape
        x_new = self.forward(x)

        if self.padding[0]:
            degree = self._degree(x.shape, x_new.shape)
            KQI.W += degree.sum() * self.out_channels * self.in_channels
        else:
            KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size) * self.in_channels

        return x_new


    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        _, H = volume.shape
        
        if self.padding[0]:
            volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1]+2*self.padding[0]))
            if volume_backward is None:
                degree = self._degree(self.input_size, volume.shape)
                for c,i in itertools.product(range(self.in_channels), range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0])):
                    volume_back_padding[c, i:H*self.stride[0]+i:self.stride[0]] += self.out_channels + (volume / degree / self.in_channels).sum(dim=0)
                volume_backward = volume_back_padding[:,self.padding[0]:-self.padding[0]].clone()
                
            for cin, cout, i in itertools.product(range(self.in_channels), range(self.out_channels), range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0])):
                reference = volume_back_padding[cin, i:H*self.stride[0]+i:self.stride[0]]
                reference = volume[cout] / degree / self.in_channels
                reference[max(0,self.padding[0]-i):min(H,H-i+self.padding[0])] = volume_backward[cin, max(0,i-self.padding[0]):min(H,H+i-self.padding[0])]
                KQI.kqi += self.KQI_formula((volume[cout] / degree / self.in_channels), reference)
        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                for c,i in itertools.product(range(self.in_channels), range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0])):
                    volume_backward[c, i:H*self.stride[0]+i:self.stride[0]] += self.out_channels + (volume / np.prod(self.kernel_size) / self.in_channels).sum(dim=0)
            
            for cin, cout,i in itertools.product(range(self.in_channels), range(self.out_channels), range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0])):
                KQI.kqi += self.KQI_formula((volume[cout] / np.prod(self.kernel_size) / self.in_channels), volume_backward[cin, i:H*self.stride[0]+i:self.stride[0]])
        
        logging.debug(f'Conv1d: KQI={KQI.kqi}, node={np.product(volume.shape)}, volume={volume.sum()}')
        return volume_backward
    

    def _degree(self, input_size, output_size):
        _, HI = input_size
        _, HO = output_size

        degree = torch.zeros(HO)
        for i in range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]):
            Hleft = max(0, math.ceil((self.padding[0]-i) / self.stride[0]))
            Hright = min(HO, math.ceil((HI-i+self.padding[0]) / self.stride[0]))
            
            degree[Hleft:Hright] += 1

        return degree
    
class Conv2d(torch.nn.Conv2d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_size = x.shape
        x_new = self.forward(x)

        if self.padding[0] or self.padding[1]:
            degree = self._degree(x.shape, x_new.shape)
            KQI.W += degree.sum() * self.out_channels * self.in_channels
        else:
            KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size) * self.in_channels

        return x_new


    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        _, H, W = volume.shape
        
        if self.padding[0] or self.padding[1]:
            volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1]+2*self.padding[0],  self.input_size[2]+2*self.padding[1]))
            if volume_backward is None:
                degree = self._degree(self.input_size, volume.shape)
                for c,i,j in itertools.product(range(self.in_channels), range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1])):
                    volume_back_padding[c, i:H*self.stride[0]+i:self.stride[0], j:W*self.stride[1]+j:self.stride[1]] += self.out_channels + (volume / degree / self.in_channels).sum(dim=0)
                volume_backward = volume_back_padding[:,self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]].clone()
                
            for cin, cout,i,j in itertools.product(range(self.in_channels), range(self.out_channels), range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1])):
                reference = volume_back_padding[cin, i:H*self.stride[0]+i:self.stride[0], j:W*self.stride[1]+j:self.stride[1]]
                reference = volume[cout] / degree / self.in_channels
                reference[max(0,self.padding[0]-i):min(H,H-i+self.padding[0]), max(0,self.padding[1]-j):min(W,W-j+self.padding[1])] = volume_backward[cin, max(0,i-self.padding[0]):min(H,H+i-self.padding[0]), max(0,j-self.padding[1]):min(W,W+j-self.padding[1])]
                KQI.kqi += self.KQI_formula((volume[cout] / degree / self.in_channels), reference)
        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                for c,i,j in itertools.product(range(self.in_channels), range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1])):
                    volume_backward[c, i:H*self.stride[0]+i:self.stride[0], j:W*self.stride[1]+j:self.stride[1]] += self.out_channels + (volume / np.prod(self.kernel_size) / self.in_channels).sum(dim=0)
            
            for cin, cout,i,j in itertools.product(range(self.in_channels), range(self.out_channels), range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1])):
                KQI.kqi += self.KQI_formula((volume[cout] / np.prod(self.kernel_size) / self.in_channels), volume_backward[cin, i:H*self.stride[0]+i:self.stride[0], j:W*self.stride[1]+j:self.stride[1]])
        
        logging.debug(f'Conv2d: KQI={KQI.kqi}, node={np.product(volume.shape)}, volume={volume.sum()}')
        return volume_backward
    

    def _degree(self, input_size, output_size):
        _, HI, WI = input_size
        _, HO, WO = output_size

        degree = torch.zeros(HO, WO)
        for i,j in itertools.product(range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1])):
            Hleft = max(0, math.ceil((self.padding[0]-i) / self.stride[0]))
            Hright = min(HO, math.ceil((HI-i+self.padding[0]) / self.stride[0]))
            Wleft = max(0, math.ceil((self.padding[1]-j) / self.stride[1]))
            Wright = min(WO, math.ceil((WI-j+self.padding[1]) / self.stride[1]))
            degree[Hleft:Hright, Wleft:Wright] += 1

        return degree
    
class Conv3d(torch.nn.Conv3d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_size = x.shape
        x_new = self.forward(x)

        if self.padding[0] or self.padding[1] or self.padding[2]:
            degree = self._degree(x.shape, x_new.shape)
            KQI.W += degree.sum() * self.out_channels * self.in_channels
        else:
            KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size) * self.in_channels

        return x_new


    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        _, H, W, L = volume.shape
        
        if self.padding[0] or self.padding[1] or self.padding[2]:
            volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1]+2*self.padding[0],  self.input_size[2]+2*self.padding[1], self.input_size[3]+2*self.padding[2]))
            if volume_backward is None:
                degree = self._degree(self.input_size, volume.shape)
                for c,i,j,k in itertools.product(range(self.in_channels), range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2]*self.dilation[2], self.dilation[2])):
                    volume_back_padding[c, i:H*self.stride[0]+i:self.stride[0], j:W*self.stride[1]+j:self.stride[1], k:L*self.stride[2]+k:self.stride[2]] += self.out_channels + (volume / degree / self.in_channels).sum(dim=0)
                volume_backward = volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1], self.padding[2]:-self.padding[2]].clone()
                
            for cin,cout,i,j,k in itertools.product(range(self.in_channels), range(self.out_channels), range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2]*self.dilation[2], self.dilation[2])):
                reference = volume_back_padding[cin, i:H*self.stride[0]+i:self.stride[0], j:W*self.stride[1]+j:self.stride[1], k:L*self.stride[2]+k:self.stride[2]]
                reference = volume[cout] / degree / self.in_channels
                reference[max(0,self.padding[0]-i):min(H,H-i+self.padding[0]), max(0,self.padding[1]-j):min(W,W-j+self.padding[1]), max(0,self.padding[2]-k):min(L,L-k+self.padding[2])] = volume_backward[cin, max(0,i-self.padding[0]):min(H,H+i-self.padding[0]), max(0,j-self.padding[1]):min(W,W+j-self.padding[1]), max(0,k-self.padding[2]):min(L,L+k-self.padding[2])]
                KQI.kqi += self.KQI_formula((volume[cout] / degree / self.in_channels), reference)
        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                for c,i,j,k in itertools.product(range(self.in_channels), range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2]*self.dilation[2], self.dilation[2])):
                    volume_backward[c, i:H*self.stride[0]+i:self.stride[0], j:W*self.stride[1]+j:self.stride[1], k:L*self.stride[2]+k:self.stride[2]] += self.out_channels + (volume / np.prod(self.kernel_size) / self.in_channels).sum(dim=0)
            
            for cin, cout,i,j,k in itertools.product(range(self.in_channels), range(self.out_channels), range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2]*self.dilation[2], self.dilation[2])):
                KQI.kqi += self.KQI_formula((volume[cout] / np.prod(self.kernel_size) / self.in_channels), volume_backward[cin, i:H*self.stride[0]+i:self.stride[0], j:W*self.stride[1]+j:self.stride[1], k:L*self.stride[2]+k:self.stride[2]])
        
        logging.debug(f'Conv3d: KQI={KQI.kqi}, node={np.product(volume.shape)}, volume={volume.sum()}')
        return volume_backward
    

    def _degree(self, input_size, output_size):
        _, HI, WI, LI = input_size
        _, HO, WO, LO = output_size

        degree = torch.zeros(HO, WO, LO)
        for i,j,k in itertools.product(range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2]*self.dilation[2], self.dilation[2])):
            Hleft = max(0, math.ceil((self.padding[0]-i) / self.stride[0]))
            Hright = min(HO, math.ceil((HI-i+self.padding[0]) / self.stride[0]))
            Wleft = max(0, math.ceil((self.padding[1]-j) / self.stride[1]))
            Wright = min(WO, math.ceil((WI-j+self.padding[1]) / self.stride[1]))
            Lleft = max(0, math.ceil((self.padding[2]-k) / self.stride[2]))
            Lright = min(LO, math.ceil((LI-k+self.padding[2]) / self.stride[2]))
            
            degree[Hleft:Hright, Wleft:Wright, Lleft:Lright] += 1

        return degree