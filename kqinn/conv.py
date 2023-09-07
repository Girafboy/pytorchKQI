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
            x_new = self.forward(x)
           
            _,H_new,W_new = x_new.shape
            KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size) * self.in_channels
            
            connections_to_padding = (4 * (2*self.kernel_size[0]-1)+ (2*H_new+2*W_new-8)*self.kernel_size[0]) * self.out_channels
            
            KQI.W -= connections_to_padding
            
            
        else:
            x_new = self.forward(x)
            KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size) * self.in_channels
            
        return x_new


    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if self.padding[0] or self.padding[1]:
            if volume_backward is None:
                
                volume_backward = torch.zeros(self.input_size)
                _, H, W = volume.shape

                # nodes inside
                for c,i,j in itertools.product(range(self.in_channels), range(-1, self.kernel_size[0]*self.dilation[0]-1, self.dilation[0]), range(-self.padding[1], self.kernel_size[1]*self.dilation[1]-self.padding[1], self.dilation[1])):
                    volume_backward[c, self.stride[0]+i:(H-1)*self.stride[0]+i:self.stride[0], self.stride[1]+j:(W-1)*self.stride[0]+j:self.stride[1]] += self.out_channels + (volume[:,1:-1,1:-1] / np.prod(self.kernel_size) / self.in_channels).sum(dim=0)

                # nodes in the corner
                for c, i, j in itertools.product(range(self.in_channels), [0, self.padding[0], H-2*self.padding[0],H-self.padding[0] ], [0, self.padding[1], W-2*self.padding[1], W-self.padding[1]]):
                    i_offset = 0 if i in [0, self.padding[0]] else H-self.padding[0]
                    j_offset = 0 if j in [0, self.padding[1]] else W-self.padding[1]
                    volume_index = (i_offset, j_offset)
                    volume_backward[c, i, j] += self.out_channels + (volume[:, volume_index[0], volume_index[1]] /4 / self.in_channels).sum(dim=0)
                
                
                # nodes on the sides  
                for c,i,j in itertools.product(range(self.in_channels), range(0, self.kernel_size[0]*self.dilation[0]-1, self.dilation[0]), range(-1, self.kernel_size[1]*self.dilation[1]-1, self.dilation[1])):
                    
                    volume_backward[c, i , self.stride[1]+j:(W-1)*self.stride[1]+j:self.stride[1]] += self.out_channels + (volume[:,0,self.padding[1]:W-self.padding[1]] / 6 / self.in_channels).sum(dim=0)
                for c,i,j in itertools.product(range(self.in_channels), range(-1, self.kernel_size[0]*self.dilation[0]-2, self.dilation[0]), range(-1, self.kernel_size[1]*self.dilation[1]-1, self.dilation[1])):
                    
                    volume_backward[c, H-self.padding[0]+i , self.stride[1]+j:(W-1)*self.stride[1]+j:self.stride[1]] += self.out_channels + (volume[:,27,1:27] / 6 / self.in_channels).sum(dim=0)
                for c,i,j in itertools.product(range(self.in_channels), range(-1, self.kernel_size[0]*self.dilation[0]-1, self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1]-1, self.dilation[1])):
                    
                    volume_backward[c, self.stride[0]+i:(H-1)*self.stride[0]+i:self.stride[0] ,j] += self.out_channels + (volume[:,1:27,0] / 6 / self.in_channels).sum(dim=0)
                for c,i,j in itertools.product(range(self.in_channels), range(-1, self.kernel_size[0]*self.dilation[0]-1, self.dilation[0]), range(-1, self.kernel_size[1]*self.dilation[1]-2, self.dilation[1])):
                    
                    volume_backward[c, self.stride[0]+i:(H-1)*self.stride[0]+i:self.stride[0] ,W-self.padding[1]+j] += self.out_channels + (volume[:,1:27,27] / 6 / self.in_channels).sum(dim=0)



            for c,i,j in itertools.product(range(self.in_channels), range(-1, self.kernel_size[0]*self.dilation[0]-1, self.dilation[0]), range(-1, self.kernel_size[1]*self.dilation[1]-1, self.dilation[1])):

                KQI.kqi += self.KQI_formula((volume[0,1:27,1:27] / np.prod(self.kernel_size) / self.in_channels), volume_backward[c, 1+i:H-1+i, 1+j:W-1+j]) * self.out_channels
            
            for c, i, j in itertools.product(range(self.in_channels), [0, self.padding[0], H-2*self.padding[0],H-self.padding[0] ], [0, self.padding[1], W-2*self.padding[1], W-self.padding[1]]):
                i_offset = 0 if i in [0, self.padding[0]] else H-self.padding[0]
                j_offset = 0 if j in [0, self.padding[1]] else W-self.padding[1]
                volume_index = (i_offset, j_offset)
                KQI.kqi += self.KQI_formula((volume[0, volume_index[0], volume_index[1]] / 4 / self.in_channels), volume_backward[c, i,j]) * self.out_channels
            
            for c,i,j in itertools.product(range(self.in_channels), range(0, self.kernel_size[0]*self.dilation[0]-1, self.dilation[0]), range(-1, self.kernel_size[1]*self.dilation[1]-1, self.dilation[1])):
                KQI.kqi += self.KQI_formula((volume[0,0,1:27]/ 6 / self.in_channels), volume_backward[c, i  , 1+j:27+j]) * self.out_channels
            for c,i,j in itertools.product(range(self.in_channels), range(-1, self.kernel_size[0]*self.dilation[0]-2, self.dilation[0]), range(-1, self.kernel_size[1]*self.dilation[1]-1, self.dilation[1])):
                KQI.kqi += self.KQI_formula((volume[0,27,1:27]/ 6 / self.in_channels), volume_backward[c, 27+i , 1+j:27+j]) * self.out_channels
                
            for c,i,j in itertools.product(range(self.in_channels), range(-1, self.kernel_size[0]*self.dilation[0]-1, self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1]-1, self.dilation[1])):
                KQI.kqi += self.KQI_formula((volume[0,1:27,0]/ 6 / self.in_channels), volume_backward[c, 1+i:27+i ,j]) * self.out_channels
                 
            for c,i,j in itertools.product(range(self.in_channels), range(-1, self.kernel_size[0]*self.dilation[0]-1, self.dilation[0]), range(-1, self.kernel_size[1]*self.dilation[1]-2, self.dilation[1])):
                KQI.kqi += self.KQI_formula((volume[0,1:27,27]/ 6 / self.in_channels), volume_backward[c, 1+i:27+i ,27+j]) * self.out_channels
          

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