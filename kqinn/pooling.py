import torch
import numpy as np
import logging
import itertools

from .kqi import KQI


class MaxPool2d(torch.nn.MaxPool2d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        dilation = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)
        stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        padding = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)

        self.input_size = x.shape
        x_new = self.forward(x)
        assert x_new.shape[-3] == x.size(0)

        if padding[0] or padding[1]:
            C_new, H_new, W_new = x_new.shape

            degree = torch.zeros((x.shape[0],x.shape[1]+2*padding[0],x.shape[2]+2*padding[1]))
            for c, i, j in itertools.product(range(x.size(0)),
                                             range(0, kernel_size[0] * dilation[0], dilation[0]),
                                             range(0, kernel_size[1] * dilation[1], dilation[1])):
                degree[c, i:H_new * stride[0] + i:stride[0], j:W_new * stride[1] + j:stride[1]] += 1

            # Extract the non-padding part
            degree = degree[:, padding[0]:-padding[0], padding[1]:-padding[1]]

            KQI.W += degree.sum()

        else:
            KQI.W += np.prod(x_new.shape) * np.prod(kernel_size)

        return x_new

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        C, H, W = volume.shape
        kernel_size = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        dilation = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)
        stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        padding = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)

        if padding[0] or padding[1]:
            if volume_backward is None:

                degree_back = torch.zeros((self.input_size[0],self.input_size[1]+2*padding[0],self.input_size[2]+2*padding[1]))
                for c, i, j in itertools.product(range(volume.size(0)),
                                                 range(0, kernel_size[0] * dilation[0], dilation[0]),
                                                 range(0, kernel_size[1] * dilation[1], dilation[1])):
                    degree_back[c, i:H * stride[0] + i:stride[0], j:W * stride[1] + j:stride[1]] += 1
                # Extract the real part of degree
                degree_back[:,0:padding[0],:] = 0
                degree_back[:, -padding[0]:-1, :] = 0
                degree_back[:, :, 0:padding[0]] = 0
                degree_back[:, :, -padding[0]:-1] = 0
                degree = torch.zeros(C,H,W)
                for c, i, j in itertools.product(range(volume.size(0)),
                                                 range(0, kernel_size[0] * dilation[0], dilation[0]),
                                                 range(0, kernel_size[1] * dilation[1], dilation[1])):
                    degree[:, :, :] += degree_back[c, i:H * stride[0] + i:stride[0], j:W * stride[1] + j:stride[1]]

                #二维即可
                virtual_volume_backward = torch.zeros(self.input_size[0],self.input_size[1]+2*padding[0],self.input_size[2]+2*padding[1])
                for c, i, j in itertools.product(range(volume.size(0)),
                                                 range(0, kernel_size[0] * dilation[0], dilation[0]),
                                                 range(0, kernel_size[1] * dilation[1], dilation[1])):
                    virtual_volume_backward[c, i:H * stride[0] + i:stride[0],
                    j:W * stride[1] + j:stride[1]] += 1 + (volume / degree / volume.size(0)).sum(dim=0)

                # Extract the non-padding part of volume_backward
                volume_backward = virtual_volume_backward[:, padding[0]:-padding[0], padding[1]:-padding[1]]

            temp_volume_backward = torch.zeros(self.input_size[0], self.input_size[1] + 2 * padding[0],
                                               self.input_size[2] + 2 * padding[1])
            for c, i, j in itertools.product(range(volume.size(0)),
                                             range(0, kernel_size[0] * dilation[0], dilation[0]),
                                             range(0, kernel_size[1] * dilation[1], dilation[1])):
               temp_volume_backward[c, i:H * stride[0] + i:stride[0], j:W * stride[1] + j:stride[1]] = volume[c] / degree[c]
                # =? +=? 如有多个孩子节点呢？
                
            temp_volume_backward[:, padding[0]:-padding[0], padding[1]:-padding[1]] = volume_backward

            for c, i, j in itertools.product(range(volume.size(0)),
                                             range(0, kernel_size[0] * dilation[0], dilation[0]),
                                             range(0, kernel_size[1] * dilation[1], dilation[1])):
               KQI.kqi += self.KQI_formula((volume[0] / np.prod(kernel_size)),
                                            temp_volume_backward[c, i:H * stride[0] + i:stride[0], j:W * stride[1] + j:stride[1]])



        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                for c, i, j in itertools.product(range(volume.size(0)),
                                                 range(0, kernel_size[0] * dilation[0], dilation[0]),
                                                 range(0, kernel_size[1] * dilation[1], dilation[1])):
                    volume_backward[c, i:H * stride[0] + i:stride[0], j:W * stride[1] + j:stride[1]] += 1 + (
                            volume / np.prod(kernel_size) / volume.size(0)).sum(dim=0)

            for c, i, j in itertools.product(range(volume.size(0)),
                                             range(0, kernel_size[0] * dilation[0], dilation[0]),
                                             range(0, kernel_size[1] * dilation[1], dilation[1])):
                KQI.kqi += self.KQI_formula((volume[0] / np.prod(kernel_size)),
                                            volume_backward[c, i:H * stride[0] + i:stride[0],
                                            j:W * stride[1] + j:stride[1]])

        logging.debug(f'MaxPool2d: KQI={KQI.kqi}, node={np.product(volume.shape)}, volume={volume.sum()}')
        return volume_backward
