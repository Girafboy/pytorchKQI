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

            degree = torch.zeros((C_new,x.shape[1]+2*padding[0],x.shape[2]+2*padding[1]))
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
        # nn.Conv2d always stores the below parameters as tuples, while nn.MaxPool2d might store them as integers if they are initialized as int
        kernel_size = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        dilation = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)
        stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        padding = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)

        if padding[0] or padding[1]:
            if volume_backward is None:
                #H_in = (H + 1 - 1) * stride[0] + 1 - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1)
                #W_in = (W + 1 - 1) * stride[1] + 1 - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1)

                degree = torch.zeros((self.input_size[0],self.input_size[1]+2*padding[0],self.input_size[2]+2*padding[1]))
                for c, i, j in itertools.product(range(volume.size(0)),
                                                 range(0, kernel_size[0] * dilation[0], dilation[0]),
                                                 range(0, kernel_size[1] * dilation[1], dilation[1])):
                    degree[c, i:H * stride[0] + i:stride[0], j:W * stride[1] + j:stride[1]] += 1
                # Extract the real degree part
                start_H = (degree.shape[1] - volume.shape[1]) // 2
                end_H = start_H + volume.shape[1]
                start_W = (degree.shape[2] - volume.shape[2]) // 2
                end_W = start_W + volume.shape[2]
                degree = degree[:, start_H:end_H, start_W:end_W] # to be revised


                virtual_volume_backward = torch.zeros(self.input_size[0],self.input_size[1]+2*padding[0],self.input_size[2]+2*padding[1])
                for c, i, j in itertools.product(range(volume.size(0)),
                                                 range(0, kernel_size[0] * dilation[0], dilation[0]),
                                                 range(0, kernel_size[1] * dilation[1], dilation[1])):
                    # print(size_backward)
                    # print(volume_backward[c, i:H * stride[0] + i:stride[0], j:W * stride[1] + j:stride[1]].shape)
                    # print(volume_backward.shape)
                    # print(volume.shape)
                    # print(degree.shape)
                    # print((volume / degree / volume.size(0)).sum(dim=0).shape)
                    virtual_volume_backward[c, i:H * stride[0] + i:stride[0],
                    j:W * stride[1] + j:stride[1]] += 1 + (volume / degree / volume.size(0)).sum(dim=0)



            for c, i, j in itertools.product(range(volume.size(0)),
                                             range(0, kernel_size[0] * dilation[0], dilation[0]),
                                             range(0, kernel_size[1] * dilation[1], dilation[1])):

               #virtual_volume_backward[,,] = volume[] / degree[] # to be completed

                KQI.kqi += self.KQI_formula((volume[0] / np.prod(kernel_size)),
                                            virtual_volume_backward[c, i:H * stride[0] + i:stride[0], j:W * stride[1] + j:stride[1]])

            # Extract the non-padding part
            volume_backward = virtual_volume_backward[:, padding[0]:-padding[0], padding[1]:-padding[1]]


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
