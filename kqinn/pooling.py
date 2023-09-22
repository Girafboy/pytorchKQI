import torch
import numpy as np
import logging
import itertools
import math

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
            _, H_new, W_new = x_new.shape
            _, H, W = x.shape

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
        _, H, W = volume.shape
        _, H_backward, W_backward = self.input_size
        kernel_size = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        dilation = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)
        stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        padding = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)

        if padding[0] or padding[1]:
            if volume_backward is None:

                degree = torch.zeros(volume.shape)
                for c, i, j in itertools.product(range(volume.size(0)),
                                                 range(0, kernel_size[0] * dilation[0], dilation[0]),
                                                 range(0, kernel_size[1] * dilation[1], dilation[1])):
                    degree[c, max(0, math.ceil((padding[0] - i) / stride[0])):min(H, (
                                H_backward - 1 - i + padding[0]) // stride[0] + 1),max(0, math.ceil((padding[1] - j) / stride[1])):min(W, (
                                W_backward - 1 - j + padding[1]) // stride[1] + 1)] += 1

                virtual_volume_backward = torch.zeros(self.input_size[0],self.input_size[1]+2*padding[0],self.input_size[2]+2*padding[1])
                for c, i, j in itertools.product(range(volume.size(0)),
                                                 range(0, kernel_size[0] * dilation[0], dilation[0]),
                                                 range(0, kernel_size[1] * dilation[1], dilation[1])):
                    virtual_volume_backward[c, i:H * stride[0] + i:stride[0], j:W * stride[1] + j:stride[1]] += 1 + (volume / degree / volume.size(0)).sum(dim=0)

                # Extract the non-padding part of volume_backward
                volume_backward = virtual_volume_backward[:, padding[0]:-padding[0], padding[1]:-padding[1]]

            for c, i, j in itertools.product(range(volume.size(0)),
                                             range(0, kernel_size[0] * dilation[0], dilation[0]),
                                             range(0, kernel_size[1] * dilation[1], dilation[1])):
                virtual_volume_backward = torch.zeros(self.input_size[0], self.input_size[1] + 2 * padding[0], self.input_size[2] + 2 * padding[1])

                virtual_volume_backward[c, i:H * stride[0] + i:stride[0], j:W * stride[1] + j:stride[1]] += volume[0] / degree[0] / volume.size(0)

                virtual_volume_backward[c, i:H * stride[0] + i:stride[0], j:W * stride[1] + j:stride[1]][max(0, padding[0] - i):min(H, H - i + padding[0]),
                max(0, padding[1] - j):min(W, W - j + padding[1])] = volume_backward[c, max(0, i - padding[0]):min(H,H + i -padding[0]), max(0, j - padding[1]):min(W,W + j -padding[1])]

                KQI.kqi += self.KQI_formula((volume[0] / np.prod(kernel_size)), virtual_volume_backward[c, i:H * stride[0] + i:stride[0], j:W * stride[1] + j:stride[1]])


        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                for c, i, j in itertools.product(range(volume.size(0)),
                                                 range(0, kernel_size[0] * dilation[0], dilation[0]),
                                                 range(0, kernel_size[1] * dilation[1], dilation[1])):
                    volume_backward[c, i:H * stride[0] + i:stride[0], j:W * stride[1] + j:stride[1]] += 1 + (volume / np.prod(kernel_size) / volume.size(0)).sum(dim=0)

            for c, i, j in itertools.product(range(volume.size(0)),
                                             range(0, kernel_size[0] * dilation[0], dilation[0]),
                                             range(0, kernel_size[1] * dilation[1], dilation[1])):
                KQI.kqi += self.KQI_formula((volume[0] / np.prod(kernel_size)), volume_backward[c, i:H * stride[0] + i:stride[0], j:W * stride[1] + j:stride[1]])

        logging.debug(f'MaxPool2d: KQI={KQI.kqi}, node={np.product(volume.shape)}, volume={volume.sum()}')
        return volume_backward
