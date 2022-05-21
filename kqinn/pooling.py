import torch
import numpy as np
import logging
import itertools
import math

from .kqi import KQI


class MaxPool2d(torch.nn.MaxPool2d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.kernel_size = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        self.dilation = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)

        self.input_size = x.shape
        x_new = self.forward(x)
        assert x_new.shape[-3] == x.size(0)

        if self.padding[0] or self.padding[1]:
            degree = self._degree(x.shape, x_new.shape)
            KQI.W += degree.sum() * x.size(0)
        else:
            KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size)

        return x_new

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        _, H, W = volume.shape

        if self.padding[0] or self.padding[1]:
            volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1] + 2 * self.padding[0], self.input_size[2] + 2 * self.padding[1]))
            if volume_backward is None:
                degree = self._degree(self.input_size, volume.shape)
                for c, i, j in itertools.product(range(volume.size(0)),
                                                 range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]),
                                                 range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1])):
                    volume_back_padding[c, i:H * self.stride[0] + i:self.stride[0], j:W * self.stride[1] + j:self.stride[1]] += 1 + volume[c] / degree
                volume_backward = volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]].clone()

            for c, i, j in itertools.product(range(volume.size(0)),
                                                     range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]),
                                                     range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1])):
                tmp = volume_back_padding.clone()
                H_index = torch.LongTensor([k for k in range(i, H * self.stride[0] + i, self.stride[0])])
                W_index = torch.LongTensor([k for k in range(j, W * self.stride[1] + j, self.stride[1])])
                x, y = torch.meshgrid(H_index, W_index)
                tmp[c, x, y] = volume[c] / degree
                H_index = torch.LongTensor([k for k in range(i, H * self.stride[0] + i, self.stride[0]) if k >= self.padding[0] and k < volume_back_padding.shape[1] - self.padding[0]])
                W_index = torch.LongTensor([k for k in range(j, W * self.stride[1] + j, self.stride[1]) if k >= self.padding[1] and k < volume_back_padding.shape[2] - self.padding[1]])
                x_padding, y_padding = torch.meshgrid(H_index, W_index)
                tmp[c, x_padding, y_padding] = volume_back_padding[c, x_padding, y_padding]
                KQI.kqi += self.KQI_formula(volume[c] / degree, tmp[c, x, y])

        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                for c, i, j in itertools.product(range(volume.size(0)),
                                                 range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]),
                                                 range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1])):
                    volume_backward[c, i:H * self.stride[0] + i:self.stride[0], j:W * self.stride[1] + j:self.stride[1]] += 1 + volume[c] / np.prod(self.kernel_size) / volume.size(0)

            for c, i, j in itertools.product(range(volume.size(0)),
                                             range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]),
                                             range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1])):
                KQI.kqi += self.KQI_formula(volume[0] / np.prod(self.kernel_size), volume_backward[c, i:H * self.stride[0] + i:self.stride[0], j:W * self.stride[1] + j:self.stride[1]])

        logging.debug(f'MaxPool2d: KQI={KQI.kqi}, node={np.product(volume.shape)}, volume={volume.sum()}')

        return volume_backward

    def _degree(self, input_size, output_size):
        _, HI, WI = input_size
        _, HO, WO = output_size

        degree = torch.zeros(HO, WO)
        for i,j in itertools.product(range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1])):
            Hleft = max(0, math.ceil((self.padding[0] - i) / self.stride[0]))
            Hright = min(HO, math.ceil((HI - i + self.padding[0]) / self.stride[0]))
            Wleft = max(0, math.ceil((self.padding[1] - j) / self.stride[1]))
            Wright = min(WO, math.ceil((WI - j + self.padding[1]) / self.stride[1]))
            degree[Hleft:Hright, Wleft:Wright] += 1

        return degree