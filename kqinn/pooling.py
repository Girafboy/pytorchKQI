import torch
import numpy as np
import logging
import itertools
import math

from .kqi import KQI


class MaxPool1d(torch.nn.MaxPool1d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_size = x.shape
        x_new = self.forward(x)

        if self.padding:
            degree = self._degree(x.shape, x_new.shape)
            KQI.W += degree.sum() * x.size(0)
        else:
            KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size)

        return x_new

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        _, H = volume.shape

        if self.padding:
            volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1] + 2 * self.padding))
            if volume_backward is None:
                degree = self._degree(self.input_size, volume.shape)
                for c, i in itertools.product(range(volume.size(0)), range(0, self.kernel_size * self.dilation, self.dilation)):
                    volume_back_padding[c, i:H * self.stride + i:self.stride] += 1 + volume[c] / degree
                volume_backward = volume_back_padding[:, self.padding:-self.padding].clone()

            for c, i in itertools.product(range(volume.size(0)), range(0, self.kernel_size * self.dilation, self.dilation)):
                i_ = next(k for k in range(i, volume_back_padding.shape[1], self.stride) if k >= self.padding)

                tmp = volume_back_padding.clone()
                tmp[c, i:H * self.stride + i:self.stride] = volume[c] / degree
                tmp[c, i_:-self.padding:self.stride] = volume_back_padding[c, i_:-self.padding:self.stride]

                KQI.kqi += self.KQI_formula(volume[c] / degree, tmp[c, i:H * self.stride + i:self.stride])

        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                for c, i in itertools.product(range(volume.size(0)), range(0, self.kernel_size * self.dilation, self.dilation)):
                    volume_backward[c, i:H * self.stride + i:self.stride] += 1 + volume[c] / np.prod(self.kernel_size) / volume.size(0)

            for c, i in itertools.product(range(volume.size(0)), range(0, self.kernel_size * self.dilation, self.dilation)):
                KQI.kqi += self.KQI_formula(volume[c] / np.prod(self.kernel_size), volume_backward[c, i:H * self.stride + i:self.stride])

        logging.debug(f'MaxPool1d: KQI={KQI.kqi}, node={np.product(volume.shape)}, volume={volume.sum()}')

        return volume_backward

    def _degree(self, input_size, output_size):
        _, HI = input_size
        _, HO = output_size

        degree = torch.zeros(HO)
        for i in range(0, self.kernel_size*self.dilation, self.dilation):
            Hleft = max(0, math.ceil((self.padding - i) / self.stride))
            Hright = min(HO, math.ceil((HI - i + self.padding) / self.stride))

            degree[Hleft:Hright] += 1
        return degree


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
                i_, j_ = next(k for k in range(i, volume_back_padding.shape[1], self.stride[0]) if k >= self.padding[0]), next(k for k in range(j, volume_back_padding.shape[2], self.stride[1]) if k >= self.padding[1])

                tmp = volume_back_padding.clone()
                tmp[c, i:H * self.stride[0] + i:self.stride[0], j:W * self.stride[1] + j:self.stride[1]] = volume[c] / degree
                tmp[c, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1]] = volume_back_padding[c, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1]]

                KQI.kqi += self.KQI_formula(volume[c] / degree, tmp[c, i:H * self.stride[0] + i:self.stride[0], j:W * self.stride[1] + j:self.stride[1]])

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
        for i, j in itertools.product(range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1])):
            Hleft = max(0, math.ceil((self.padding[0] - i) / self.stride[0]))
            Hright = min(HO, math.ceil((HI - i + self.padding[0]) / self.stride[0]))
            Wleft = max(0, math.ceil((self.padding[1] - j) / self.stride[1]))
            Wright = min(WO, math.ceil((WI - j + self.padding[1]) / self.stride[1]))
            degree[Hleft:Hright, Wleft:Wright] += 1
        return degree

class MaxPool3d(torch.nn.MaxPool3d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.kernel_size = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size, self.kernel_size)
        self.dilation = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation, self.dilation)
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride, self.stride)
        self.padding = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding, self.padding)

        self.input_size = x.shape
        x_new = self.forward(x)

        if self.padding[0] or self.padding[1] or self.padding[2]:
            degree = self._degree(x.shape, x_new.shape)
            KQI.W += degree.sum() * x.size(0)
        else:
            KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size)

        return x_new

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        _, H, W, L = volume.shape

        if self.padding[0] or self.padding[1] or self.padding[2]:
            volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1] + 2 * self.padding[0], self.input_size[2] + 2 * self.padding[1], self.input_size[3] + 2 * self.padding[2]))
            if volume_backward is None:
                degree = self._degree(self.input_size, volume.shape)
                for c, i, j, k in itertools.product(range(volume.size(0)), range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2] * self.dilation[2], self.dilation[2])):
                    volume_back_padding[c, i:H * self.stride[0] + i:self.stride[0], j:W * self.stride[1] + j:self.stride[1], k:L * self.stride[2] + k:self.stride[2]] += 1 + volume[c] / degree
                volume_backward = volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1], self.padding[2]:-self.padding[2]].clone()

            for c, i, j, k in itertools.product(range(volume.size(0)), range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2] * self.dilation[2], self.dilation[2])):
                i_, j_, k_ = next(m for m in range(i, volume_back_padding.shape[1], self.stride[0]) if m >= self.padding[0]), next(m for m in range(j, volume_back_padding.shape[2], self.stride[1]) if m >= self.padding[1]), next(m for m in range(k, volume_back_padding.shape[3], self.stride[2]) if m >= self.padding[2])

                tmp = volume_back_padding.clone()
                tmp[c, i:H * self.stride[0] + i:self.stride[0], j:W * self.stride[1] + j:self.stride[1], k:L * self.stride[2] + k:self.stride[2]] = volume[c] / degree
                tmp[c, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1], k_:-self.padding[2]:self.stride[2]] = volume_back_padding[c, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1], k_:-self.padding[2]:self.stride[2]]

                KQI.kqi += self.KQI_formula(volume[c] / degree, tmp[c, i:H * self.stride[0] + i:self.stride[0], j:W * self.stride[1] + j:self.stride[1], k:L * self.stride[2] + k:self.stride[2]])

        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                for c, i, j, k in itertools.product(range(volume.size(0)), range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2] * self.dilation[2], self.dilation[2])):
                    volume_backward[c, i:H * self.stride[0] + i:self.stride[0], j:W * self.stride[1] + j:self.stride[1], k:L * self.stride[2] + j:self.stride[2]] += 1 + volume[c] / np.prod(self.kernel_size) / volume.size(0)

            for c, i, j, k in itertools.product(range(volume.size(0)), range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2] * self.dilation[2], self.dilation[2])):
                KQI.kqi += self.KQI_formula(volume[0] / np.prod(self.kernel_size), volume_backward[c, i:H * self.stride[0] + i:self.stride[0], j:W * self.stride[1] + j:self.stride[1], k:L * self.stride[2] + j:self.stride[2]])

        logging.debug(f'MaxPool3d: KQI={KQI.kqi}, node={np.product(volume.shape)}, volume={volume.sum()}')

        return volume_backward

    def _degree(self, input_size, output_size):
        _, HI, WI, LI = input_size
        _, HO, WO, LO = output_size

        degree = torch.zeros(HO, WO, LO)
        for i, j, k in itertools.product(range(0, self.kernel_size[0]*self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1]*self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2]*self.dilation[2], self.dilation[2])):
            Hleft = max(0, math.ceil((self.padding[0] - i) / self.stride[0]))
            Hright = min(HO, math.ceil((HI - i + self.padding[0]) / self.stride[0]))
            Wleft = max(0, math.ceil((self.padding[1] - j) / self.stride[1]))
            Wright = min(WO, math.ceil((WI - j + self.padding[1]) / self.stride[1]))
            Lleft = max(0, math.ceil((self.padding[2] - k) / self.stride[2]))
            Lright = min(LO, math.ceil((LI - k + self.padding[2]) / self.stride[2]))

            degree[Hleft:Hright, Wleft:Wright, Lleft:Lright] += 1
        return degree
