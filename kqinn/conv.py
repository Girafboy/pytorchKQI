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

        degree = self._degree(x.shape, x_new.shape)
        KQI.W += degree.sum() * self.out_channels * self.in_channels

        return x_new

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        _, H = volume.shape
        indexing = lambda i: [slice(None), slice(i, H * self.stride[0] + i, self.stride[0])]

        start = self.padding[0]
        end = None if self.padding[0] == 0 else -self.padding[0]

        volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1] + 2 * self.padding[0]))
        degree = self._degree(self.input_size, volume.shape)
        if volume_backward is None:
            for c, i in itertools.product(range(self.in_channels), range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0])):
                volume_back_padding[c, indexing(i)[1]] += self.out_channels + (volume / degree / self.in_channels).sum(dim=0)
            volume_backward = volume_back_padding[:, start:end].clone()

        volume_back_padding[:, start:end] = volume_backward
        tmp = volume_back_padding.clone()
        for cout, i in itertools.product(range(self.out_channels), range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0])):
            i_ = next(m for m in range(i, volume_back_padding.shape[1], self.stride[0]) if m >= self.padding[0])
            tmp[indexing(i)] = volume[cout] / degree / self.in_channels
            tmp[:, i_:end:self.stride[0]] = volume_back_padding[:, i_:end:self.stride[0]]
            KQI.kqi += self.KQI_formula((volume[cout] / degree / self.in_channels).expand(self.in_channels, -1), tmp[indexing(i)])

        logging.debug(f'Conv1d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward

    def _degree(self, input_size, output_size):
        _, HI = input_size
        _, HO = output_size

        degree = torch.zeros(HO)
        for i in range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]):
            Hleft = max(0, math.ceil((self.padding[0] - i) / self.stride[0]))
            Hright = min(HO, math.ceil((HI - i + self.padding[0]) / self.stride[0]))

            degree[Hleft:Hright] += 1

        return degree


class Conv2d(torch.nn.Conv2d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_size = x.shape
        x_new = self.forward(x)

        degree = self._degree(x.shape, x_new.shape)
        KQI.W += degree.sum() * self.out_channels * self.in_channels

        return x_new

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        _, H, W = volume.shape
        indexing = lambda i, j: [slice(None), slice(i, H * self.stride[0] + i, self.stride[0]), slice(j, W * self.stride[1] + j, self.stride[1])]

        start = self.padding
        end = [None if pad == 0 else -pad for pad in self.padding]

        volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1] + 2 * self.padding[0], self.input_size[2] + 2 * self.padding[1]))
        degree = self._degree(self.input_size, volume.shape)
        if volume_backward is None:
            for c, i, j in itertools.product(range(self.in_channels), range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1])):
                volume_back_padding[c, indexing(i, j)[1], indexing(i, j)[2]] += self.out_channels + (volume / degree / self.in_channels).sum(dim=0)
            volume_backward = volume_back_padding[:, start[0]:end[0], start[1]:end[1]].clone()

        volume_back_padding[:, start[0]:end[0], start[1]:end[1]] = volume_backward
        tmp = volume_back_padding.clone()
        for cout, i, j in itertools.product(range(self.out_channels), range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1])):
            i_, j_ = next(m for m in range(i, volume_back_padding.shape[1], self.stride[0]) if m >= self.padding[0]), next(m for m in range(j, volume_back_padding.shape[2], self.stride[1]) if m >= self.padding[1])
            tmp[indexing(i, j)] = volume[cout] / degree / self.in_channels
            tmp[:, i_:end[0]:self.stride[0], j_:end[1]:self.stride[1]] = volume_back_padding[:, i_:end[0]:self.stride[0], j_:end[1]:self.stride[1]]
            KQI.kqi += self.KQI_formula((volume[cout] / degree / self.in_channels).expand(self.in_channels, -1, -1), tmp[indexing(i, j)])

        logging.debug(f'Conv2d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward

    def _degree(self, input_size, output_size):
        _, HI, WI = input_size
        _, HO, WO = output_size

        degree = torch.zeros(HO, WO)
        for i, j in itertools.product(range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1])):
            Hleft = max(0, math.ceil((self.padding[0] - i) / self.stride[0]))
            Hright = min(HO, math.ceil((HI - i + self.padding[0]) / self.stride[0]))
            Wleft = max(0, math.ceil((self.padding[1] - j) / self.stride[1]))
            Wright = min(WO, math.ceil((WI - j + self.padding[1]) / self.stride[1]))
            degree[Hleft:Hright, Wleft:Wright] += 1

        return degree


class Conv3d(torch.nn.Conv3d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_size = x.shape
        x_new = self.forward(x)

        degree = self._degree(x.shape, x_new.shape)
        KQI.W += degree.sum() * self.out_channels * self.in_channels

        return x_new

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        _, H, W, L = volume.shape
        indexing = lambda i, j, k: [slice(None), slice(i, H * self.stride[0] + i, self.stride[0]), slice(j, W * self.stride[1] + j, self.stride[1]), slice(k, L * self.stride[2] + k, self.stride[2])]

        start = self.padding
        end = [None if pad == 0 else -pad for pad in self.padding]

        volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1] + 2 * self.padding[0], self.input_size[2] + 2 * self.padding[1], self.input_size[3] + 2 * self.padding[2]))
        degree = self._degree(self.input_size, volume.shape)
        if volume_backward is None:
            for c, i, j, k in itertools.product(range(self.in_channels), range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2] * self.dilation[2], self.dilation[2])):
                volume_back_padding[c, indexing(i, j, k)[1], indexing(i, j, k)[2], indexing(i, j, k)[3]] += self.out_channels + (volume / degree / self.in_channels).sum(dim=0)
            volume_backward = volume_back_padding[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]].clone()

        volume_back_padding[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]] = volume_backward
        tmp = volume_back_padding.clone()
        for cout, i, j, k in itertools.product(range(self.out_channels), range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2] * self.dilation[2], self.dilation[2])):
            i_, j_, k_ = next(m for m in range(i, volume_back_padding.shape[1], self.stride[0]) if m >= self.padding[0]), next(m for m in range(j, volume_back_padding.shape[2], self.stride[1]) if m >= self.padding[1]), next(m for m in range(k, volume_back_padding.shape[3], self.stride[2]) if m >= self.padding[2])
            tmp[indexing(i, j, k)] = volume[cout] / degree / self.in_channels
            tmp[:, i_:end[0]:self.stride[0], j_:end[1]:self.stride[1], k_:end[2]:self.stride[2]] = volume_back_padding[:, i_:end[0]:self.stride[0], j_:end[1]:self.stride[1], k_:end[2]:self.stride[2]]
            KQI.kqi += self.KQI_formula((volume[cout] / degree / self.in_channels).expand(self.in_channels, -1, -1, -1), tmp[indexing(i, j, k)])

        logging.debug(f'Conv3d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward

    def _degree(self, input_size, output_size):
        _, HI, WI, LI = input_size
        _, HO, WO, LO = output_size

        degree = torch.zeros(HO, WO, LO)
        for i, j, k in itertools.product(range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2] * self.dilation[2], self.dilation[2])):
            Hleft = max(0, math.ceil((self.padding[0] - i) / self.stride[0]))
            Hright = min(HO, math.ceil((HI - i + self.padding[0]) / self.stride[0]))
            Wleft = max(0, math.ceil((self.padding[1] - j) / self.stride[1]))
            Wright = min(WO, math.ceil((WI - j + self.padding[1]) / self.stride[1]))
            Lleft = max(0, math.ceil((self.padding[2] - k) / self.stride[2]))
            Lright = min(LO, math.ceil((LI - k + self.padding[2]) / self.stride[2]))

            degree[Hleft:Hright, Wleft:Wright, Lleft:Lright] += 1

        return degree
