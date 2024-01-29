import torch
import numpy as np
import logging
import itertools
import math

from .kqi import KQI


class AvgPool1d(torch.nn.AvgPool1d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_size = x.shape
        x_new = self.forward(x)

        degree = self._degree(x.shape, x_new.shape)
        KQI.W += degree.sum() * x.size(0)
        return x_new

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        _, H = volume.shape
        if not self.ceil_mode:
            indexing = lambda i: [slice(None), slice(i, H * self.stride[0] + i, self.stride[0])]
            index = lambda i: [slice(None)]
        else:
            indexing = lambda i: [slice(None), slice(i, H * self.stride[0] + i if (H-1) * self.stride[0] + i < self.input_size[1] + 2 * self.padding[0] else (H-1) * self.stride[0] + i, self.stride[0])]
            index = lambda i: [slice(0, H if (H-1) * self.stride[0] + i < self.input_size[1] + 2 * self.padding[0] else H-1, 1)]

        if self.padding[0]:
            volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1] + 2 * self.padding[0]))
            degree = self._degree(self.input_size, volume.shape)
            if volume_backward is None:
                for i in range(0, self.kernel_size[0], 1):
                    volume_back_padding[indexing(i)] += 1 + volume[[slice(None)] + index(i)] / degree[index(i)]
                volume_backward = volume_back_padding[:, self.padding[0]:-self.padding[0]].clone()

            volume_back_padding[:, self.padding[0]:-self.padding[0]] = volume_backward
            tmp = volume_back_padding.clone()
            for i in range(0, self.kernel_size[0], 1):
                i_ = next(k for k in range(i, volume_back_padding.shape[1], self.stride[0]) if k >= self.padding[0])
                tmp[indexing(i)] = volume[[slice(None)]+index(i)] / degree[index(i)]
                tmp[:, i_:-self.padding[0]:self.stride[0]] = volume_back_padding[:, i_:-self.padding[0]:self.stride[0]]
                KQI.kqi += self.KQI_formula(volume[[slice(None)] + index(i)] / degree[index(i)], tmp[indexing(i)])
        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                for i in range(0, self.kernel_size[0], 1):
                    volume_backward[indexing(i)] += 1 + volume[[slice(None)] + index(i)] / np.prod(self.kernel_size[0])

            for i in range(0, self.kernel_size[0], 1):
                KQI.kqi += self.KQI_formula(volume[[slice(None)] + index(i)] / np.prod(self.kernel_size[0]), volume_backward[indexing(i)])

        logging.debug(f'AvgPool1d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')

        return volume_backward

    def _degree(self, input_size, output_size):
        _, HI = input_size
        _, HO = output_size

        degree = torch.zeros(HO)
        for i in range(0, self.kernel_size[0], 1):
            Hleft = max(0, math.ceil((self.padding[0] - i) / self.stride[0]))
            Hright = min(HO, math.ceil((HI - i + self.padding[0]) / self.stride[0]))

            degree[Hleft:Hright] += 1
        return degree


class AvgPool2d(torch.nn.AvgPool2d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.kernel_size = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)

        self.input_size = x.shape
        x_new = self.forward(x)

        if self.padding[0] or self.padding[1]:
            degree = self._degree(x.shape, x_new.shape)
            KQI.W += degree.sum() * x.size(0)
        else:
            KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size)

        return x_new

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        _, H, W = volume.shape
        indexing = lambda i, j: [slice(None), slice(i, H * self.stride[0] + i, self.stride[0]), slice(j, W * self.stride[1] + j, self.stride[1])]

        if self.padding[0] or self.padding[1]:
            volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1] + 2 * self.padding[0], self.input_size[2] + 2 * self.padding[1]))
            degree = self._degree(self.input_size, volume.shape)
            if volume_backward is None:
                for i, j in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1)):
                    volume_back_padding[indexing(i, j)] += 1 + volume / degree
                volume_backward = volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]].clone()

            volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]] = volume_backward
            tmp = volume_back_padding.clone()
            for i, j in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1)):
                i_, j_ = next(k for k in range(i, volume_back_padding.shape[1], self.stride[0]) if k >= self.padding[0]), next(k for k in range(j, volume_back_padding.shape[2], self.stride[1]) if k >= self.padding[1])
                tmp[indexing(i, j)] = volume / degree
                tmp[:, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1]] = volume_back_padding[:, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1]]
                KQI.kqi += self.KQI_formula(volume / degree, tmp[indexing(i, j)])
        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                for i, j in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1)):
                    volume_backward[indexing(i, j)] += 1 + volume / np.prod(self.kernel_size)

            for i, j in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1)):
                KQI.kqi += self.KQI_formula(volume / np.prod(self.kernel_size), volume_backward[indexing(i, j)])

        logging.debug(f'AvgPool2d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')

        return volume_backward

    def _degree(self, input_size, output_size):
        _, HI, WI = input_size
        _, HO, WO = output_size

        degree = torch.zeros(HO, WO)
        for i, j in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1)):
            Hleft = max(0, math.ceil((self.padding[0] - i) / self.stride[0]))
            Hright = min(HO, math.ceil((HI - i + self.padding[0]) / self.stride[0]))
            Wleft = max(0, math.ceil((self.padding[1] - j) / self.stride[1]))
            Wright = min(WO, math.ceil((WI - j + self.padding[1]) / self.stride[1]))
            degree[Hleft:Hright, Wleft:Wright] += 1
        return degree


class AvgPool3d(torch.nn.AvgPool3d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.kernel_size = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size, self.kernel_size)
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
        indexing = lambda i, j, k: [slice(None), slice(i, H * self.stride[0] + i, self.stride[0]), slice(j, W * self.stride[1] + j, self.stride[1]), slice(k, L * self.stride[2] + k, self.stride[2])]

        if self.padding[0] or self.padding[1] or self.padding[2]:
            volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1] + 2 * self.padding[0], self.input_size[2] + 2 * self.padding[1], self.input_size[3] + 2 * self.padding[2]))
            degree = self._degree(self.input_size, volume.shape)
            if volume_backward is None:
                for i, j, k in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1), range(0, self.kernel_size[2], 1)):
                    volume_back_padding[indexing(i, j, k)] += 1 + volume / degree
                volume_backward = volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1], self.padding[2]:-self.padding[2]].clone()

            volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1], self.padding[2]:-self.padding[2]] = volume_backward
            tmp = volume_back_padding.clone()
            for i, j, k in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1), range(0, self.kernel_size[2], 1)):
                i_, j_, k_ = next(m for m in range(i, volume_back_padding.shape[1], self.stride[0]) if m >= self.padding[0]), next(m for m in range(j, volume_back_padding.shape[2], self.stride[1]) if m >= self.padding[1]), next(m for m in range(k, volume_back_padding.shape[3], self.stride[2]) if m >= self.padding[2])
                tmp[indexing(i, j, k)] = volume / degree
                tmp[:, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1], k_:-self.padding[2]:self.stride[2]] = volume_back_padding[:, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1], k_:-self.padding[2]:self.stride[2]]
                KQI.kqi += self.KQI_formula(volume / degree, tmp[indexing(i, j, k)])
        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                for i, j, k in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1), range(0, self.kernel_size[2], 1)):
                    volume_backward[indexing(i, j, k)] += 1 + volume / np.prod(self.kernel_size)

            for i, j, k in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1), range(0, self.kernel_size[2], 1)):
                KQI.kqi += self.KQI_formula(volume / np.prod(self.kernel_size), volume_backward[indexing(i, j, k)])

        logging.debug(f'AvgPool3d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')

        return volume_backward

    def _degree(self, input_size, output_size):
        _, HI, WI, LI = input_size
        _, HO, WO, LO = output_size

        degree = torch.zeros(HO, WO, LO)
        for i, j, k in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1), range(0, self.kernel_size[2], 1)):
            Hleft = max(0, math.ceil((self.padding[0] - i) / self.stride[0]))
            Hright = min(HO, math.ceil((HI - i + self.padding[0]) / self.stride[0]))
            Wleft = max(0, math.ceil((self.padding[1] - j) / self.stride[1]))
            Wright = min(WO, math.ceil((WI - j + self.padding[1]) / self.stride[1]))
            Lleft = max(0, math.ceil((self.padding[2] - k) / self.stride[2]))
            Lright = min(LO, math.ceil((LI - k + self.padding[2]) / self.stride[2]))

            degree[Hleft:Hright, Wleft:Wright, Lleft:Lright] += 1
        return degree


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
        indexing = lambda i: [slice(None), slice(i, H * self.stride + i, self.stride)]

        if self.padding:
            volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1] + 2 * self.padding))
            degree = self._degree(self.input_size, volume.shape)
            if volume_backward is None:
                for i in range(0, self.kernel_size * self.dilation, self.dilation):
                    volume_back_padding[indexing(i)] += 1 + volume / degree
                volume_backward = volume_back_padding[:, self.padding:-self.padding].clone()

            volume_back_padding[:, self.padding:-self.padding] = volume_backward
            tmp = volume_back_padding.clone()
            for i in range(0, self.kernel_size * self.dilation, self.dilation):
                i_ = next(k for k in range(i, volume_back_padding.shape[1], self.stride) if k >= self.padding)
                tmp[indexing(i)] = volume / degree
                tmp[:, i_:-self.padding:self.stride] = volume_back_padding[:, i_:-self.padding:self.stride]
                KQI.kqi += self.KQI_formula(volume / degree, tmp[indexing(i)])
        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                for i in range(0, self.kernel_size * self.dilation, self.dilation):
                    volume_backward[indexing(i)] += 1 + volume / np.prod(self.kernel_size)

            for i in range(0, self.kernel_size * self.dilation, self.dilation):
                KQI.kqi += self.KQI_formula(volume / np.prod(self.kernel_size), volume_backward[indexing(i)])

        logging.debug(f'MaxPool1d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')

        return volume_backward

    def _degree(self, input_size, output_size):
        _, HI = input_size
        _, HO = output_size

        degree = torch.zeros(HO)
        for i in range(0, self.kernel_size * self.dilation, self.dilation):
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
        indexing = lambda i, j: [slice(None), slice(i, H * self.stride[0] + i, self.stride[0]), slice(j, W * self.stride[1] + j, self.stride[1])]

        if self.padding[0] or self.padding[1]:
            volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1] + 2 * self.padding[0], self.input_size[2] + 2 * self.padding[1]))
            degree = self._degree(self.input_size, volume.shape)
            if volume_backward is None:
                for i, j in itertools.product(range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1])):
                    volume_back_padding[indexing(i, j)] += 1 + volume / degree
                volume_backward = volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]].clone()

            volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]] = volume_backward
            tmp = volume_back_padding.clone()
            for i, j in itertools.product(range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1])):
                i_, j_ = next(k for k in range(i, volume_back_padding.shape[1], self.stride[0]) if k >= self.padding[0]), next(k for k in range(j, volume_back_padding.shape[2], self.stride[1]) if k >= self.padding[1])
                tmp[indexing(i, j)] = volume / degree
                tmp[:, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1]] = volume_back_padding[:, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1]]
                KQI.kqi += self.KQI_formula(volume / degree, tmp[indexing(i, j)])
        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                for i, j in itertools.product(range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1])):
                    volume_backward[indexing(i, j)] += 1 + volume / np.prod(self.kernel_size)

            for i, j in itertools.product(range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1])):
                KQI.kqi += self.KQI_formula(volume / np.prod(self.kernel_size), volume_backward[indexing(i, j)])

        logging.debug(f'MaxPool2d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')

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
        indexing = lambda i, j, k: [slice(None), slice(i, H * self.stride[0] + i, self.stride[0]), slice(j, W * self.stride[1] + j, self.stride[1]), slice(k, L * self.stride[2] + k, self.stride[2])]

        if self.padding[0] or self.padding[1] or self.padding[2]:
            volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1] + 2 * self.padding[0], self.input_size[2] + 2 * self.padding[1], self.input_size[3] + 2 * self.padding[2]))
            degree = self._degree(self.input_size, volume.shape)
            if volume_backward is None:
                for i, j, k in itertools.product(range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2] * self.dilation[2], self.dilation[2])):
                    volume_back_padding[indexing(i, j, k)] += 1 + volume / degree
                volume_backward = volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1], self.padding[2]:-self.padding[2]].clone()

            volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1], self.padding[2]:-self.padding[2]] = volume_backward
            tmp = volume_back_padding.clone()
            for i, j, k in itertools.product(range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2] * self.dilation[2], self.dilation[2])):
                i_, j_, k_ = next(m for m in range(i, volume_back_padding.shape[1], self.stride[0]) if m >= self.padding[0]), next(m for m in range(j, volume_back_padding.shape[2], self.stride[1]) if m >= self.padding[1]), next(m for m in range(k, volume_back_padding.shape[3], self.stride[2]) if m >= self.padding[2])
                tmp[indexing(i, j, k)] = volume / degree
                tmp[:, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1], k_:-self.padding[2]:self.stride[2]] = volume_back_padding[:, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1], k_:-self.padding[2]:self.stride[2]]
                KQI.kqi += self.KQI_formula(volume / degree, tmp[indexing(i, j, k)])
        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                for i, j, k in itertools.product(range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2] * self.dilation[2], self.dilation[2])):
                    volume_backward[indexing(i, j, k)] += 1 + volume / np.prod(self.kernel_size)

            for i, j, k in itertools.product(range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2] * self.dilation[2], self.dilation[2])):
                KQI.kqi += self.KQI_formula(volume / np.prod(self.kernel_size), volume_backward[indexing(i, j, k)])

        logging.debug(f'MaxPool3d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')

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


class AdaptiveAvgPool1d(torch.nn.AdaptiveAvgPool1d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_size = x.shape
        self.stride = None
        self.kernel_size = None
        self.padding = 0
        if self.output_size is None:
            self.output_size = self.input_size[1]
        self.stride = math.floor(self.input_size[1] / self.output_size)
        self.kernel_size = self.input_size[1] - (self.output_size - 1) * self.stride
        x_new = self.forward(x)

        KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size)

        return x_new

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        _, H = volume.shape
        indexing = lambda i: [slice(None), slice(i, H * self.stride + i, self.stride)]

        if volume_backward is None:
            volume_backward = torch.zeros(self.input_size)
            for i in range(0, self.kernel_size, 1):
                volume_backward[indexing(i)] += 1 + volume / np.prod(self.kernel_size)

        for i in range(0, self.kernel_size, 1):
            KQI.kqi += self.KQI_formula(volume / np.prod(self.kernel_size),
                                        volume_backward[indexing(i)])

        logging.debug(f'AdaptiveAvgPool1d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')

        return volume_backward


class AdaptiveAvgPool2d(torch.nn.AdaptiveAvgPool2d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_size = x.shape
        self.output_size = self.output_size if isinstance(self.output_size, tuple) else (self.output_size, self.output_size)
        self.stride = [None, None]
        self.kernel_size = [None, None]
        self.padding = [0, 0]
        for i in [0, 1]:
            if self.output_size[i] is None:
                self.output_size[i] = self.input_size[i+1]
            self.stride[i] = math.floor(self.input_size[i+1]/self.output_size[i])
            self.kernel_size[i] = self.input_size[i+1] - (self.output_size[i] - 1) * self.stride[i]
        x_new = self.forward(x)

        if self.padding[0] or self.padding[1]:
            degree = self._degree(x.shape, x_new.shape)
            KQI.W += degree.sum() * x.size(0)
        else:
            KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size)

        return x_new

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        _, H, W = volume.shape
        indexing = lambda i, j: [slice(None), slice(i, H * self.stride[0] + i, self.stride[0]), slice(j, W * self.stride[1] + j, self.stride[1])]

        if self.padding[0] or self.padding[1]:
            volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1] + 2 * self.padding[0], self.input_size[2] + 2 * self.padding[1]))
            degree = self._degree(self.input_size, volume.shape)
            if volume_backward is None:
                for i, j in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1)):
                    volume_back_padding[indexing(i, j)] += 1 + volume / degree
                volume_backward = volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]].clone()

            volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]] = volume_backward
            tmp = volume_back_padding.clone()
            for i, j in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1)):
                i_, j_ = next(k for k in range(i, volume_back_padding.shape[1], self.stride[0]) if k >= self.padding[0]), next(k for k in range(j, volume_back_padding.shape[2], self.stride[1]) if k >= self.padding[1])
                tmp[indexing(i, j)] = volume / degree
                tmp[:, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1]] = volume_back_padding[:, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1]]
                KQI.kqi += self.KQI_formula(volume / degree, tmp[indexing(i, j)])
        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                for i, j in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1)):
                    volume_backward[indexing(i, j)] += 1 + volume / np.prod(self.kernel_size)

            for i, j in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1)):
                KQI.kqi += self.KQI_formula(volume / np.prod(self.kernel_size), volume_backward[indexing(i, j)])

        logging.debug(f'AdaptiveAvgPool2d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')

        return volume_backward

    def _degree(self, input_size, output_size):
        _, HI, WI = input_size
        _, HO, WO = output_size

        degree = torch.zeros(HO, WO)
        for i, j in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1)):
            Hleft = max(0, math.ceil((self.padding[0] - i) / self.stride[0]))
            Hright = min(HO, math.ceil((HI - i + self.padding[0]) / self.stride[0]))
            Wleft = max(0, math.ceil((self.padding[1] - j) / self.stride[1]))
            Wright = min(WO, math.ceil((WI - j + self.padding[1]) / self.stride[1]))
            degree[Hleft:Hright, Wleft:Wright] += 1
        return degree


class AdaptiveAvgPool3d(torch.nn.AdaptiveAvgPool3d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_size = x.shape
        self.output_size = self.output_size if isinstance(self.output_size, tuple) else (self.output_size, self.output_size, self.output_size)
        self.stride = [None, None, None]
        self.kernel_size = [None, None, None]
        self.padding = [0, 0, 0]
        for i in [0, 1, 2]:
            if self.output_size[i] is None:
                self.output_size[i] = self.input_size[i + 1]
            self.stride[i] = math.floor(self.input_size[i + 1] / self.output_size[i])
            self.kernel_size[i] = self.input_size[i + 1] - (self.output_size[i] - 1) * self.stride[i]
        x_new = self.forward(x)

        if self.padding[0] or self.padding[1] or self.padding[2]:
            degree = self._degree(x.shape, x_new.shape)
            KQI.W += degree.sum() * x.size(0)
        else:
            KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size)

        return x_new

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        _, H, W, L = volume.shape
        indexing = lambda i, j, k: [slice(None), slice(i, H * self.stride[0] + i, self.stride[0]), slice(j, W * self.stride[1] + j, self.stride[1]), slice(k, L * self.stride[2] + k, self.stride[2])]

        if self.padding[0] or self.padding[1] or self.padding[2]:
            volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1] + 2 * self.padding[0], self.input_size[2] + 2 * self.padding[1], self.input_size[3] + 2 * self.padding[2]))
            degree = self._degree(self.input_size, volume.shape)
            if volume_backward is None:
                for i, j, k in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1), range(0, self.kernel_size[2], 1)):
                    volume_back_padding[indexing(i, j, k)] += 1 + volume / degree
                volume_backward = volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1], self.padding[2]:-self.padding[2]].clone()

            volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1], self.padding[2]:-self.padding[2]] = volume_backward
            tmp = volume_back_padding.clone()
            for i, j, k in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1), range(0, self.kernel_size[2], 1)):
                i_, j_, k_ = next(m for m in range(i, volume_back_padding.shape[1], self.stride[0]) if m >= self.padding[0]), next(m for m in range(j, volume_back_padding.shape[2], self.stride[1]) if m >= self.padding[1]), next(m for m in range(k, volume_back_padding.shape[3], self.stride[2]) if m >= self.padding[2])
                tmp[indexing(i, j, k)] = volume / degree
                tmp[:, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1], k_:-self.padding[2]:self.stride[2]] = volume_back_padding[:, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1], k_:-self.padding[2]:self.stride[2]]
                KQI.kqi += self.KQI_formula(volume / degree, tmp[indexing(i, j, k)])
        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                for i, j, k in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1), range(0, self.kernel_size[2], 1)):
                    volume_backward[indexing(i, j, k)] += 1 + volume / np.prod(self.kernel_size)

            for i, j, k in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1), range(0, self.kernel_size[2], 1)):
                KQI.kqi += self.KQI_formula(volume / np.prod(self.kernel_size), volume_backward[indexing(i, j, k)])

        logging.debug(f'AdaptiveAvgPool3d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')

        return volume_backward

    def _degree(self, input_size, output_size):
        _, HI, WI, LI = input_size
        _, HO, WO, LO = output_size

        degree = torch.zeros(HO, WO, LO)
        for i, j, k in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1), range(0, self.kernel_size[2], 1)):
            Hleft = max(0, math.ceil((self.padding[0] - i) / self.stride[0]))
            Hright = min(HO, math.ceil((HI - i + self.padding[0]) / self.stride[0]))
            Wleft = max(0, math.ceil((self.padding[1] - j) / self.stride[1]))
            Wright = min(WO, math.ceil((WI - j + self.padding[1]) / self.stride[1]))
            Lleft = max(0, math.ceil((self.padding[2] - k) / self.stride[2]))
            Lright = min(LO, math.ceil((LI - k + self.padding[2]) / self.stride[2]))

            degree[Hleft:Hright, Wleft:Wright, Lleft:Lright] += 1
        return degree


class AdaptiveMaxPool1d(torch.nn.AdaptiveMaxPool1d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_size = x.shape
        self.stride = None
        self.kernel_size = None
        self.padding = 0
        self.dilation = 1
        if self.output_size is None:
            self.output_size = self.input_size[1]
        self.stride = math.floor(self.input_size[1] / self.output_size)
        self.kernel_size = self.input_size[1] - (self.output_size - 1) * self.stride
        x_new = self.forward(x)

        KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size)

        return x_new

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        _, H = volume.shape
        indexing = lambda i: [slice(None), slice(i, H * self.stride + i, self.stride)]

        if volume_backward is None:
            volume_backward = torch.zeros(self.input_size)
            for i in range(0, self.kernel_size * self.dilation, self.dilation):
                volume_backward[indexing(i)] += 1 + volume / np.prod(self.kernel_size)

        for i in range(0, self.kernel_size * self.dilation, self.dilation):
            KQI.kqi += self.KQI_formula(volume / np.prod(self.kernel_size),
                                        volume_backward[indexing(i)])

        logging.debug(f'AdaptiveMaxPool1d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')

        return volume_backward


class AdaptiveMaxPool2d(torch.nn.AdaptiveMaxPool2d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_size = x.shape
        self.output_size = self.output_size if isinstance(self.output_size, tuple) else (self.output_size, self.output_size)
        self.stride = [None, None]
        self.kernel_size = [None, None]
        self.padding = [0, 0]
        self.dilation = [1, 1]
        for i in [0, 1]:
            if self.output_size[i] is None:
                self.output_size[i] = self.input_size[i + 1]
            self.stride[i] = math.floor(self.input_size[i + 1] / self.output_size[i])
            self.kernel_size[i] = self.input_size[i + 1] - (self.output_size[i] - 1) * self.stride[i]
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
        indexing = lambda i, j: [slice(None), slice(i, H * self.stride[0] + i, self.stride[0]), slice(j, W * self.stride[1] + j, self.stride[1])]

        if self.padding[0] or self.padding[1]:
            volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1] + 2 * self.padding[0], self.input_size[2] + 2 * self.padding[1]))
            degree = self._degree(self.input_size, volume.shape)
            if volume_backward is None:
                for i, j in itertools.product(range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1])):
                    volume_back_padding[indexing(i, j)] += 1 + volume / degree
                volume_backward = volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]].clone()

            volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]] = volume_backward
            tmp = volume_back_padding.clone()
            for i, j in itertools.product(range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1])):
                i_, j_ = next(k for k in range(i, volume_back_padding.shape[1], self.stride[0]) if k >= self.padding[0]), next(k for k in range(j, volume_back_padding.shape[2], self.stride[1]) if k >= self.padding[1])
                tmp[indexing(i, j)] = volume / degree
                tmp[:, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1]] = volume_back_padding[:, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1]]
                KQI.kqi += self.KQI_formula(volume / degree, tmp[indexing(i, j)])
        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                for i, j in itertools.product(range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1])):
                    volume_backward[indexing(i, j)] += 1 + volume / np.prod(self.kernel_size)

            for i, j in itertools.product(range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1])):
                KQI.kqi += self.KQI_formula(volume / np.prod(self.kernel_size), volume_backward[indexing(i, j)])

        logging.debug(f'AdaptiveMaxPool2d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')

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


class AdaptiveMaxPool3d(torch.nn.AdaptiveMaxPool3d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_size = x.shape
        self.output_size = self.output_size if isinstance(self.output_size, tuple) else (self.output_size, self.output_size, self.output_size)
        self.stride = [None, None, None]
        self.kernel_size = [None, None, None]
        self.padding = [0, 0, 0]
        self.dilation = [1, 1, 1]
        for i in [0, 1, 2]:
            if self.output_size[i] is None:
                self.output_size[i] = self.input_size[i + 1]
            self.stride[i] = math.floor(self.input_size[i + 1] / self.output_size[i])
            self.kernel_size[i] = self.input_size[i + 1] - (self.output_size[i] - 1) * self.stride[i]
        x_new = self.forward(x)

        if self.padding[0] or self.padding[1] or self.padding[2]:
            degree = self._degree(x.shape, x_new.shape)
            KQI.W += degree.sum() * x.size(0)
        else:
            KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size)

        return x_new

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        _, H, W, L = volume.shape
        indexing = lambda i, j, k: [slice(None), slice(i, H * self.stride[0] + i, self.stride[0]), slice(j, W * self.stride[1] + j, self.stride[1]), slice(k, L * self.stride[2] + k, self.stride[2])]

        if self.padding[0] or self.padding[1] or self.padding[2]:
            volume_back_padding = torch.zeros((self.input_size[0], self.input_size[1] + 2 * self.padding[0], self.input_size[2] + 2 * self.padding[1], self.input_size[3] + 2 * self.padding[2]))
            degree = self._degree(self.input_size, volume.shape)
            if volume_backward is None:
                for i, j, k in itertools.product(range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2] * self.dilation[2], self.dilation[2])):
                    volume_back_padding[indexing(i, j, k)] += 1 + volume / degree
                volume_backward = volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1], self.padding[2]:-self.padding[2]].clone()

            volume_back_padding[:, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1], self.padding[2]:-self.padding[2]] = volume_backward
            tmp = volume_back_padding.clone()
            for i, j, k in itertools.product(range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2] * self.dilation[2], self.dilation[2])):
                i_, j_, k_ = next(m for m in range(i, volume_back_padding.shape[1], self.stride[0]) if m >= self.padding[0]), next(m for m in range(j, volume_back_padding.shape[2], self.stride[1]) if m >= self.padding[1]), next(m for m in range(k, volume_back_padding.shape[3], self.stride[2]) if m >= self.padding[2])
                tmp[indexing(i, j, k)] = volume / degree
                tmp[:, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1], k_:-self.padding[2]:self.stride[2]] = volume_back_padding[:, i_:-self.padding[0]:self.stride[0], j_:-self.padding[1]:self.stride[1], k_:-self.padding[2]:self.stride[2]]
                KQI.kqi += self.KQI_formula(volume / degree, tmp[indexing(i, j, k)])
        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
                for i, j, k in itertools.product(range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2] * self.dilation[2], self.dilation[2])):
                    volume_backward[indexing(i, j, k)] += 1 + volume / np.prod(self.kernel_size)

            for i, j, k in itertools.product(range(0, self.kernel_size[0] * self.dilation[0], self.dilation[0]), range(0, self.kernel_size[1] * self.dilation[1], self.dilation[1]), range(0, self.kernel_size[2] * self.dilation[2], self.dilation[2])):
                KQI.kqi += self.KQI_formula(volume / np.prod(self.kernel_size), volume_backward[indexing(i, j, k)])

        logging.debug(f'AdaptiveMaxPool3d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')

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


class LPPool1d(torch.nn.LPPool1d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_size = x.shape
        x_new = self.forward(x)

        KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size)

        return x_new

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        _, H = volume.shape
        indexing = lambda i: [slice(None), slice(i, H * self.stride + i, self.stride)]

        if volume_backward is None:
            volume_backward = torch.zeros(self.input_size)
            for i in range(0, self.kernel_size, 1):
                volume_backward[indexing(i)] += 1 + volume / np.prod(self.kernel_size)

        for i in range(0, self.kernel_size, 1):
            KQI.kqi += self.KQI_formula(volume / np.prod(self.kernel_size), volume_backward[indexing(i)])

        logging.debug(f'LPPool1d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')

        return volume_backward


class LPPool2d(torch.nn.LPPool2d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.kernel_size = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)

        self.input_size = x.shape
        x_new = self.forward(x)

        KQI.W += np.prod(x_new.shape) * np.prod(self.kernel_size)

        return x_new

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        _, H, W = volume.shape
        indexing = lambda i, j: [slice(None), slice(i, H * self.stride[0] + i, self.stride[0]), slice(j, W * self.stride[1] + j, self.stride[1])]

        if volume_backward is None:
            volume_backward = torch.zeros(self.input_size)
            for i, j in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1)):
                volume_backward[indexing(i, j)] += 1 + volume / np.prod(self.kernel_size)

        for i, j in itertools.product(range(0, self.kernel_size[0], 1), range(0, self.kernel_size[1], 1)):
            KQI.kqi += self.KQI_formula(volume / np.prod(self.kernel_size), volume_backward[indexing(i, j)])

        logging.debug(f'LPPool2d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')

        return volume_backward
