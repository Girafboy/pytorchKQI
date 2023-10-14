import torch
import numpy as np
import logging

from torch import Tensor

from kqinn import KQI


class ReLU(torch.nn.ReLU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'ReLU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Tanh(torch.nn.Tanh, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Tanh: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Softmax(torch.nn.Softmax, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape) * x.shape[self.dim]
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = torch.mean(volume, self.dim, True).expand(volume.shape) + volume.shape[self.dim]

        KQI.kqi += self.KQI_formula(volume / volume.shape[self.dim], volume_backward) * volume.shape[self.dim]

        logging.debug(f'Softmax: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class MultiheadAttention(torch.nn.MultiheadAttention, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        # x 为 (3, seq_len, embed_dim)
        # 从 x 中分离出 k, q, v
        # k, q, v 的 shape 为 (seq_len, embed_dim)
        k, q, v = x.unbind(0)
        KQI.W += (np.prod(k.shape) + np.prod(q.shape) + np.prod(v.shape)) * np.prod(v.shape)
        res_for = self.forward(k, q, v)
        # res = torch.zeros_like(v)
        # for i in range(len(res_for)):
        #     res += res_for[i]
        # return res
        return res_for[0]

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward_ones = torch.ones((3,) + volume.shape)
            volume_backward = (volume_backward_ones *
                               (np.prod(volume_backward_ones.shape) + (volume / np.prod(volume.shape)).sum()))

        for vol in volume_backward:
            KQI.kqi += self.KQI_formula(volume / np.prod(volume.shape), vol)

        logging.debug(f'MultiHeadAttention: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward
