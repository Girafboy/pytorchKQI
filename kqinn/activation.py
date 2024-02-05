import logging
from typing import Tuple, Optional

import numpy as np
import torch

from .kqi import KQI


class Threshold(torch.nn.Threshold, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Threshold: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


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


class Hardtanh(torch.nn.Hardtanh, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Hardtanh: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class ReLU6(torch.nn.ReLU6, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'ReLU6: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Sigmoid(torch.nn.Sigmoid, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Sigmoid: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
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


class Softmax2d(torch.nn.Softmax2d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape) * x.shape[-3]
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = torch.mean(volume, -3, True).expand(volume.shape) + volume.shape[-3]

        KQI.kqi += self.KQI_formula(volume / volume.shape[-3], volume_backward) * volume.shape[-3]

        logging.debug(f'Softmax2d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class LogSoftmax(torch.nn.LogSoftmax, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape) * x.shape[self.dim]
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = torch.mean(volume, self.dim, True).expand(volume.shape) + volume.shape[self.dim]

        KQI.kqi += self.KQI_formula(volume / volume.shape[self.dim], volume_backward) * volume.shape[self.dim]

        logging.debug(f'LogSoftmax: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class ELU(torch.nn.ELU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'ELU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class SELU(torch.nn.SELU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'SELU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class CELU(torch.nn.CELU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'CELU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class GELU(torch.nn.GELU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'GELU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Hardshrink(torch.nn.Hardshrink, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Hardshrink: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class LeakyReLU(torch.nn.LeakyReLU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'LeakyReLU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class LogSigmoid(torch.nn.LogSigmoid, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'LogSigmoid: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Softplus(torch.nn.Softplus, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Softplus: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Softshrink(torch.nn.Softshrink, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Softshrink: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class MultiheadAttention(torch.nn.MultiheadAttention, KQI):
    """
    This module is modified from torch.nn.MultiheadAttention.
    We only consider the case of Query embeddings of shape (seq_len, embed_dim) for unbatched input
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None, batch_first=False, device=None, dtype=None):
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first,
                         device, dtype)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def KQIforward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        # x: query, y: key, z: value
        seq_len, embed_dim = x.shape
        num_heads = self.num_heads
        head_dim = embed_dim // num_heads

        # linear
        KQI.W += (seq_len * head_dim) ** 2 * num_heads * 3

        # MatMul
        KQI.W += (head_dim + head_dim) * (seq_len * seq_len) * num_heads

        # Scale
        KQI.W += seq_len * seq_len * num_heads

        # Mask
        # KQI.W += seq_len * seq_len * num_heads

        # Softmax
        KQI.W += (seq_len * seq_len) ** 2 * num_heads

        # MatMul
        KQI.W += (seq_len + seq_len) * (seq_len * head_dim) * num_heads

        # linear
        KQI.W += (seq_len * embed_dim) ** 2

        return self.forward(x, y, z)

    def KQIbackward(self,
                    volume: torch.Tensor,
                    volume_weight: torch.Tensor = None,
                    volume_backward_k: torch.Tensor = None,
                    volume_backward_q: torch.Tensor = None,
                    volume_backward_v: torch.Tensor = None,
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len, embed_dim = volume.shape
        num_heads = self.num_heads
        head_dim = embed_dim // num_heads

        # linear
        volume_7_all = torch.ones(volume.shape) * (
                np.prod(volume.shape) + (volume / np.prod(volume.shape)).sum())
        for vol in volume_7_all.flatten():
            KQI.kqi += self.KQI_formula(volume / np.prod(volume.shape), vol)

        volume_7 = volume_7_all.reshape((seq_len, num_heads, head_dim)).sum(1) / num_heads

        # MatMul
        volume_6 = torch.ones((seq_len, seq_len)) * (head_dim + volume_7.sum() / (seq_len * 2) / seq_len)
        volume_2_v = torch.ones((seq_len, head_dim)) * (seq_len + volume_7.sum() / (seq_len * 2) / head_dim)
        for col in volume_6:
            for vol in col:
                KQI.kqi += self.KQI_formula(volume_7[0, :] / (seq_len * 2), vol) * num_heads
        for col in volume_2_v:
            for vol in col:
                KQI.kqi += self.KQI_formula(volume_7[:, 0] / (seq_len * 2), vol) * num_heads

        # Softmax
        volume_4 = torch.ones((seq_len, seq_len)) * (seq_len * seq_len + volume_6.sum() / (seq_len * seq_len))
        KQI.kqi += self.KQI_formula(volume_6 / np.prod(volume_6.shape), volume_4) * np.prod(volume_6.shape) * num_heads

        # Mask
        # volume_4 = volume_5 + 1
        # KQI.kqi += self.KQI_formula(volume_5, volume_4) * num_heads

        # Scale
        volume_3 = volume_4 + 1
        KQI.kqi += self.KQI_formula(volume_4, volume_3) * num_heads

        # MatMul
        volume_2_q = torch.ones((seq_len, head_dim)) * (seq_len + volume_3.sum() / (head_dim * 2) / seq_len)
        volume_2_k = torch.ones((seq_len, head_dim)) * (seq_len + volume_3.sum() / (head_dim * 2) / seq_len)
        for col in volume_2_q:
            for vol in col:
                KQI.kqi += self.KQI_formula(volume_3[0, :] / (head_dim * 2), vol) * num_heads
        for col in volume_2_k:
            for vol in col:
                KQI.kqi += self.KQI_formula(volume_3[0, :] / (head_dim * 2), vol) * num_heads

        # Linear
        volume_1_q = torch.ones((seq_len, head_dim)) * (seq_len * head_dim + volume_2_q.sum() / (seq_len * head_dim))
        volume_1_k = torch.ones((seq_len, head_dim)) * (seq_len * head_dim + volume_2_k.sum() / (seq_len * head_dim))
        volume_1_v = torch.ones((seq_len, head_dim)) * (seq_len * head_dim + volume_2_v.sum() / (seq_len * head_dim))

        KQI.kqi += self.KQI_formula(volume_2_q / np.prod(volume_2_q.shape), volume_1_q) * np.prod(
            volume_2_q.shape) * num_heads
        KQI.kqi += self.KQI_formula(volume_2_k / np.prod(volume_2_k.shape), volume_1_k) * np.prod(
            volume_2_k.shape) * num_heads
        KQI.kqi += self.KQI_formula(volume_2_v / np.prod(volume_2_v.shape), volume_1_v) * np.prod(
            volume_2_v.shape) * num_heads

        volume_backward_q = volume_1_q.repeat(1, num_heads)
        volume_backward_k = volume_1_k.repeat(1, num_heads)
        volume_backward_v = volume_1_v.repeat(1, num_heads)

        return volume_backward_k, volume_backward_q, volume_backward_v


class PReLU(torch.nn.PReLU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'PReLU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Softsign(torch.nn.Softshrink, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Softsign: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Softmin(torch.nn.Softmax, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape) * x.shape[self.dim]
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = torch.mean(volume, self.dim, True).expand(volume.shape) + volume.shape[self.dim]

        KQI.kqi += self.KQI_formula(volume / volume.shape[self.dim], volume_backward) * volume.shape[self.dim]

        logging.debug(f'Softmin: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Tanhshrink(torch.nn.Tanhshrink, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Tanhshrink: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class RReLU(torch.nn.RReLU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'RReLU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class GLU(torch.nn.GLU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward_half = volume / 2 + 1
            volume_backward = torch.cat((volume_backward_half, volume_backward_half), dim=self.dim)
        KQI.kqi += 2 * self.KQI_formula(volume / 2, volume_backward_half)
        logging.debug(f'GLU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Hardsigmoid(torch.nn.Hardsigmoid, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Hardsigmoid: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Hardswish(torch.nn.Hardswish, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Hardswish: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class SiLU(torch.nn.SiLU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'SiLU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Mish(torch.nn.Mish, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Mish: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward
