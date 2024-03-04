import logging

import numpy as np
import torch

from .kqi import KQI
from .normalization import LayerNorm as LayerNormKQI
from .activation import MultiheadAttention as MultiheadAttentionKQI
from .linear import Linear as LinearKQI
from .dropout import Dropout as DropoutKQI

class TransformerEncoderLayer(torch.nn.TransformerEncoderLayer, KQI):
    """
    This module is modified from torch.nn.TransformerEncoderLayer.
    We only consider the case of norm_first=True.
    """
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, embed_dim = x.shape
        d_model = embed_dim
        dim_feedforward = self.linear1.out_features

        # Norm1
        KQI.W += d_model ** 2 * seq_len * batch_size

        # ------------------------- MultiheadAttention -------------------------
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        KQI.W += (seq_len * head_dim) ** 2 * num_heads * 3 * batch_size
        # MatMul
        KQI.W += (head_dim + head_dim) * (seq_len * seq_len) * num_heads * batch_size
        # Scale
        KQI.W += seq_len * seq_len * num_heads * batch_size
        # Mask
        # KQI.W += seq_len * seq_len * num_heads
        # Softmax
        KQI.W += (seq_len * seq_len) ** 2 * num_heads * batch_size
        # MatMul
        KQI.W += (seq_len + seq_len) * (seq_len * head_dim) * num_heads * batch_size
        # linear
        KQI.W += (seq_len * embed_dim) ** 2 * batch_size
        # ------------------------- MultiheadAttention -------------------------

        # Add
        KQI.W += d_model * 2 * seq_len * batch_size

        # Norm2
        KQI.W += d_model ** 2 * seq_len * batch_size

        # Linear1
        KQI.W += d_model * dim_feedforward * seq_len * batch_size

        # Linear2
        KQI.W += dim_feedforward * d_model * seq_len * batch_size

        # Add
        KQI.W += d_model * 2 * seq_len * batch_size

        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        seq_len, batch_size, embed_dim = volume.shape
        d_model = embed_dim
        dim_feedforward = self.linear1.out_features

        # Add
        volume_12 = volume / 2 + 1
        volume_9_1 = volume / 2 + 1

        volume_12 = volume_12.sum(0).sum(0) / (seq_len * batch_size)
        # Linear2
        volume_11 = torch.ones(dim_feedforward) * (np.prod(d_model) + (volume_12 / dim_feedforward).sum())
        # Linear1
        volume_10 = torch.ones(d_model) * (np.prod(dim_feedforward) + (volume_11 / d_model).sum())
        # Norm2
        volume_9 = torch.ones(d_model) * (np.prod(d_model) + (volume_10 / d_model).sum())

        volume_9 = volume_9.expand(seq_len, batch_size, d_model)
        volume_12 = volume_12.expand(seq_len, batch_size, d_model)
        volume_9 += volume_9_1

        KQI.kqi += self.KQI_formula(volume / 2, volume_12)
        KQI.kqi += self.KQI_formula(volume / 2, volume_9)

        volume_12 = volume_12.sum(0).sum(0) / (seq_len * batch_size)
        for vol in volume_11:
            KQI.kqi += self.KQI_formula(volume_12 / dim_feedforward, vol) * seq_len * batch_size

        for vol in volume_10:
            KQI.kqi += self.KQI_formula(volume_11 / d_model, vol) * seq_len * batch_size

        volume_9 = volume_9.sum(0).sum(0) / (seq_len * batch_size)
        for vol in volume_9:
            KQI.kqi += self.KQI_formula(volume_10 / d_model, vol) * seq_len * batch_size
        volume_9 = volume_9.expand(seq_len, batch_size, d_model)

        # Add
        volume_8 = volume_9 / 2 + 1
        volume_0_1 = volume_9 / 2 + 1

        # Reshape size from (seq_len, batch_size, embed_dim) to (seq_len, embed_dim)
        volume_8 = volume_8.sum(1) / batch_size
        # ------------------------- MultiheadAttention -------------------------
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        # linear
        volume_7_all = torch.ones(volume_8.shape) * (
                np.prod(volume_8.shape) + (volume_8 / np.prod(volume_8.shape)).sum())
        volume_7 = volume_7_all.reshape((seq_len, num_heads, head_dim)).sum(1) / num_heads
        # MatMul
        volume_6 = torch.ones((seq_len, seq_len)) * (head_dim + volume_7.sum() / (seq_len * 2) / seq_len)
        volume_2_v = torch.ones((seq_len, head_dim)) * (seq_len + volume_7.sum() / (seq_len * 2) / head_dim)
        # Softmax
        volume_4 = torch.ones((seq_len, seq_len)) * (seq_len * seq_len + volume_6.sum() / (seq_len * seq_len))
        # Mask
        # volume_4 = volume_5 + 1
        # Scale
        volume_3 = volume_4 + 1
        # MatMul
        volume_2_q = torch.ones((seq_len, head_dim)) * (seq_len + volume_3.sum() / (head_dim * 2) / seq_len)
        volume_2_k = torch.ones((seq_len, head_dim)) * (seq_len + volume_3.sum() / (head_dim * 2) / seq_len)
        # Linear
        volume_1_q = torch.ones((seq_len, head_dim)) * (seq_len * head_dim + volume_2_q.sum() / (seq_len * head_dim))
        volume_1_k = torch.ones((seq_len, head_dim)) * (seq_len * head_dim + volume_2_k.sum() / (seq_len * head_dim))
        volume_1_v = torch.ones((seq_len, head_dim)) * (seq_len * head_dim + volume_2_v.sum() / (seq_len * head_dim))
        volume_backward_q = volume_1_q.repeat(1, num_heads)
        volume_backward_k = volume_1_k.repeat(1, num_heads)
        volume_backward_v = volume_1_v.repeat(1, num_heads)
        # ------------------------- MultiheadAttention -------------------------
        # Reshape size from (seq_len, embed_dim) to (seq_len, batch_size, embed_dim)
        volume_1 = volume_backward_q + volume_backward_k + volume_backward_v
        volume_1 = volume_1.unsqueeze(1).expand(seq_len, batch_size, d_model)
        volume_8 = volume_8.unsqueeze(1).expand(seq_len, batch_size, d_model)

        # Norm1
        volume_1 = volume_1.sum(0).sum(0) / (seq_len * batch_size)
        volume_0 = torch.ones(d_model) * (np.prod(d_model) + (volume_1 / d_model).sum())
        volume_1 = volume_1.expand(seq_len, batch_size, d_model)
        volume_0 = volume_0.expand(seq_len, batch_size, d_model)
        volume_0 += volume_0_1

        if volume_backward is None:
            volume_backward = volume_0
        KQI.kqi += self.KQI_formula(volume_9 / 2, volume_backward)
        KQI.kqi += self.KQI_formula(volume_9 / 2, volume_8)

        for vol in volume_7_all.flatten():
            KQI.kqi += self.KQI_formula(volume_8 / np.prod(volume_8.shape), vol) * batch_size

        for col in volume_6:
            for vol in col:
                KQI.kqi += self.KQI_formula(volume_7[0, :] / (seq_len * 2), vol) * num_heads * batch_size
        for col in volume_2_v:
            for vol in col:
                KQI.kqi += self.KQI_formula(volume_7[:, 0] / (seq_len * 2), vol) * num_heads * batch_size

        KQI.kqi += self.KQI_formula(volume_6 / np.prod(volume_6.shape), volume_4) * np.prod(
            volume_6.shape) * num_heads * batch_size

        # KQI.kqi += self.KQI_formula(volume_5, volume_4) * num_heads
        KQI.kqi += self.KQI_formula(volume_4, volume_3) * num_heads * batch_size

        for col in volume_2_q:
            for vol in col:
                KQI.kqi += self.KQI_formula(volume_3[0, :] / (head_dim * 2), vol) * num_heads * batch_size
        for col in volume_2_k:
            for vol in col:
                KQI.kqi += self.KQI_formula(volume_3[0, :] / (head_dim * 2), vol) * num_heads * batch_size

        volume_2_kqv = torch.cat([volume_2_q, volume_2_k, volume_2_v], dim=1)
        volume_1_kqv = volume_1_q + volume_1_k + volume_1_v
        for vol in volume_1_kqv.flatten():
            KQI.kqi += self.KQI_formula(volume_2_kqv / np.prod(volume_1_kqv.shape), vol) * num_heads * batch_size

        volume_backward = volume_backward.sum(0).sum(0) / (seq_len * batch_size)
        for vol in volume_backward:
            KQI.kqi += self.KQI_formula(volume_1 / d_model, vol) * seq_len * batch_size
        volume_backward = volume_backward.expand(seq_len, batch_size, d_model)

        logging.debug(
            f'TransformerEncoderLayer: KQI={KQI.kqi}, node={np.prod(volume_0.shape)}, volume={volume_0.sum()}')
        return volume_backward


class TransformerDecoderLayer(torch.nn.TransformerDecoderLayer, KQI):
    """
    This module is modified from torch.nn.TransformerDecoderLayer.
    We only consider the case of norm_first=True.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = 'relu', layer_norm_eps: float = 1e-5, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None):
        super().__init__(d_model, nhead, dim_feedforward, norm_first)

        self.self_attn = MultiheadAttentionKQI(embed_dim=d_model, num_heads=nhead)  # mask to be added
        self.multihead_attn = MultiheadAttentionKQI(embed_dim=d_model, num_heads=nhead)
        self.linear1 = LinearKQI(in_features=d_model, out_features=dim_feedforward, bias=False)
        self.linear2 = LinearKQI(in_features=d_model, out_features=dim_feedforward, bias=False)
        self.norm1 = LayerNormKQI(normalized_shape=d_model)
        self.norm2 = LayerNormKQI(normalized_shape=d_model)  # to be revised
        self.norm3 = LayerNormKQI(normalized_shape=d_model)  # to be revised


    def KQIforward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, embed_dim = x.size()
        if self.memory is None:
            raise ValueError("Encoder output not found.")
        if self.norm_first:
            tgt = x
            x = self.norm1.KQIforward(x)
            x = self.self_attn.KQIforward(x, x, x)
            # ADD to be implemented
            x = self.norm2.KQIforward(x)
            x = self.multihead_attn.KQIforward(x, memory, memory)
            # ADD to be implemented
            x = self.norm3(x)
            x = self.linear1.KQIforward(x)  # _ff_block
            x = self.linear2.KQIforward(x)  # _ff_block
            # ADD to be implemented
        else:
            raise ValueError("Non-norm_first not implemented.")
        return x


    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if self.memory is None:
            raise ValueError("Encoder output not found.")
        if self.norm_first:
            # ADD to be implemented
            volume = self.linear2.KQIbackward(volume.flatten())  # _ff_block
            volume = self.linear1.KQIbackward(volume.flatten())  # _ff_block
            volume = self.norm3.KQIbackward(volume)   # to be revised
            # ADD to be implemented
            volume_backward_k, volume_backward_q, volume_backward_v = self.multihead_attn.KQIbackward(volume)  # _mha_block
            volume = torch.cat([volume_backward_q, volume_backward_k, volume_backward_v], dim=1)  # _mha_block
            volume = self.norm2.KQIbackward(volume)   # to be revised
            # ADD to be implemented
            volume = self.self_attn.KQIbackward(volume)
            volume = self.norm1.KQIbackward(volume, volume_backward)
        else:
            raise ValueError("Non-norm_first not implemented.")
        return volume



