import logging

import numpy as np
import torch

from .kqi import KQI


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
    def KQIforward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, embed_dim = x.shape
        d_model = embed_dim
        dim_feedforward = self.linear1.out_features

        # Norm1
        KQI.W += d_model ** 2 * seq_len * batch_size

        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        # ------------------------- MaskedSelfAttention -------------------------
        KQI.W += (seq_len * head_dim) ** 2 * num_heads * 3 * batch_size
        # MatMul
        KQI.W += (head_dim + head_dim) * (seq_len * seq_len) * num_heads * batch_size
        # Scale
        KQI.W += seq_len * seq_len * num_heads * batch_size
        # Mask
        KQI.W += seq_len * seq_len * num_heads
        # Softmax
        KQI.W += (seq_len * seq_len) ** 2 * num_heads * batch_size
        # MatMul
        KQI.W += (seq_len + seq_len) * (seq_len * head_dim) * num_heads * batch_size
        # linear
        KQI.W += (seq_len * embed_dim) ** 2 * batch_size
        # ------------------------- MaskedSelfAttention -------------------------

        # Add
        KQI.W += d_model * 2 * seq_len * batch_size

        # Norm2
        KQI.W += d_model ** 2 * seq_len * batch_size

        # ------------------------- MultiheadAttention -------------------------
        KQI.W += (seq_len * head_dim) ** 2 * num_heads * 2 * batch_size  # For keys and values, which come from memory
        KQI.W += (seq_len * head_dim) ** 2 * num_heads * batch_size  # For queries
        # MatMul
        KQI.W += (head_dim + head_dim) * (seq_len * seq_len) * num_heads * batch_size
        # Scale
        KQI.W += seq_len * seq_len * num_heads * batch_size
        # Mask
        #KQI.W += seq_len * seq_len * num_heads
        # Softmax
        KQI.W += (seq_len * seq_len) ** 2 * num_heads * batch_size
        # MatMul
        KQI.W += (seq_len + seq_len) * (seq_len * head_dim) * num_heads * batch_size
        # linear
        KQI.W += (seq_len * embed_dim) ** 2 * batch_size
        # ------------------------- MultiheadAttention -------------------------

        # Add
        KQI.W += d_model * 2 * seq_len * batch_size

        # Norm3
        KQI.W += d_model ** 2 * seq_len * batch_size

        # Linear1
        KQI.W += d_model * dim_feedforward * seq_len * batch_size

        # Linear2
        KQI.W += dim_feedforward * d_model * seq_len * batch_size

        # Add
        KQI.W += d_model * 2 * seq_len * batch_size

        return self.forward(x, memory)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        seq_len, batch_size, embed_dim = volume.shape
        d_model = embed_dim
        dim_feedforward = self.linear1.out_features

        # Add
        volume_21 = volume / 2 + 1
        volume_18_1 = volume / 2 + 1

        volume_21 = volume_21.sum(0).sum(0) / (seq_len * batch_size)
        # Linear2
        volume_20 = torch.ones(dim_feedforward) * (np.prod(d_model) + (volume_21 / dim_feedforward).sum())
        # Linear1
        volume_19 = torch.ones(d_model) * (np.prod(dim_feedforward) + (volume_20 / d_model).sum())
        # Norm3
        volume_18 = torch.ones(d_model) * (np.prod(d_model) + (volume_19 / d_model).sum())

        volume_18 = volume_18.expand(seq_len, batch_size, d_model)
        volume_21 = volume_21.expand(seq_len, batch_size, d_model)
        volume_18 += volume_18_1

        KQI.kqi += self.KQI_formula(volume / 2, volume_21)
        KQI.kqi += self.KQI_formula(volume / 2, volume_18)
        logging.debug(f'L22: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')

        volume_21 = volume_21.sum(0).sum(0) / (seq_len * batch_size)
        for vol in volume_20:
            KQI.kqi += self.KQI_formula(volume_21 / dim_feedforward, vol) * seq_len * batch_size
        volume_21 = volume_21.expand(seq_len, batch_size, d_model)
        logging.debug(f'L21: KQI={KQI.kqi}, node={np.prod(volume_21.shape)}, volume={volume_21.sum()}')

        for vol in volume_19:
            KQI.kqi += self.KQI_formula(volume_20 / d_model, vol) * seq_len * batch_size
        volume_20 = volume_20.expand(seq_len, batch_size, dim_feedforward)
        logging.debug(f'L20: KQI={KQI.kqi}, node={np.prod(volume_20.shape)}, volume={volume_21.sum()}')

        volume_18 = volume_18.sum(0).sum(0) / (seq_len * batch_size)
        for vol in volume_18:
            KQI.kqi += self.KQI_formula(volume_19 / d_model, vol) * seq_len * batch_size
        volume_18 = volume_18.expand(seq_len, batch_size, d_model)
        volume_19 = volume_19.expand(seq_len, batch_size, d_model)
        logging.debug(f'L19: KQI={KQI.kqi}, node={np.prod(volume_19.shape)}, volume={volume_19.sum()}')

        # Add
        volume_17 = volume_18 / 2 + 1
        volume_9_1 = volume_18 / 2 + 1

        # Reshape size from (seq_len, batch_size, embed_dim) to (seq_len, embed_dim)
        volume_17 = volume_17.sum(1) / batch_size

        # ------------------------- MultiheadAttention -------------------------
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        # linear
        volume_16_all = torch.ones(volume_17.shape) * (
                np.prod(volume_17.shape) + (volume_17 / np.prod(volume_17.shape)).sum())
        volume_16 = volume_16_all.reshape((seq_len, num_heads, head_dim)).sum(1) / num_heads
        # MatMul
        volume_15 = torch.ones((seq_len, seq_len)) * (head_dim + volume_16.sum() / (seq_len * 2) / seq_len)
        volume_11_v = torch.ones((seq_len, head_dim)) * (seq_len + volume_16.sum() / (seq_len * 2) / head_dim)
        # Softmax
        volume_13 = torch.ones((seq_len, seq_len)) * (seq_len * seq_len + volume_16.sum() / (seq_len * seq_len))
        # Mask
        # volume_13 = volume_14 + 1
        # Scale
        volume_12 = volume_13 + 1
        # MatMul
        volume_11_q = torch.ones((seq_len, head_dim)) * (seq_len + volume_12.sum() / (head_dim * 2) / seq_len)
        volume_11_k = torch.ones((seq_len, head_dim)) * (seq_len + volume_12.sum() / (head_dim * 2) / seq_len)
        # Linear
        volume_10_q = torch.ones((seq_len, head_dim)) * (seq_len * head_dim + volume_11_q.sum() / (seq_len * head_dim))
        volume_10_k = torch.ones((seq_len, head_dim)) * (seq_len * head_dim + volume_11_k.sum() / (seq_len * head_dim))
        volume_10_v = torch.ones((seq_len, head_dim)) * (seq_len * head_dim + volume_11_v.sum() / (seq_len * head_dim))
        volume_backward_q = volume_10_q.repeat(1, num_heads)
        volume_backward_k = volume_10_k.repeat(1, num_heads)
        volume_backward_v = volume_10_v.repeat(1, num_heads)
        # ------------------------- MultiheadAttention -------------------------
        # Reshape size from (seq_len, embed_dim) to (seq_len, batch_size, embed_dim)
        volume_10 = volume_backward_q + volume_backward_k + volume_backward_v
        volume_10 = volume_10.unsqueeze(1).expand(seq_len, batch_size, d_model)
        volume_17 = volume_17.unsqueeze(1).expand(seq_len, batch_size, d_model)

        # Norm2
        volume_10 = volume_10.sum(0).sum(0) / (seq_len * batch_size)
        volume_9 = torch.ones(d_model) * (np.prod(d_model) + (volume_10 / d_model).sum())

        volume_10 = volume_10.expand(seq_len, batch_size, d_model)
        volume_9 = volume_9.expand(seq_len, batch_size, d_model) #？
        volume_9 += volume_9_1

        KQI.kqi += self.KQI_formula(volume_18 / 2, volume_9)
        KQI.kqi += self.KQI_formula(volume_18 / 2, volume_17)

        for vol in volume_16_all.flatten():
            KQI.kqi += self.KQI_formula(volume_17 / np.prod(volume_17.shape), vol) * batch_size

        for col in volume_15:
            for vol in col:
                KQI.kqi += self.KQI_formula(volume_16[0, :] / (seq_len * 2), vol) * num_heads * batch_size
        for col in volume_11_v:
            for vol in col:
                KQI.kqi += self.KQI_formula(volume_16[:, 0] / (seq_len * 2), vol) * num_heads * batch_size

        KQI.kqi += self.KQI_formula(volume_15 / np.prod(volume_15.shape), volume_13) * np.prod(
            volume_15.shape) * num_heads * batch_size

        # KQI.kqi += self.KQI_formula(volume_14, volume_13) * num_heads
        KQI.kqi += self.KQI_formula(volume_13, volume_12) * num_heads * batch_size

        for col in volume_11_q:
            for vol in col:
                KQI.kqi += self.KQI_formula(volume_12[0, :] / (head_dim * 2), vol) * num_heads * batch_size
        for col in volume_11_k:
            for vol in col:
                KQI.kqi += self.KQI_formula(volume_12[0, :] / (head_dim * 2), vol) * num_heads * batch_size

        volume_11_kqv = torch.cat([volume_11_q, volume_11_k, volume_11_v], dim=1)
        volume_10_kqv = volume_10_q + volume_10_k + volume_10_v
        for vol in volume_10_kqv.flatten():
            KQI.kqi += self.KQI_formula(volume_11_kqv / np.prod(volume_10_kqv.shape), vol) * num_heads * batch_size

        volume_9 = volume_9.sum(0).sum(0) / (seq_len * batch_size)
        for vol in volume_9:
            KQI.kqi += self.KQI_formula(volume_10 / d_model, vol) * seq_len * batch_size
        volume_9 = volume_9.expand(seq_len, batch_size, d_model)

        # Add
        volume_8 = volume_9 / 2 + 1
        volume_0_1 = volume_9 / 2 + 1

        # Reshape size from (seq_len, batch_size, embed_dim) to (seq_len, embed_dim)
        volume_8 = volume_8.sum(1) / batch_size
        # ------------------------- MaskedSelfAttention -------------------------
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
        volume_5 = torch.ones((seq_len, seq_len)) * (seq_len * seq_len + volume_6.sum() / (seq_len * seq_len))
        # Mask
        volume_4 = volume_5 + 1
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
        # ------------------------- MaskedSelfAttention -------------------------
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
        logging.debug(f'L9: KQI={KQI.kqi}, node={np.prod(volume_9.shape)}, volume={volume_9.sum()}')

        for vol in volume_7_all.flatten():
            KQI.kqi += self.KQI_formula(volume_8 / np.prod(volume_8.shape), vol) * batch_size

        logging.debug(f'L8: KQI={KQI.kqi}, node={np.prod(volume_8.shape)}, volume={volume_8.sum()}')

        for col in volume_6:
            for vol in col:
                KQI.kqi += self.KQI_formula(volume_7[0, :] / (seq_len * 2), vol) * num_heads * batch_size
        for col in volume_2_v:
            for vol in col:
                KQI.kqi += self.KQI_formula(volume_7[:, 0] / (seq_len * 2), vol) * num_heads * batch_size
        logging.debug(f'L7: KQI={KQI.kqi}, node={np.prod(volume_7.shape)}, volume={volume_7.sum()}')

        KQI.kqi += self.KQI_formula(volume_6 / np.prod(volume_6.shape), volume_4) * np.prod(
            volume_6.shape) * num_heads * batch_size
        logging.debug(f'L6: KQI={KQI.kqi}, node={np.prod(volume_6.shape)}, volume={volume_6.sum()}')

        KQI.kqi += self.KQI_formula(volume_5, volume_4) * num_heads
        logging.debug(f'L5: KQI={KQI.kqi}, node={np.prod(volume_5.shape)}, volume={volume_5.sum()}')

        KQI.kqi += self.KQI_formula(volume_4, volume_3) * num_heads * batch_size
        logging.debug(f'L4: KQI={KQI.kqi}, node={np.prod(volume_4.shape)}, volume={volume_4.sum()}')

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
            f'TransformerDecoderLayer: KQI={KQI.kqi}, node={np.prod(volume_0.shape)}, volume={volume_0.sum()}')
        return volume_backward
