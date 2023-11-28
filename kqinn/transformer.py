import logging

import numpy as np
import torch

from .kqi import KQI


class TransformerEncoderLayer(torch.nn.TransformerEncoderLayer, KQI):
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
        volume_9_1 = volume + 1
        volume_12 = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_9_1)
        KQI.kqi += self.KQI_formula(volume, volume_12)

        volume_12 = volume_12.sum(0).sum(0) / (seq_len * batch_size)
        # Linear2
        volume_11 = torch.ones(dim_feedforward) * (np.prod(d_model) + (volume_12 / dim_feedforward).sum())
        for vol in volume_11:
            KQI.kqi += self.KQI_formula(volume_12 / dim_feedforward, vol) * seq_len * batch_size

        # Linear1
        volume_10 = torch.ones(d_model) * (np.prod(dim_feedforward) + (volume_11 / d_model).sum())
        for vol in volume_10:
            KQI.kqi += self.KQI_formula(volume_11 / d_model, vol) * seq_len * batch_size

        # Norm2
        volume_9 = torch.ones(d_model) * (np.prod(d_model) + (volume_10 / d_model).sum())
        for vol in volume_9:
            KQI.kqi += self.KQI_formula(volume_10 / d_model, vol) * seq_len * batch_size
        volume_9 = volume_9.expand(seq_len, batch_size, d_model)
        volume_9 += volume_9_1

        # Add
        volume_0_1 = volume_9 + 1
        volume_8 = volume_9 + 1
        KQI.kqi += self.KQI_formula(volume_9, volume_0_1)
        KQI.kqi += self.KQI_formula(volume_9, volume_8)

        # Reshape size from (seq_len, batch_size, embed_dim) to (seq_len, embed_dim)
        volume_8 = volume_8.sum(1) / batch_size
        # ------------------------- MultiheadAttention -------------------------
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads

        # linear
        volume_7_all = torch.ones(volume_8.shape) * (
                np.prod(volume_8.shape) + (volume_8 / np.prod(volume_8.shape)).sum())
        for vol in volume_7_all.flatten():
            KQI.kqi += self.KQI_formula(volume_8 / np.prod(volume_8.shape), vol) * batch_size
        volume_7 = volume_7_all.reshape((seq_len, num_heads, head_dim)).sum(1) / num_heads
        # MatMul
        volume_6 = torch.ones((seq_len, seq_len)) * (head_dim + volume_7.sum() / (seq_len * 2) / seq_len)
        volume_2_v = torch.ones((seq_len, head_dim)) * (seq_len + volume_7.sum() / (seq_len * 2) / head_dim)
        for col in volume_6:
            for vol in col:
                KQI.kqi += self.KQI_formula(volume_7[0, :] / (seq_len * 2), vol) * num_heads * batch_size
        for col in volume_2_v:
            for vol in col:
                KQI.kqi += self.KQI_formula(volume_7[:, 0] / (seq_len * 2), vol) * num_heads * batch_size
        # Softmax
        volume_4 = torch.ones((seq_len, seq_len)) * (seq_len * seq_len + volume_6.sum() / (seq_len * seq_len))
        KQI.kqi += self.KQI_formula(volume_6 / np.prod(volume_6.shape), volume_4) * np.prod(
            volume_6.shape) * num_heads * batch_size
        # Mask
        # volume_4 = volume_5 + 1
        # KQI.kqi += self.KQI_formula(volume_5, volume_4) * num_heads
        # Scale
        volume_3 = volume_4 + 1
        KQI.kqi += self.KQI_formula(volume_4, volume_3) * num_heads * batch_size
        # MatMul
        volume_2_q = torch.ones((seq_len, head_dim)) * (seq_len + volume_3.sum() / (head_dim * 2) / seq_len)
        volume_2_k = torch.ones((seq_len, head_dim)) * (seq_len + volume_3.sum() / (head_dim * 2) / seq_len)
        for col in volume_2_q:
            for vol in col:
                KQI.kqi += self.KQI_formula(volume_3[0, :] / (head_dim * 2), vol) * num_heads * batch_size
        for col in volume_2_k:
            for vol in col:
                KQI.kqi += self.KQI_formula(volume_3[0, :] / (head_dim * 2), vol) * num_heads * batch_size
        # Linear
        volume_1_q = torch.ones((seq_len, head_dim)) * (seq_len * head_dim + volume_2_q.sum() / (seq_len * head_dim))
        volume_1_k = torch.ones((seq_len, head_dim)) * (seq_len * head_dim + volume_2_k.sum() / (seq_len * head_dim))
        volume_1_v = torch.ones((seq_len, head_dim)) * (seq_len * head_dim + volume_2_v.sum() / (seq_len * head_dim))
        KQI.kqi += self.KQI_formula(volume_2_q / np.prod(volume_2_q.shape), volume_1_q) * np.prod(
            volume_2_q.shape) * num_heads * batch_size
        KQI.kqi += self.KQI_formula(volume_2_k / np.prod(volume_2_k.shape), volume_1_k) * np.prod(
            volume_2_k.shape) * num_heads * batch_size
        KQI.kqi += self.KQI_formula(volume_2_v / np.prod(volume_2_v.shape), volume_1_v) * np.prod(
            volume_2_v.shape) * num_heads * batch_size
        volume_backward_q = volume_1_q.repeat(1, num_heads)
        volume_backward_k = volume_1_k.repeat(1, num_heads)
        volume_backward_v = volume_1_v.repeat(1, num_heads)
        volume_1 = volume_backward_q + volume_backward_k + volume_backward_v
        # ------------------------- MultiheadAttention -------------------------
        # Reshape size from (seq_len, embed_dim) to (seq_len, batch_size, embed_dim)
        volume_1 = volume_1.unsqueeze(1).expand(seq_len, batch_size, d_model)

        volume_1 = volume_1.sum(0).sum(0) / (seq_len * batch_size)
        # Norm1
        volume_0 = torch.ones(d_model) * (np.prod(d_model) + (volume_1 / d_model).sum())
        for vol in volume_0:
            KQI.kqi += self.KQI_formula(volume_1 / d_model, vol) * seq_len * batch_size
        volume_0 = volume_0.expand(seq_len, batch_size, d_model)

        # Add
        volume_0 += volume_0_1

        logging.debug(f'LayerNorm: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_0
