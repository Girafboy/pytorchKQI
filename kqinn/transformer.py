import logging

import numpy as np
import torch
from torch import Tensor

import kqinn
from .kqi import KQI
from .normalization import LayerNorm as LayerNormKQI
from .activation import MultiheadAttention as MultiheadAttentionKQI
from .linear import Linear as LinearKQI
from .dropout import Dropout as DropoutKQI

class TransformerEncoder(torch.nn.TransformerEncoder, KQI):
    """
    This module is modified from torch.nn.TransformerEncoder.
    We only consider the case of norm_first=True.
    """

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__(encoder_layer, num_layers, norm, enable_nested_tensor, mask_check)
        self.encoder_layer = encoder_layer
        self.num_layers = num_layers

    def KQIforward(self, x: Tensor) -> Tensor:
        for i in range(self.num_layers):
            self.encoder_layer.KQIforward(x)

        return self.forward(x)

    def KQIbackward(self, volume: Tensor, volume_backward: Tensor = None) -> Tensor:
        volume = torch.zeros_like(volume)
        for i in range(self.num_layers):
            volume = self.encoder_layer.KQIbackward(volume)

        return volume


class TransformerEncoderLayer(torch.nn.TransformerEncoderLayer, KQI):
    """
    This module is modified from torch.nn.TransformerEncoderLayer.
    We only consider the case of norm_first=True.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 norm_first: bool = True):
        if not norm_first:
            raise NotImplementedError('norm_first=False is not supported now')

        super().__init__(d_model, nhead, dim_feedforward, dropout, norm_first=norm_first)
        self.norm_first = norm_first
        self.sa_block = sa_block(d_model, nhead, dropout)
        self.ff_block = ff_block(d_model, dim_feedforward, dropout)

    def KQIforward(self, x: Tensor) -> Tensor:
        x = kqinn.Branch(self.sa_block, kqinn.EmptyModule()).KQIforward(x)
        x = kqinn.Branch(self.ff_block, kqinn.EmptyModule()).KQIforward(x)

        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        volume = kqinn.Branch(self.ff_block, kqinn.EmptyModule()).KQIbackward(volume)
        volume = kqinn.Branch(self.sa_block, kqinn.EmptyModule()).KQIbackward(volume)
        logging.debug(f'TransformerEncoderLayer: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume


class sa_block(KQI):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        self.self_attn = kqinn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = kqinn.LayerNorm(d_model)
        self.dropout1 = kqinn.Dropout(dropout)

    def KQIforward(self, x: Tensor) -> Tensor:
        x = self.norm1.KQIforward(x)
        x = self.self_attn.KQIforward(x, x, x)[0]
        x = self.dropout1.KQIforward(x)

        return x

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        volume = self.dropout1.KQIbackward(volume)

        tmp_kqi = KQI.kqi.clone()
        volume_backward_k, volume_backward_q, volume_backward_v = self.self_attn.KQIbackward(volume)
        volume_kqv = volume_backward_q + volume_backward_k + volume_backward_v
        KQI.kqi = tmp_kqi
        self.self_attn.KQIbackward(volume=volume, volume_backward_k=volume_kqv, volume_backward_q=volume_kqv,
                                   volume_backward_v=volume_kqv)
        volume = self.norm1.KQIbackward(volume_kqv, volume_backward)
        logging.debug(f'sa_block: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume


class ff_block(KQI):
    def __init__(self, d_model: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        self.linear1 = kqinn.Linear(d_model, dim_feedforward)
        self.dropout = kqinn.Dropout(dropout)
        self.linear2 = kqinn.Linear(dim_feedforward, d_model)
        self.norm2 = kqinn.LayerNorm(d_model)
        self.dropout2 = kqinn.Dropout(dropout)

    def KQIforward(self, x: Tensor) -> Tensor:
        x = self.norm2.KQIforward(x)
        x = self.linear1.KQIforward(x)
        x = self.dropout.KQIforward(x)
        x = self.linear2.KQIforward(x)
        x = self.dropout2.KQIforward(x)

        return x

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        volume = self.dropout2.KQIbackward(volume)
        volume = self.linear2.KQIbackward(volume)
        volume = self.dropout.KQIbackward(volume)
        volume = self.linear1.KQIbackward(volume)
        volume = self.norm2.KQIbackward(volume, volume_backward)
        logging.debug(f'ff_block: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume


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



