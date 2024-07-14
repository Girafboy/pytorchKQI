import logging
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

import kqinn
from .kqi import KQI


class Transformer(torch.nn.Transformer, KQI):
    """
    This module is modified from torch.nn.Transformer.
    We only consider the case of norm_first=True.
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.d_model = d_model
        self.nhead = nhead
        self.encoder = TransformerEncoder(TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, norm_first=True), num_encoder_layers)
        self.decoder = TransformerDecoder(TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, norm_first=True), num_decoder_layers)

    def KQIforward(self, src: Tensor, tgt: Tensor) -> Tensor:
        memory = self.encoder.KQIforward(src)
        return self.decoder.KQIforward(tgt, memory)

    def KQIbackward(self, volume: Tensor, volume_backward_src: Tensor = None, volume_backward_tgt: Tensor = None) -> Tuple[Tensor, Tensor]:
        volume_tgt, volume_mem = self.decoder.KQIbackward(volume, volume_backward_tgt, None)
        volume_src = self.encoder.KQIbackward(volume_mem, volume_backward_src)
        logging.debug(f'Transformer: KQI={KQI.kqi}, node={np.prod(volume_src.shape) + np.prod(volume_tgt.shape)}, volume={volume_src.sum() + volume_tgt.sum()}')
        return volume_src, volume_tgt


class TransformerEncoder(torch.nn.TransformerEncoder, KQI):
    """
    This module is modified from torch.nn.TransformerEncoder.
    We only consider the case of norm_first=True.
    """

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True):
        super().__init__(encoder_layer, num_layers, norm, enable_nested_tensor)
        if self.norm is not None:
            raise NotImplementedError('norm is not None is not supported now')
        self.encoder_layer = encoder_layer
        self.num_layers = num_layers

    def KQIforward(self, x: Tensor) -> Tensor:
        for i in range(self.num_layers):
            self.encoder_layer.KQIforward(x)
        return self.forward(x)

    def KQIbackward(self, volume: Tensor, volume_backward: Tensor = None) -> Tensor:
        for i in range(self.num_layers - 1):
            volume = self.encoder_layer.KQIbackward(volume, None)
        volume = self.encoder_layer.KQIbackward(volume, volume_backward)
        logging.debug(f'TransformerEncoder: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume


class TransformerDecoder(torch.nn.TransformerDecoder, KQI):
    """
    This module is modified from torch.nn.TransformerDecoder.
    We only consider the case of norm_first=True.
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__(decoder_layer, num_layers, norm)
        if self.norm is not None:
            raise NotImplementedError('norm is not None is not supported now')
        self.decoder_layer = decoder_layer
        self.num_layers = num_layers

    def KQIforward(self, x: Tensor, memory: Tensor) -> Tensor:
        for i in range(self.num_layers):
            x = self.decoder_layer.KQIforward(x, memory)

        return self.forward(x, memory)

    def KQIbackward(self, volume: Tensor, volume_backward: Tensor = None, volume_backward_mem: Tensor = None) -> Tuple[Tensor, Tensor]:
        volume_mem = torch.zeros_like(volume)

        tmp_kqi = KQI.kqi.clone()
        tmp_volume = volume.clone()
        logging.debug('>>> TransformerDecoder: Trying')
        for i in range(self.num_layers - 1):
            volume, volume_mem_tmp = self.decoder_layer.KQIbackward(volume, None)
            volume_mem += volume_mem_tmp
        volume, volume_mem_tmp = self.decoder_layer.KQIbackward(volume, volume_backward)
        volume_mem += volume_mem_tmp
        logging.debug('<<< TransformerDecoder: Trying down')

        KQI.kqi = tmp_kqi
        volume = tmp_volume
        if volume_backward_mem is None:
            for i in range(self.num_layers - 1):
                volume = self.decoder_layer.KQIbackward(volume, None, volume_mem)[0]
            volume = self.decoder_layer.KQIbackward(volume, volume_backward, volume_mem)[0]
        else:
            for i in range(self.num_layers - 1):
                volume = self.decoder_layer.KQIbackward(volume, None, volume_backward_mem)[0]
            volume = self.decoder_layer.KQIbackward(volume, volume_backward, volume_backward_mem)[0]

        logging.debug(f'TransformerDecoder: KQI={KQI.kqi}, node={np.prod(volume.shape) + np.prod(volume_mem.shape)}, volume={volume.sum() + volume_mem.sum()}, '
                      f'tgt_node={np.prod(volume.shape)}, tgt_volume={volume.sum()}, mem_node={np.prod(volume_mem.shape)}, mem_volume={volume_mem.sum()}')
        return volume, volume_mem


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


class TransformerDecoderLayer(torch.nn.TransformerDecoderLayer, KQI):
    """
    This module is modified from torch.nn.TransformerDecoderLayer.
    We only consider the case of norm_first=True.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = 'relu', layer_norm_eps: float = 1e-5, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None):
        if not norm_first:
            raise NotImplementedError('norm_first=False is not supported now')

        super().__init__(d_model, nhead, dim_feedforward, norm_first=norm_first)
        self.norm_first = norm_first
        self.sa_block = sa_block(d_model, nhead, dropout)
        self.mha_block = mha_block(d_model, nhead, dropout)
        self.ff_block = ff_block(d_model, dim_feedforward, dropout)

    def KQIforward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = kqinn.Branch(self.sa_block, kqinn.EmptyModule()).KQIforward(x)
        x = self.mha_block.KQIforward(x, memory)
        KQI.W += np.prod(x.shape) * 2
        x = kqinn.Branch(self.ff_block, kqinn.EmptyModule()).KQIforward(x)

        return self.forward(x, memory)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None,
                    volume_backward_mem: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        volume = kqinn.Branch(self.ff_block, kqinn.EmptyModule()).KQIbackward(volume)

        tmp_kqi = KQI.kqi.clone()
        tmp_volume = volume.clone()
        logging.debug('>>> mha_block: Trying')
        # Adding one here is because there is a hidden layer of addition
        volume, volume_mem = self.mha_block.KQIbackward(tmp_volume / 2 + 1, None, volume_backward_mem)
        logging.debug('<<< mha_block: Trying down')
        volume += tmp_volume / 2
        KQI.kqi = tmp_kqi
        volume, volume_mem = self.mha_block.KQIbackward(tmp_volume / 2 + 1, volume, volume_backward_mem)
        KQI.kqi += self.KQI_formula(tmp_volume / 2, tmp_volume / 2 + 1)
        KQI.kqi += self.KQI_formula(tmp_volume / 2, volume + 1)

        logging.debug(f'mha_block: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')

        volume = kqinn.Branch(self.sa_block, kqinn.EmptyModule()).KQIbackward(volume + 1)

        logging.debug(f'TransformerDecoderLayer: KQI={KQI.kqi}, node={np.prod(volume.shape) + np.prod(volume_mem.shape)}, volume={volume.sum() + volume_mem.sum()}, '
                      f'tgt_node={np.prod(volume.shape)}, tgt_volume={volume.sum()}, mem_node={np.prod(volume_mem.shape)}, mem_volume={volume_mem.sum()}')
        return volume, volume_mem


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


class mha_block(KQI):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        self.multihead_attn = kqinn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm2 = kqinn.LayerNorm(d_model)
        self.dropout2 = kqinn.Dropout(dropout)

    def KQIforward(self, x: Tensor, memory: Tensor) -> Tensor:
        x = self.norm2.KQIforward(x)
        x = self.multihead_attn.KQIforward(x, memory, memory)[0]
        x = self.dropout2.KQIforward(x)

        return x

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None,
                    volume_backward_mem: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        volume = self.dropout2.KQIbackward(volume)

        # Save the original KQI and volume
        tmp_kqi = KQI.kqi.clone()
        tmp_volume = volume.clone()
        # Calculate the volume for the backward pass
        logging.debug('>>> multihead_attn: Trying')
        volume_backward_k, volume_backward_q, volume_backward_v = self.multihead_attn.KQIbackward(volume)
        logging.debug('<<< multihead_attn: Trying down')
        # Get true volume
        volume_mem = volume_backward_k + volume_backward_v
        volume = volume_backward_q
        # Restore the original KQI and volume and perform the backward pass
        KQI.kqi = tmp_kqi
        if volume_backward_mem is None:
            self.multihead_attn.KQIbackward(volume=tmp_volume, volume_backward_k=volume_mem, volume_backward_q=volume,
                                            volume_backward_v=volume_mem)
        else:
            self.multihead_attn.KQIbackward(volume=tmp_volume, volume_backward_k=volume_backward_mem,
                                            volume_backward_q=volume, volume_backward_v=volume_backward_mem)

        volume = self.norm2.KQIbackward(volume, volume_backward)
        logging.debug(f'mha_block: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume, volume_mem
