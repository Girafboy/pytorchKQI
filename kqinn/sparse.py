import torch
import numpy as np
import itertools
import logging

from .kqi import KQI

class Embedding(torch.nn.Embedding, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape) * self.embedding_dim
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        C, H = volume.shape[:2]
        if volume_backward is None:
            volume_backward = torch.zeros(volume.shape[:2])
            for i, j in itertools.product(range(C), range(H)):
                volume_backward[i, j] += self.embedding_dim + volume[i, j, :].sum()
        for i, j, k in itertools.product(range(C), range(H), range(self.embedding_dim)):
            KQI.kqi += self.KQI_formula(volume[i,j,k], volume_backward[i,j])
        logging.debug(f'Embedding: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward
