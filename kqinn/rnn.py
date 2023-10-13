import torch
import numpy as np
import itertools
import logging

from .kqi import KQI


class RNN(torch.nn.RNN, KQI):
    def KQIforward(self, x: torch.Tensor, hx: torch.Tensor = None) -> torch.Tensor:
        KQI.W += (x.shape[0]-1)*((self.input_size+self.hidden_size) * self.hidden_size + self.hidden_size) + self.input_size * self.hidden_size + self.hidden_size
        if self.num_layers > 1:
            KQI.W += (self.num_layers-1) * ((x.shape[0]-1)*((self.hidden_size+self.hidden_size) * self.hidden_size + self.hidden_size) + self.hidden_size * self.hidden_size + self.hidden_size)

        return self.forward(x, hx)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        volume_layers = [torch.zeros((volume.shape[0], self.hidden_size))] * self.num_layers + [torch.zeros((volume.shape[0], self.input_size))]
        volume_layers[0] = volume
        for volume_fore, volume_back in zip(volume_layers[:-1], volume_layers[1:]):
            self._single_layer_volume(volume_fore, volume_back)

        if volume_backward is None:
            volume_backward = volume_layers[-1]
        else:
            volume_layers[-1] = volume_backward

        for volume_fore, volume_back in zip(volume_layers[:-1], volume_layers[1:]):
            self._single_layer_kqi(volume_fore, volume_back)

        logging.debug(f'RNN: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward

    def _single_layer_kqi(self, volume_fore, volume_back):
        for i in reversed(range(1, volume_fore.shape[0])):
            # Tanh
            volume_hidden = volume_fore[i] + 1
            KQI.kqi += self.KQI_formula(volume_fore[i], volume_hidden)
            # Linear
            for vol in itertools.chain(volume_fore[i-1], volume_back[i]):
                KQI.kqi += self.KQI_formula(volume_hidden/(volume_back.shape[1] + self.hidden_size), vol)

        # Tanh top
        volume_hidden = volume_fore[0] + 1
        KQI.kqi += self.KQI_formula(volume_fore[0], volume_hidden)
        # Linear top
        for vol in volume_back[0]:
            KQI.kqi += self.KQI_formula(volume_hidden/volume_back.shape[1], vol)

    def _single_layer_volume(self, volume_fore, volume_back):
        for i in reversed(range(1, volume_fore.shape[0])):
            # Tanh
            volume_hidden = volume_fore[i] + 1
            # Linear
            volume_back[i] += self.hidden_size + volume_hidden.sum() / (volume_back.shape[1] + self.hidden_size)
            volume_fore[i-1] += self.hidden_size + volume_hidden.sum() / (volume_back.shape[1] + self.hidden_size)

        # Tanh top
        volume_hidden = volume_fore[0] + 1
        # Linear top
        volume_back[0] += self.hidden_size + volume_hidden.sum() / volume_back.shape[1]

        return volume_fore, volume_back
