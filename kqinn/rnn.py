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

        volume_backward = self.caculate_kqi(volume, volume_backward)

        logging.debug(f'RNN: KQI={KQI.kqi}, node={np.product(volume.shape)}, volume={volume.sum()}')
        return volume_backward
    


    def caculate_volume(self, volume: torch.Tensor):

        def single_layer_volume(hidden_size, input_size, volume_output):
            backward_list = []
            length = volume_output.shape[0]
            for i in reversed(range(length)):
                # Tanh volume
                volume_hidden = volume_output[i] + 1
                # linear volume
                in_features_size = input_size + hidden_size if i > 0 else input_size
                volume_linear = torch.ones(in_features_size) * (hidden_size + (volume_hidden / in_features_size).sum())

                backward_list.insert(0, volume_linear[:input_size])
                if i>0:
                    volume_output[i-1] += volume_linear[input_size:]
            
            backward_volume = torch.stack(backward_list)
            return backward_volume, volume_output

        layer_volume = []
        volume_temp = volume.clone()
        for i in reversed(range(1, self.num_layers)):
            volume_temp, real_volume = single_layer_volume(self.hidden_size, self.hidden_size, volume_temp)
            layer_volume.insert(0, real_volume)
        volume_temp, real_volume = single_layer_volume(self.hidden_size, self.input_size, volume_temp)
        layer_volume.insert(0, real_volume)
        layer_volume.insert(0, volume_temp)

        return layer_volume


    def caculate_kqi(self, volume: torch.Tensor, volume_backward: torch.Tensor = None):

        def single_layer_kqi(self, volume_output, volume_upper):
            length = volume_output.shape[0]
            input_size = volume_upper.shape[1]
            for i in reversed(range(length)):
                # Tanh
                volume_hidden = volume_output[i] + 1
                KQI.kqi += self.KQI_formula(volume_output[i], volume_hidden)
                # linear
                if i>0:
                    in_features_size = input_size + self.hidden_size
                    for vol in  volume_output[i-1]:
                        KQI.kqi += self.KQI_formula(volume_hidden/in_features_size, vol)
                    for vol in  volume_upper[i]:
                        KQI.kqi += self.KQI_formula(volume_hidden/in_features_size, vol)
                elif i==0:
                    in_features_size = input_size
                    for vol in  volume_upper[i]:
                        KQI.kqi += self.KQI_formula(volume_hidden/in_features_size, vol)

        if volume_backward == None:
            layer_volume = self.caculate_volume(volume)
        else:
            layer_volume = self.caculate_volume(volume)
            layer_volume[0] += volume_backward
        for i in reversed(range(1, self.num_layers+1)):
            single_layer_kqi(self, layer_volume[i], layer_volume[i-1])

        return layer_volume[0]
   

