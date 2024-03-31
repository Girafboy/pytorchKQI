import torch
import numpy as np
import itertools
import logging
from typing import Tuple, Optional

from .kqi import KQI


class RNN(torch.nn.RNN, KQI):
    def KQIforward(self, x: torch.Tensor, hx: torch.Tensor = None) -> torch.Tensor:
        KQI.W += (x.shape[0] - 1) * ((self.input_size + self.hidden_size) * self.hidden_size + self.hidden_size) + self.input_size * self.hidden_size + self.hidden_size
        if self.num_layers > 1:
            KQI.W += (self.num_layers - 1) * ((x.shape[0] - 1) * ((self.hidden_size + self.hidden_size) * self.hidden_size + self.hidden_size) + self.hidden_size * self.hidden_size + self.hidden_size)

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
            for vol in itertools.chain(volume_fore[i - 1], volume_back[i]):
                KQI.kqi += self.KQI_formula(volume_hidden / (volume_back.shape[1] + self.hidden_size), vol)

        # Tanh top
        volume_hidden = volume_fore[0] + 1
        KQI.kqi += self.KQI_formula(volume_fore[0], volume_hidden)
        # Linear top
        for vol in volume_back[0]:
            KQI.kqi += self.KQI_formula(volume_hidden / volume_back.shape[1], vol)

    def _single_layer_volume(self, volume_fore, volume_back):
        for i in reversed(range(1, volume_fore.shape[0])):
            # Tanh
            volume_hidden = volume_fore[i] + 1
            # Linear
            volume_back[i] += self.hidden_size + volume_hidden.sum() / (volume_back.shape[1] + self.hidden_size)
            volume_fore[i - 1] += self.hidden_size + volume_hidden.sum() / (volume_back.shape[1] + self.hidden_size)

        # Tanh top
        volume_hidden = volume_fore[0] + 1
        # Linear top
        volume_back[0] += self.hidden_size + volume_hidden.sum() / volume_back.shape[1]

        return volume_fore, volume_back


class LSTM(torch.nn.LSTM, KQI):
    def KQIforward(self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        KQI.W += (self.input_size * self.hidden_size + self.hidden_size) * 4 + self.hidden_size * 8 + (x.shape[0] - 1) * (((self.input_size + self.hidden_size) * self.hidden_size + self.hidden_size) * 4 + self.hidden_size * 9)
        if self.num_layers > 1:
            KQI.W += (self.num_layers - 1) * (self.hidden_size * self.hidden_size + self.hidden_size) * 4 + self.hidden_size * 8 + (x.shape[0] - 1) * (((self.hidden_size + self.hidden_size) * self.hidden_size + self.hidden_size) * 4 + self.hidden_size * 9)
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

        logging.debug(f'LSTM: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward

    def _single_layer_kqi(self, volume_fore, volume_back):
        volume_hidden_2_1_list = []
        volume_Ct_list = []
        for i in reversed(range(0, volume_fore.shape[0])):
            volume_hidden_3_1 = volume_fore[i] / 2 + 1
            KQI.kqi += self.KQI_formula(volume_fore[i] / 2, volume_hidden_3_1)  # ht half
            volume_Ot = volume_fore[i] / 2 + 1
            KQI.kqi += self.KQI_formula(volume_fore[i] / 2, volume_Ot)  # ht half
            if i == volume_fore.shape[0] - 1:
                volume_Ct = volume_hidden_3_1 + 1
            else:
                volume_Ct = volume_hidden_3_1 + 1 + volume_hidden_2_1_list[0] / 2 + 1
            volume_Ct_list.insert(0, volume_Ct)
            KQI.kqi += self.KQI_formula(volume_hidden_3_1, volume_Ct)  # hidden_3_1

            volume_hidden_2_1 = volume_Ct / 2 + 1
            volume_hidden_2_1_list.insert(0, volume_hidden_2_1)
            KQI.kqi += self.KQI_formula(volume_Ct / 2, volume_hidden_2_1)  # Ct half
            volume_hidden_2_2 = volume_Ct / 2 + 1
            KQI.kqi += self.KQI_formula(volume_Ct / 2, volume_hidden_2_2)  # Ct half

            if i > 0:
                volume_ft = volume_hidden_2_1 / 2 + 1
                KQI.kqi += self.KQI_formula(volume_hidden_2_1 / 2, volume_ft)  # hidden_2_1 half
            else:
                volume_ft = volume_hidden_2_1 + 1
                KQI.kqi += self.KQI_formula(volume_hidden_2_1, volume_ft)  # hidden_2_1
            volume_it = volume_hidden_2_2 / 2 + 1
            KQI.kqi += self.KQI_formula(volume_hidden_2_2 / 2, volume_it)  # hidden_2_2 half
            volume_C_t = volume_hidden_2_2 / 2 + 1
            KQI.kqi += self.KQI_formula(volume_hidden_2_2 / 2, volume_C_t)  # hidden_2_2 half

            volume_hidden_1_1 = volume_ft + 1
            KQI.kqi += self.KQI_formula(volume_ft, volume_hidden_1_1)  # ft
            volume_hidden_1_2 = volume_it + 1
            KQI.kqi += self.KQI_formula(volume_it, volume_hidden_1_2)  # it
            volume_hidden_1_3 = volume_C_t + 1
            KQI.kqi += self.KQI_formula(volume_C_t, volume_hidden_1_3)  # C_t
            volume_hidden_1_4 = volume_Ot + 1
            KQI.kqi += self.KQI_formula(volume_Ot, volume_hidden_1_4)  # Ot
            volume_hiddens = [volume_hidden_1_1, volume_hidden_1_2, volume_hidden_1_3, volume_hidden_1_4]

            if i > 0:
                for volume_hidden in volume_hiddens:
                    for vol in itertools.chain(volume_fore[i - 1], volume_back[i]):
                        KQI.kqi += self.KQI_formula(volume_hidden / (volume_back.shape[1] + self.hidden_size), vol)
            elif i == 0:
                for volume_hidden in volume_hiddens:
                    for vol in volume_back[0]:
                        KQI.kqi += self.KQI_formula(volume_hidden / volume_back.shape[1], vol)

        for i in reversed(range(0, volume_fore.shape[0])):
            if i > 0:
                KQI.kqi += self.KQI_formula(volume_hidden_2_1_list[i] / 2, volume_Ct_list[i - 1])  # hidden_2_1 half

    def _single_layer_volume(self, volume_fore, volume_back):
        volume_hidden_2_1_list = []
        for i in reversed(range(0, volume_fore.shape[0])):
            volume_hidden_3_1 = volume_fore[i] / 2 + 1
            volume_Ot = volume_fore[i] / 2 + 1
            if i == volume_fore.shape[0] - 1:
                volume_Ct = volume_hidden_3_1 + 1
            else:
                volume_Ct = volume_hidden_3_1 + 1 + volume_hidden_2_1_list[0] / 2 + 1

            volume_hidden_2_1 = volume_Ct / 2 + 1
            volume_hidden_2_1_list.insert(0, volume_hidden_2_1)
            volume_hidden_2_2 = volume_Ct / 2 + 1
            if i > 0:
                volume_ft = volume_hidden_2_1 / 2 + 1
            else:
                volume_ft = volume_hidden_2_1 + 1
            volume_it = volume_hidden_2_2 / 2 + 1
            volume_C_t = volume_hidden_2_2 / 2 + 1

            volume_hidden_1_1 = volume_ft + 1
            volume_hidden_1_2 = volume_it + 1
            volume_hidden_1_3 = volume_C_t + 1
            volume_hidden_1_4 = volume_Ot + 1
            volume_hiddens = [volume_hidden_1_1, volume_hidden_1_2, volume_hidden_1_3, volume_hidden_1_4]

            if i > 0:
                for volume_hidden in volume_hiddens:
                    volume_back[i] += self.hidden_size + volume_hidden.sum() / (volume_back.shape[1] + self.hidden_size)
                    volume_fore[i - 1] += self.hidden_size + volume_hidden.sum() / (volume_back.shape[1] + self.hidden_size)
            elif i == 0:
                for volume_hidden in volume_hiddens:
                    volume_back[0] += self.hidden_size + volume_hidden.sum() / volume_back.shape[1]

        return volume_fore, volume_back


class GRU(torch.nn.RNN, KQI):
    def KQIforward(self, x: torch.Tensor, hx: torch.Tensor = None) -> torch.Tensor:
        KQI.W += self.input_size * self.hidden_size * 3 + self.hidden_size * 11 + (x.shape[0] - 1) * ((self.input_size + self.hidden_size) * self.hidden_size * 3 + self.hidden_size * 13)
        if self.num_layers > 1:
            KQI.W += (self.num_layers - 1) * (self.hidden_size * self.hidden_size * 3 + self.hidden_size * 11 + (x.shape[0] - 1) * ((self.hidden_size + self.hidden_size) * self.hidden_size * 3 + self.hidden_size * 13))

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

        logging.debug(f'GRU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward

    def _single_layer_kqi(self, volume_fore, volume_back):
        for i in reversed(range(0, volume_fore.shape[0])):
            volume_ht_pre_left = volume_fore[i] / 2 + 1
            KQI.kqi += self.KQI_formula(volume_fore[i] / 2, volume_ht_pre_left)  # ht
            volume_ht_pre_right = volume_fore[i] / 2 + 1
            KQI.kqi += self.KQI_formula(volume_fore[i] / 2, volume_ht_pre_right)  # ht
            volume_nt = volume_ht_pre_left / 2 + 1
            KQI.kqi += self.KQI_formula(volume_ht_pre_left / 2, volume_nt)
            volume_1_zt = volume_ht_pre_left / 2 + 1
            KQI.kqi += self.KQI_formula(volume_ht_pre_left / 2, volume_1_zt)  # 1-zt
            volume_nt_pre = volume_nt + 1
            KQI.kqi += self.KQI_formula(volume_nt, volume_nt_pre)  # nt
            volume_rt_hn = volume_nt_pre / (volume_back.shape[1] + 1) + 1  # pay attention to volume_xt
            KQI.kqi += self.KQI_formula(volume_nt_pre / (volume_back.shape[1] + 1), volume_rt_hn)
            for vol in volume_back[i]:
                KQI.kqi += self.KQI_formula(volume_nt_pre / (volume_back.shape[1] + 1), vol)

            volume_zt = volume_1_zt + 1  # Can't caculate KQI_zt now

            if i > 0:
                volume_hn = volume_rt_hn / 2 + 1
                KQI.kqi += self.KQI_formula(volume_rt_hn / 2, volume_hn)

                volume_rt = volume_rt_hn / 2 + 1
                KQI.kqi += self.KQI_formula(volume_rt_hn / 2, volume_rt)
                volume_hr = volume_rt + 1
                KQI.kqi += self.KQI_formula(volume_rt, volume_hr)

                volume_zt += volume_ht_pre_right / 2 + 1
                KQI.kqi += self.KQI_formula(volume_ht_pre_right / 2, volume_zt)
                KQI.kqi += self.KQI_formula(volume_1_zt, volume_zt)
                volume_hz = volume_zt + 1
                KQI.kqi += self.KQI_formula(volume_zt, volume_hz)

                for vol in itertools.chain(volume_fore[i - 1], volume_back[i]):
                    KQI.kqi += self.KQI_formula(volume_hr / (volume_back.shape[1] + self.hidden_size), vol)  # hr
                    KQI.kqi += self.KQI_formula(volume_hz / (volume_back.shape[1] + self.hidden_size), vol)  # hz
                for vol in volume_fore[i - 1]:
                    KQI.kqi += self.KQI_formula(volume_hn / self.hidden_size, vol)  # hn
                KQI.kqi += self.KQI_formula(volume_ht_pre_right / 2, volume_fore[i - 1])

            elif i == 0:
                volume_rt = volume_rt_hn + 1
                KQI.kqi += self.KQI_formula(volume_rt_hn, volume_rt)
                volume_hr = volume_rt + 1
                KQI.kqi += self.KQI_formula(volume_rt, volume_hr)

                volume_zt += volume_ht_pre_right + 1
                KQI.kqi += self.KQI_formula(volume_ht_pre_right, volume_zt)
                KQI.kqi += self.KQI_formula(volume_1_zt, volume_zt)
                volume_hz = volume_zt + 1
                KQI.kqi += self.KQI_formula(volume_zt, volume_hz)

                for vol in volume_back[i]:
                    KQI.kqi += self.KQI_formula(volume_hr / volume_back.shape[1], vol)
                    KQI.kqi += self.KQI_formula(volume_hz / volume_back.shape[1], vol)

    def _single_layer_volume(self, volume_fore, volume_back):
        for i in reversed(range(0, volume_fore.shape[0])):
            volume_ht_pre_left = volume_fore[i] / 2 + 1
            volume_ht_pre_right = volume_fore[i] / 2 + 1
            volume_nt = volume_ht_pre_left / 2 + 1
            volume_1_zt = volume_ht_pre_left / 2 + 1
            volume_nt_pre = volume_nt + 1
            volume_rt_hn = volume_nt_pre / (volume_back.shape[1] + 1) + 1  # pay attention to volume_xt
            volume_back[i] += self.hidden_size + volume_nt_pre.sum() / (volume_back.shape[1] + 1)
            volume_zt = volume_1_zt + 1

            if i > 0:
                volume_hn = volume_rt_hn / 2 + 1
                volume_rt = volume_rt_hn / 2 + 1
                volume_hr = volume_rt + 1
                volume_zt += volume_ht_pre_right / 2 + 1
                volume_hz = volume_zt + 1
                volume_back[i] += self.hidden_size + volume_hr.sum() / (volume_back.shape[1] + self.hidden_size)  # From hr
                volume_back[i] += self.hidden_size + volume_hz.sum() / (volume_back.shape[1] + self.hidden_size)  # From hz
                volume_fore[i - 1] += self.hidden_size + volume_hr.sum() / (volume_back.shape[1] + self.hidden_size)  # From hr
                volume_fore[i - 1] += self.hidden_size + volume_hz.sum() / (volume_back.shape[1] + self.hidden_size)  # From hz
                volume_fore[i - 1] += self.hidden_size + volume_hn.sum() / self.hidden_size  # From hn
                volume_fore[i - 1] += volume_ht_pre_right / 2 + 1
            elif i == 0:
                volume_rt = volume_rt_hn + 1
                volume_hr = volume_rt + 1
                volume_zt += volume_ht_pre_right + 1
                volume_hz = volume_zt + 1
                volume_back[i] += self.hidden_size + volume_hr.sum() / volume_back.shape[1]  # From hr
                volume_back[i] += self.hidden_size + volume_hz.sum() / volume_back.shape[1]  # From hz

        return volume_fore, volume_back
