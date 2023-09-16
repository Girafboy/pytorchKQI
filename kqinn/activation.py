import torch
import numpy as np
import logging

from .kqi import KQI


class ReLU(torch.nn.ReLU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'ReLU: KQI={KQI.kqi}, node={np.product(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class SoftMax(torch.nn.Softmax, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_dim = x.ndim
        self.input_size = x.shape
        KQI.W += np.prod(x.shape) * x.shape[self.dim]
        # print(KQI.W)
        # print(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = torch.zeros(self.input_size)

        volume_backward_fal = volume_backward.flatten()
        volume_fal = volume.flatten()
        # print(f'{volume_backward_fal}=shape')
        num = np.prod(self.input_size)
        # print(f'input_dim={self.input_dim}')
        # print(f'num={num}')
        tensor_set = np.prod(self.input_size[self.dim:])
        # print(f'tensor_set={tensor_set}')
        tensor_set_num = int(num / tensor_set)
        # print(f'tensor_set_num={tensor_set_num}')
        per_tensor_size = self.input_size[self.dim]
        # print(per_tensor_size)
        for i in range(tensor_set_num):
            tensor_num = int(tensor_set / self.input_size[self.dim])
            if self.dim == self.input_dim - 1 | self.dim == -1:
                interval = 1
            else:
                interval = np.prod(self.input_size[self.dim + 1:])
            for j in range(tensor_num):
                vector = []
                for k in range(per_tensor_size):
                    mm = int(i * tensor_set + j + k * interval)
                    vector.append(mm)
                # print(vector)
                for m in vector:
                    volume_backward_fal[m] += per_tensor_size
                    # print(volume_backward_fal[m])
                    for n in vector:
                        volume_backward_fal[m] += volume_fal[n] / per_tensor_size
                for A in vector:
                    for B in vector:
                        if volume_fal[A].item() == 0:
                            continue
                        else:
                            kqi = - volume_fal[A].item() / per_tensor_size / KQI.W * np.log2(
                                volume_fal[A].item() / per_tensor_size / volume_backward_fal[B].item())
                            KQI.kqi += kqi
        volume_backward_fal.reshape(self.input_size)
        volume_fal.reshape(self.input_size)
        # print(f'KQI={KQI.kqi}')
        logging.debug(f'SoftMax: KQI={KQI.kqi}, node={np.product(volume.shape)}, volume={volume.sum()}')
        return volume_backward
