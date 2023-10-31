import torch
import numpy as np
import logging

from .kqi import KQI


class Threshold(torch.nn.Threshold, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Threshold: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class ReLU(torch.nn.ReLU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'ReLU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Hardtanh(torch.nn.Hardtanh, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Hardtanh: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class ReLU6(torch.nn.ReLU6, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'ReLU6: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Sigmoid(torch.nn.Sigmoid, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Sigmoid: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Tanh(torch.nn.Tanh, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Tanh: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Softmax(torch.nn.Softmax, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape) * x.shape[self.dim]
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = torch.mean(volume, self.dim, True).expand(volume.shape) + volume.shape[self.dim]

        KQI.kqi += self.KQI_formula(volume / volume.shape[self.dim], volume_backward) * volume.shape[self.dim]

        logging.debug(f'Softmax: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Softmax2d(torch.nn.Softmax2d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape) * x.shape[-3]
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = torch.mean(volume, -3, True).expand(volume.shape) + volume.shape[-3]

        KQI.kqi += self.KQI_formula(volume / volume.shape[-3], volume_backward) * volume.shape[-3]

        logging.debug(f'Softmax2d: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class LogSoftmax(torch.nn.LogSoftmax, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape) * x.shape[self.dim]
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = torch.mean(volume, self.dim, True).expand(volume.shape) + volume.shape[self.dim]

        KQI.kqi += self.KQI_formula(volume / volume.shape[self.dim], volume_backward) * volume.shape[self.dim]

        logging.debug(f'LogSoftmax: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class ELU(torch.nn.ELU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'ELU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class SELU(torch.nn.SELU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'SELU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class CELU(torch.nn.CELU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'CELU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class GELU(torch.nn.GELU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'GELU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Hardshrink(torch.nn.Hardshrink, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Hardshrink: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class LeakyReLU(torch.nn.LeakyReLU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'LeakyReLU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class LogSigmoid(torch.nn.LogSigmoid, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'LogSigmoid: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Softplus(torch.nn.Softplus, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Softplus: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Softshrink(torch.nn.Softshrink, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Softshrink: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class PReLU(torch.nn.PReLU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'PReLU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Softsign(torch.nn.Softshrink, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Softsign: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Softmin(torch.nn.Softmax, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape) * x.shape[self.dim]
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = torch.mean(volume, self.dim, True).expand(volume.shape) + volume.shape[self.dim]

        KQI.kqi += self.KQI_formula(volume / volume.shape[self.dim], volume_backward) * volume.shape[self.dim]

        logging.debug(f'Softmin: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Tanhshrink(torch.nn.Tanhshrink, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Tanhshrink: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class RReLU(torch.nn.RReLU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'RReLU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class GLU(torch.nn.GLU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward_half = volume / 2 + 1
            volume_backward = torch.cat((volume_backward_half, volume_backward_half), dim=self.dim)
        KQI.kqi += 2*self.KQI_formula(volume / 2, volume_backward_half)
        logging.debug(f'GLU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Hardsigmoid(torch.nn.Hardsigmoid, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Hardsigmoid: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Hardswish(torch.nn.Hardswish, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Hardswish: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class SiLU(torch.nn.SiLU, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'SiLU: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward


class Mish(torch.nn.Mish, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W += np.prod(x.shape)
        return self.forward(x)

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        if volume_backward is None:
            volume_backward = volume + 1
        KQI.kqi += self.KQI_formula(volume, volume_backward)
        logging.debug(f'Mish: KQI={KQI.kqi}, node={np.prod(volume.shape)}, volume={volume.sum()}')
        return volume_backward
