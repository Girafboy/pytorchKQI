import torch
import numpy as np
import logging
import itertools

from .kqi import KQI


class MaxPool2d(torch.nn.MaxPool2d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        padding = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)

        self.input_size = x.shape
        x_new = self.forward(x)
        assert x_new.shape[-3] == x.size(0)

        if padding[0] or padding[1]:
            C_new, H_new, W_new = x_new.shape
            padding_weights = 0
            # calculating the weights connected to the padding, which are redundant weights
            if self.stride == self.kernel_size:
                # when stride is the default value (stride = kernel_size)
                corner = kernel_size[0] * padding[1] + kernel_size[1] * padding[0] - padding[0] * padding[1]
                top_edge = (W_new - 2) * kernel_size[1] * padding[0]
                bottom_edge = top_edge
                left_edge = (H_new - 2) * kernel_size[0] * padding[1]
                right_edge = left_edge
                padding_weights = 4 * corner + top_edge + bottom_edge + left_edge + right_edge

            KQI.W += np.prod(x_new.shape) * np.prod(kernel_size)
            KQI.W -= padding_weights

        else:
            KQI.W += np.prod(x_new.shape) * np.prod(kernel_size)

        return x_new
    

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        C, H, W = volume.shape
        # nn.Conv2d always stores the below parameters as tuples, while nn.MaxPool2d might store them as integers if they are initialized as int
        kernel_size = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        dilation = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)
        stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        padding = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)


        if padding[0] or padding[1]:
            expanded_volume = torch.zeros((C, H + 2 * padding[0], W + 2 * padding[1]))
            expanded_volume.fill_(volume.mean().item() / np.prod(kernel_size))
            expanded_volume[:, padding[0]:padding[0] + H, padding[1]:padding[1] + W] = volume

            if volume_backward is None:
                volume_backward = torch.zeros_like(expanded_volume)
            for c, i, j in itertools.product(range(expanded_volume.size(0)),
                                             range(0, kernel_size[0] * dilation[0], dilation[0]),
                                             range(0, kernel_size[1] * dilation[1], dilation[1])):
                expanded_H, expanded_W = H + 2 * padding[0], W + 2 * padding[1]
                volume_backward[c, i:expanded_H * stride[0] + i:stride[0], j:expanded_W * stride[1] + j:stride[1]] += 1 + (
                            expanded_volume / np.prod(kernel_size) / expanded_volume.size(0)).sum(dim=0)

            for c, i, j in itertools.product(range(volume.size(0)),
                                             range(0, kernel_size[0] * dilation[0], dilation[0]),
                                             range(0, kernel_size[1] * dilation[1], dilation[1])):
                KQI.kqi += self.KQI_formula((volume[0] / np.prod(kernel_size)),
                                            volume_backward[c, i:H * stride[0] + i:stride[0],
                                            j:W * stride[1] + j:stride[1]])
            # Extract the non-padding part
            volume_backward = volume_backward[:, padding[0]:H + padding[0], padding[1]:W + padding[1]]


        else:
            if volume_backward is None:
                volume_backward = torch.zeros(self.input_size)
            for c, i, j in itertools.product(range(volume.size(0)),
                                             range(0, kernel_size[0] * dilation[0], dilation[0]),
                                             range(0, kernel_size[1] * dilation[1], dilation[1])):
                volume_backward[c, i:H * stride[0] + i:stride[0], j:W * stride[1] + j:stride[1]] += 1 + (
                        volume / np.prod(kernel_size) / volume.size(0)).sum(dim=0)

            for c, i, j in itertools.product(range(volume.size(0)),
                                             range(0, kernel_size[0] * dilation[0], dilation[0]),
                                             range(0, kernel_size[1] * dilation[1], dilation[1])):
                KQI.kqi += self.KQI_formula((volume[0] / np.prod(kernel_size)),
                                            volume_backward[c, i:H * stride[0] + i:stride[0],
                                            j:W * stride[1] + j:stride[1]])

        logging.debug(f'MaxPool2d: KQI={KQI.kqi}, node={np.product(volume.shape)}, volume={volume.sum()}')
        return volume_backward


