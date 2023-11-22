import itertools
import logging

import torch

import kqinn
import kqitool


def test_LayerNorm():
    dim_1 = 3
    dim_2 = 4
    dim_3 = 5

    class TestLayerNorm(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = kqinn.LayerNorm(normalized_shape=dim_3)
            self.layer2 = kqinn.Linear(in_features=dim_1 * dim_2 * dim_3, out_features=dim_1 * dim_2 * dim_3,
                                       bias=False)
            self.layer3 = kqinn.LayerNorm(normalized_shape=[dim_2, dim_3])
            self.layer4 = kqinn.Linear(in_features=dim_1 * dim_2 * dim_3, out_features=dim_1 * dim_2 * dim_3,
                                       bias=False)
            self.layer5 = kqinn.LayerNorm(normalized_shape=[dim_1, dim_2, dim_3])

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(dim_1, dim_2, dim_3))
            x = self.layer4(x.flatten())
            x = self.layer5(x.reshape(dim_1, dim_2, dim_3))
            return x

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.layer1.KQIforward(x)
            x = self.layer2.KQIforward(x.flatten())
            x = self.layer3.KQIforward(x.reshape(dim_1, dim_2, dim_3))
            x = self.layer4.KQIforward(x.flatten())
            x = self.layer5.KQIforward(x.reshape(dim_1, dim_2, dim_3))
            return x

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            volume = self.layer5.KQIbackward(volume)
            volume = self.layer4.KQIbackward(volume.flatten())
            volume = self.layer3.KQIbackward(volume.reshape(dim_1, dim_2, dim_3))
            volume = self.layer2.KQIbackward(volume.flatten())
            volume = self.layer1.KQIbackward(volume.reshape(dim_1, dim_2, dim_3), volume_backward)
            return volume

        def true_kqi(self):
            G = kqitool.DiGraph()

            for i, j, k in itertools.product(range(dim_1), range(dim_2), range(dim_3)):
                G.add_node(f'L0_{i}-{j}-{k}', [])

            # Normalize with the last one dimension
            for i, j in itertools.product(range(dim_1), range(dim_2)):
                preds = [f'L0_{i}-{j}-{k}' for k in range(dim_3)]
                for k in range(dim_3):
                    G.add_node(f'L1_{i}-{j}-{k}', preds)

            # Linear
            preds = [f'L1_{i}-{j}-{k}' for i, j, k in itertools.product(range(dim_1), range(dim_2), range(dim_3))]
            for i, j, k in itertools.product(range(dim_1), range(dim_2), range(dim_3)):
                G.add_node(f'L2_{i}-{j}-{k}', preds)

            # Normalize with the last two dimensions
            for i in range(dim_1):
                preds = [f'L2_{i}-{j}-{k}' for j, k in itertools.product(range(dim_2), range(dim_3))]
                for j, k in itertools.product(range(dim_2), range(dim_3)):
                    G.add_node(f'L3_{i}-{j}-{k}', preds)

            # Linear
            preds = [f'L3_{i}-{j}-{k}' for i, j, k in itertools.product(range(dim_1), range(dim_2), range(dim_3))]
            for i, j, k in itertools.product(range(dim_1), range(dim_2), range(dim_3)):
                G.add_node(f'L4_{i}-{j}-{k}', preds)

            # Normalize with all dimensions
            preds = [f'L4_{i}-{j}-{k}' for i, j, k in itertools.product(range(dim_1), range(dim_2), range(dim_3))]
            for i, j, k in itertools.product(range(dim_1), range(dim_2), range(dim_3)):
                G.add_node(f'L5_{i}-{j}-{k}', preds)

            return sum(map(lambda m: G.kqi(m), G.nodes()))

    kqi = TestLayerNorm().KQI(torch.randn(dim_1, dim_2, dim_3))
    true = TestLayerNorm().true_kqi()
    print("true_kqi: ", true)
    print("kqi: ", kqi)
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test_LayerNorm()
