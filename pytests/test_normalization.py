import itertools
import logging

import torch
import testtool
import kqinn
import kqitool


def test_LayerNorm():
    dim_1 = 3
    dim_2 = 4
    dim_3 = 5

    class TestLayerNorm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = torch.nn.LayerNorm(normalized_shape=dim_3)
            self.layer2 = torch.nn.Linear(in_features=dim_1 * dim_2 * dim_3, out_features=dim_1 * dim_2 * dim_3,
                                          bias=False)
            self.layer3 = torch.nn.LayerNorm(normalized_shape=[dim_2, dim_3])
            self.layer4 = torch.nn.Linear(in_features=dim_1 * dim_2 * dim_3, out_features=dim_1 * dim_2 * dim_3,
                                          bias=False)
            self.layer5 = torch.nn.LayerNorm(normalized_shape=[dim_1, dim_2, dim_3])

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(dim_1, dim_2, dim_3))
            x = self.layer4(x.flatten())
            x = self.layer5(x.reshape(dim_1, dim_2, dim_3))
            return x
        
    testtool.testKQI(TestLayerNorm(), torch.randn(dim_1, dim_2, dim_3))



def test_GroupNorm():
    class TestGroupNorm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = torch.nn.GroupNorm(num_groups=3, num_channels=6, affine=False)
            self.layer2 = torch.nn.Linear(in_features=1 * 6 * 5 * 5, out_features=1 * 6 * 5 * 5,
                                       bias=False)
            self.layer3 = torch.nn.GroupNorm(num_groups=2, num_channels=6, affine=False)
            self.layer4 = torch.nn.Linear(in_features=1 * 6 * 5 * 5, out_features=1 * 6 * 5 * 5,
                                       bias=False)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(1, 6, 5, 5))
            x = self.layer4(x.flatten())
            return x

    testtool.testKQI(TestGroupNorm(), torch.randn(1, 6, 5, 5))


def test_LocalResponseNorm():
    class TestLocalResponseNorm(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = kqinn.LocalResponseNorm(3)
            self.layer2 = kqinn.Linear(in_features=1 * 6 * 10 * 10, out_features=1 * 6 * 10 * 10,
                                       bias=False)
            self.layer3 = kqinn.LocalResponseNorm(2)
            self.layer4 = kqinn.Linear(in_features=1 * 6 * 10 * 10, out_features=1 * 6 * 10 * 10,
                                       bias=False)
            self.layer5 = kqinn.LocalResponseNorm(4)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(1, 6, 10, 10))
            x = self.layer4(x.flatten())
            x = self.layer5(x.reshape(1, 6, 10, 10))
            return x

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.layer1.KQIforward(x)
            x = self.layer2.KQIforward(x.flatten())
            x = self.layer3.KQIforward(x.reshape(1, 6, 10, 10))
            x = self.layer4.KQIforward(x.flatten())
            x = self.layer5.KQIforward(x.reshape(1, 6, 10, 10))
            return x

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            volume = self.layer5.KQIbackward(volume)
            volume = self.layer4.KQIbackward(volume.flatten())
            volume = self.layer3.KQIbackward(volume.reshape(1, 6, 10, 10))
            volume = self.layer2.KQIbackward(volume.flatten())
            volume = self.layer1.KQIbackward(volume.reshape(1, 6, 10, 10), volume_backward)
            return volume

        def true_kqi(self):
            G = kqitool.DiGraph()

            for i, j, k in itertools.product(range(10), range(10), range(6)):
                G.add_node(f'L0_{i}-{j}-{k}', [])

            # LocalResponseNorm(3)
            for i, j, k in itertools.product(range(10), range(10), range(6)):
                preds = [f'L0_{i}-{j}-{m}' for m in range(max(0, k - 1), min(5, k + 1) + 1)]
                G.add_node(f'L1_{i}-{j}-{k}', preds)

            # Linear
            preds = [f'L1_{i}-{j}-{k}' for i, j, k in itertools.product(range(10), range(10), range(6))]
            for i, j, k in itertools.product(range(10), range(10), range(6)):
                G.add_node(f'L2_{i}-{j}-{k}', preds)

            # LocalResponseNorm(2)
            for i, j, k in itertools.product(range(10), range(10), range(6)):
                preds = [f'L2_{i}-{j}-{m}' for m in range(max(0, k - 1), min(5, k + 1) + 1)]
                G.add_node(f'L3_{i}-{j}-{k}', preds)

            # Linear
            preds = [f'L3_{i}-{j}-{k}' for i, j, k in itertools.product(range(10), range(10), range(6))]
            for i, j, k in itertools.product(range(10), range(10), range(6)):
                G.add_node(f'L4_{i}-{j}-{k}', preds)

            # LocalResponseNorm(4)
            for i, j, k in itertools.product(range(10), range(10), range(6)):
                preds = [f'L4_{i}-{j}-{m}' for m in range(max(0, k - 2), min(5, k + 2) + 1)]
                G.add_node(f'L5_{i}-{j}-{k}', preds)

            return sum(map(lambda m: G.kqi(m), G.nodes()))

    kqi = TestLocalResponseNorm().KQI(torch.randn(1, 6, 10, 10))
    true = TestLocalResponseNorm().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001
        


if __name__ == '__main__':
    test_LayerNorm()
    test_GroupNorm()
    test_LocalResponseNorm()
