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


def test_GroupNorm():
    class TestGroupNorm(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = kqinn.GroupNorm(num_groups=3, num_channels=6, affine = False)
            self.layer2 = kqinn.Linear(in_features=1*6*10*10, out_features=1*6*10*10,
                                       bias=False)
            self.layer3 = kqinn.GroupNorm(num_groups=2, num_channels=6, affine = False)
            self.layer4 = kqinn.Linear(in_features=1*6*10*10, out_features=1*6*10*10,
                                       bias=False)
            self.layer5 = kqinn.GroupNorm(num_groups=1, num_channels=6, affine = False)

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

            # Separate 6 channels into 3 groups
            preds1 = [f'L0_{i}-{j}-{k}' for i, j, k in itertools.product(range(10), range(10), range(2))]
            preds2 = [f'L0_{i}-{j}-{k}' for i, j, k in itertools.product(range(10), range(10), range(2,4))]
            preds3 = [f'L0_{i}-{j}-{k}' for i, j, k in itertools.product(range(10), range(10), range(4,6))]
            for i, j in itertools.product(range(10), range(10)):
                for k in range(2):
                    G.add_node(f'L1_{i}-{j}-{k}', preds1)
                for k in range(2,4):
                    G.add_node(f'L1_{i}-{j}-{k}', preds2)
                for k in range(4,6):
                    G.add_node(f'L1_{i}-{j}-{k}', preds3)

            # Linear
            preds = [f'L1_{i}-{j}-{k}' for i, j, k in itertools.product(range(10), range(10), range(6))]
            for i, j, k in itertools.product(range(10), range(10), range(6)):
                G.add_node(f'L2_{i}-{j}-{k}', preds)

            # Separate 6 channels into 2 groups
            preds1 = [f'L2_{i}-{j}-{k}' for i, j, k in itertools.product(range(10), range(10), range(3))]
            preds2 = [f'L2_{i}-{j}-{k}' for i, j, k in itertools.product(range(10), range(10), range(3,6))]
            for i, j in itertools.product(range(10), range(10)):
                for k in range(3):
                    G.add_node(f'L3_{i}-{j}-{k}', preds1)
                for k in range(3,6):
                    G.add_node(f'L3_{i}-{j}-{k}', preds2)

            # Linear
            preds = [f'L3_{i}-{j}-{k}' for i, j, k in itertools.product(range(10), range(10), range(6))]
            for i, j, k in itertools.product(range(10), range(10), range(6)):
                G.add_node(f'L4_{i}-{j}-{k}', preds)

            # Put all 6 channels into a single group
            preds = [f'L4_{i}-{j}-{k}' for i, j, k in itertools.product(range(10), range(10), range(6))]
            for i, j, k in itertools.product(range(10), range(10), range(6)):
                G.add_node(f'L5_{i}-{j}-{k}', preds)

            return sum(map(lambda m: G.kqi(m), G.nodes()))

    kqi = TestGroupNorm().KQI(torch.randn(1, 6, 10, 10))
    true = TestGroupNorm().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_LocalResponseNorm():
    class TestLocalResponseNorm(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = kqinn.LocalResponseNorm(3)
            self.layer2 = kqinn.Linear(in_features=1*6*10*10, out_features=1*6*10*10,
                                       bias=False)
            self.layer3 = kqinn.LocalResponseNorm(2)
            self.layer4 = kqinn.Linear(in_features=1*6*10*10, out_features=1*6*10*10,
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

            #LocalResponseNorm(3)
            for i, j, k in itertools.product(range(10), range(10), range(6)):
                preds = [f'L0_{i}-{j}-{m}' for m in range(max(0,k-1),min(5,k+1)+1)]
                G.add_node(f'L1_{i}-{j}-{k}', preds)
            
            # Linear
            preds = [f'L1_{i}-{j}-{k}' for i, j, k in itertools.product(range(10), range(10), range(6))]
            for i, j, k in itertools.product(range(10), range(10), range(6)):
                G.add_node(f'L2_{i}-{j}-{k}', preds)


            #LocalResponseNorm(2)
            for i, j, k in itertools.product(range(10), range(10), range(6)):
                preds = [f'L2_{i}-{j}-{m}' for m in range(max(0,k-1),min(5,k+1)+1)]
                G.add_node(f'L3_{i}-{j}-{k}', preds)
            
            # Linear
            preds = [f'L3_{i}-{j}-{k}' for i, j, k in itertools.product(range(10), range(10), range(6))]
            for i, j, k in itertools.product(range(10), range(10), range(6)):
                G.add_node(f'L4_{i}-{j}-{k}', preds)


            #LocalResponseNorm(4)
            for i, j, k in itertools.product(range(10), range(10), range(6)):
                preds = [f'L4_{i}-{j}-{m}' for m in range(max(0,k-2),min(5,k+2)+1)]
                G.add_node(f'L5_{i}-{j}-{k}', preds)


            kqi = sum(map(lambda k: G.kqi(k) if "L5_" in k else 0, G.nodes()))
            logging.debug(f'L5: KQI={kqi}, node={len([k for k in G.nodes() if "L5_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L5_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L4_" in k else 0, G.nodes()))
            logging.debug(f'L4: KQI={kqi}, node={len([k for k in G.nodes() if "L4_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L4_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L3_" in k else 0, G.nodes()))
            logging.debug(f'L3: KQI={kqi}, node={len([k for k in G.nodes() if "L3_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L3_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L2_" in k else 0, G.nodes()))
            logging.debug(f'L2: KQI={kqi}, node={len([k for k in G.nodes() if "L2_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L2_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L1_" in k else 0, G.nodes()))
            logging.debug(f'L1: KQI={kqi}, node={len([k for k in G.nodes() if "L1_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L1_" in k])}')
            logging.debug(f'Total volume = {G.graph_volume()}')

            return sum(map(lambda m: G.kqi(m), G.nodes()))

    kqi = TestLocalResponseNorm().KQI(torch.randn(1, 6, 10, 10))
    true = TestLocalResponseNorm().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test_LayerNorm()
    test_GroupNorm()
    test_LocalResponseNorm()
