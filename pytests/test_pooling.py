import torch
import kqinn
import kqitool
import itertools
import logging

def test_MaxPool1d():
    class TestMaxPool1d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28
                kqinn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26
                kqinn.MaxPool1d(kernel_size=2, stride=2, padding=1, dilation=1)
            )
            self.layers2 = kqinn.Sequential(
                # 3x14
                kqinn.Linear(in_features=3 * 14, out_features=100, bias=False),
                kqinn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.layers1.KQIforward(x)
            x = x.flatten()
            x = self.layers2.KQIforward(x)
            return x

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            volume = self.layers2.KQIbackward(volume)
            volume = volume.reshape(3, 14)
            volume = self.layers1.KQIbackward(volume, volume_backward)

            return volume

        def true_kqi(self):
            G = kqitool.DiGraph()
            for i in range(28):
                G.add_node(f'L1_{i}', [])
            for i in range(26):
                preds = [f'L1_{k1}' for k1 in [i, i+1, i+2]]
                G.add_node(f'L2_{i}_1', preds)
                G.add_node(f'L2_{i}_2', preds)
                G.add_node(f'L2_{i}_3', preds)
            for i in range(14):
                for k2 in [1, 2, 3]:
                    preds = [f'L2_{k1}_{k2}' for k1 in [i*2-1, i*2] if 0 <= k1 < 26]
                    G.add_node(f'L3_{i}_{k2}', preds)

            for i in range(100):
                preds = [f'L3_{k1}_{k2}' for k1 in range(14) for k2 in [1, 2, 3]]
                G.add_node(f'L4_{i}', preds)
            for i in range(10):
                preds = [f'L4_{k}' for k in range(100)]
                G.add_node(f'L5_{i}', preds)

            kqi = sum(map(lambda k: G.kqi(k) if "L5_" in k else 0, G.nodes()))
            logging.debug(
                f'L5: KQI={kqi}, node={len([k for k in G.nodes() if "L5_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L5_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L4_" in k else 0, G.nodes()))
            logging.debug(
                f'L4: KQI={kqi}, node={len([k for k in G.nodes() if "L4_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L4_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L3_" in k else 0, G.nodes()))
            logging.debug(
                f'L3: KQI={kqi}, node={len([k for k in G.nodes() if "L3_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L3_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L2_" in k else 0, G.nodes()))
            logging.debug(
                f'L2: KQI={kqi}, node={len([k for k in G.nodes() if "L2_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L2_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L1_" in k else 0, G.nodes()))
            logging.debug(
                f'L1: KQI={kqi}, node={len([k for k in G.nodes() if "L1_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L1_" in k])}')
            logging.debug(f'Total volume = {G.graph_volume()}')

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestMaxPool1d().KQI(torch.randn(1, 28))
    true = TestMaxPool1d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001

def test_MaxPool2d():
    class TestMaxPool2d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26x26
                kqinn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1)
            )
            self.layers2 = kqinn.Sequential(
                # 3x14x14
                kqinn.Linear(in_features=3 * 14 * 14, out_features=100, bias=False),
                kqinn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.layers1.KQIforward(x)
            x = x.flatten()
            x = self.layers2.KQIforward(x)
            return x

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            volume = self.layers2.KQIbackward(volume)
            volume = volume.reshape(3, 14, 14)
            volume = self.layers1.KQIbackward(volume, volume_backward)

            return volume

        def true_kqi(self):
            G = kqitool.DiGraph()
            for i, j in itertools.product(range(28), range(28)):
                G.add_node(f'L1_{i}-{j}', [])
            for i, j in itertools.product(range(26), range(26)):
                preds = [f'L1_{k1}-{k2}' for k1, k2 in itertools.product([i, i+1, i+2], [j, j+1, j+2])]
                G.add_node(f'L2_{i}-{j}_1', preds)
                G.add_node(f'L2_{i}-{j}_2', preds)
                G.add_node(f'L2_{i}-{j}_3', preds)
            for i, j in itertools.product(range(14), range(14)):
                for k3 in [1, 2, 3]:
                    preds = [f'L2_{k1}-{k2}_{k3}' for k1, k2 in itertools.product([i*2-1, i*2], [j*2-1, j*2])
                             if 0 <= k1 < 26 and 0 <= k2 < 26]
                    G.add_node(f'L3_{i}-{j}_{k3}', preds)

            for i in range(100):
                preds = [f'L3_{k1}-{k2}_{k3}' for k1, k2 in itertools.product(range(14), range(14)) for k3 in [1, 2, 3]]
                G.add_node(f'L4_{i}', preds)
            for i in range(10):
                preds = [f'L4_{k}' for k in range(100)]
                G.add_node(f'L5_{i}', preds)

            kqi = sum(map(lambda k: G.kqi(k) if "L5_" in k else 0, G.nodes()))
            logging.debug(
                f'L5: KQI={kqi}, node={len([k for k in G.nodes() if "L5_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L5_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L4_" in k else 0, G.nodes()))
            logging.debug(
                f'L4: KQI={kqi}, node={len([k for k in G.nodes() if "L4_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L4_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L3_" in k else 0, G.nodes()))
            logging.debug(
                f'L3: KQI={kqi}, node={len([k for k in G.nodes() if "L3_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L3_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L2_" in k else 0, G.nodes()))
            logging.debug(
                f'L2: KQI={kqi}, node={len([k for k in G.nodes() if "L2_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L2_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L1_" in k else 0, G.nodes()))
            logging.debug(
                f'L1: KQI={kqi}, node={len([k for k in G.nodes() if "L1_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L1_" in k])}')
            logging.debug(f'Total volume = {G.graph_volume()}')

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestMaxPool2d().KQI(torch.randn(1, 28, 28))
    true = TestMaxPool2d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001

def test_MaxPool3d():
    class TestMaxPool3d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28x28
                kqinn.Conv3d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26x26x26
                kqinn.MaxPool3d(kernel_size=2, stride=2, padding=1, dilation=1)
            )
            self.layers2 = kqinn.Sequential(
                # 3x14x14x14
                kqinn.Linear(in_features=3 * 14 * 14 * 14, out_features=100, bias=False),
                kqinn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.layers1.KQIforward(x)
            x = x.flatten()
            x = self.layers2.KQIforward(x)
            return x

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            volume = self.layers2.KQIbackward(volume)
            volume = volume.reshape(3, 14, 14, 14)
            volume = self.layers1.KQIbackward(volume, volume_backward)

            return volume

        def true_kqi(self):
            G = kqitool.DiGraph()
            for i, j, k in itertools.product(range(28), range(28), range(28)):
                G.add_node(f'L1_{i}-{j}-{k}', [])
            for i, j, k in itertools.product(range(26), range(26), range(26)):
                preds = [f'L1_{k1}-{k2}-{k3}' for k1, k2, k3 in itertools.product([i, i+1, i+2], [j, j+1, j+2], [k, k+1, k+2])]
                G.add_node(f'L2_{i}-{j}-{k}_1', preds)
                G.add_node(f'L2_{i}-{j}-{k}_2', preds)
                G.add_node(f'L2_{i}-{j}-{k}_3', preds)
            for i, j, k in itertools.product(range(14), range(14), range(14)):
                for k4 in [1, 2, 3]:
                    preds = [f'L2_{k1}-{k2}-{k3}_{k4}' for k1, k2, k3 in itertools.product([i*2-1, i*2], [j*2-1, j*2], [k*2-1, k*2])
                             if 0 <= k1 < 26 and 0 <= k2 < 26 and 0 <= k3 < 26]
                    G.add_node(f'L3_{i}-{j}-{k}_{k4}', preds)

            for i in range(100):
                preds = [f'L3_{k1}-{k2}-{k3}_{k4}' for k1, k2, k3 in itertools.product(range(14), range(14), range(14)) for k4 in [1, 2, 3]]
                G.add_node(f'L4_{i}', preds)
            for i in range(10):
                preds = [f'L4_{k}' for k in range(100)]
                G.add_node(f'L5_{i}', preds)

            kqi = sum(map(lambda k: G.kqi(k) if "L5_" in k else 0, G.nodes()))
            logging.debug(
                f'L5: KQI={kqi}, node={len([k for k in G.nodes() if "L5_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L5_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L4_" in k else 0, G.nodes()))
            logging.debug(
                f'L4: KQI={kqi}, node={len([k for k in G.nodes() if "L4_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L4_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L3_" in k else 0, G.nodes()))
            logging.debug(
                f'L3: KQI={kqi}, node={len([k for k in G.nodes() if "L3_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L3_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L2_" in k else 0, G.nodes()))
            logging.debug(
                f'L2: KQI={kqi}, node={len([k for k in G.nodes() if "L2_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L2_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L1_" in k else 0, G.nodes()))
            logging.debug(
                f'L1: KQI={kqi}, node={len([k for k in G.nodes() if "L1_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L1_" in k])}')
            logging.debug(f'Total volume = {G.graph_volume()}')

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestMaxPool3d().KQI(torch.randn(1, 28, 28, 28))
    true = TestMaxPool3d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test_MaxPool1d()
    test_MaxPool2d()
    test_MaxPool3d()
