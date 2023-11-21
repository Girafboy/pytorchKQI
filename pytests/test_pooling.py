import torch
import kqinn
import kqitool
import itertools
import logging


def test_AvgPool1d():
    class TestAvgPool1d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28
                kqinn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26
                kqinn.AvgPool1d(kernel_size=2, stride=2, padding=1)
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

    kqi = TestAvgPool1d().KQI(torch.randn(1, 28))
    true = TestAvgPool1d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_AvgPool2d():
    class TestAvgPool2d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26x26
                kqinn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            )
            self.layers2 = kqinn.Sequential(
                # 3x13x13
                kqinn.Linear(in_features=3 * 13 * 13, out_features=100, bias=False),
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
            volume = volume.reshape(3, 13, 13)
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
            for i, j in itertools.product(range(13), range(13)):
                for k3 in [1, 2, 3]:
                    preds = [f'L2_{k1}-{k2}_{k3}' for k1, k2 in itertools.product([i*2, i*2+1], [j*2, j*2+1])]
                    G.add_node(f'L3_{i}-{j}_{k3}', preds)

            for i in range(100):
                preds = [f'L3_{k1}-{k2}_{k3}' for k1, k2 in itertools.product(range(13), range(13)) for k3 in [1, 2, 3]]
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

    kqi = TestAvgPool2d().KQI(torch.randn(1, 28, 28))
    true = TestAvgPool2d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_AvgPool3d():
    class TestAvgPool3d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x8x8x8
                kqinn.Conv3d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x6x6x6
                kqinn.AvgPool3d(kernel_size=2, stride=2, padding=1)
            )
            self.layers2 = kqinn.Sequential(
                # 3x4x4x4
                kqinn.Linear(in_features=3 * 4 * 4 * 4, out_features=100, bias=False),
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
            volume = volume.reshape(3, 4, 4, 4)
            volume = self.layers1.KQIbackward(volume, volume_backward)

            return volume

        def true_kqi(self):
            G = kqitool.DiGraph()
            for i, j, k in itertools.product(range(8), range(8), range(8)):
                G.add_node(f'L1_{i}-{j}-{k}', [])
            for i, j, k in itertools.product(range(6), range(6), range(6)):
                preds = [f'L1_{k1}-{k2}-{k3}' for k1, k2, k3 in itertools.product([i, i+1, i+2], [j, j+1, j+2], [k, k+1, k+2])]
                G.add_node(f'L2_{i}-{j}-{k}_1', preds)
                G.add_node(f'L2_{i}-{j}-{k}_2', preds)
                G.add_node(f'L2_{i}-{j}-{k}_3', preds)
            for i, j, k in itertools.product(range(4), range(4), range(4)):
                for k4 in [1, 2, 3]:
                    preds = [f'L2_{k1}-{k2}-{k3}_{k4}' for k1, k2, k3 in itertools.product([i*2-1, i*2], [j*2-1, j*2], [k*2-1, k*2]) if 0 <= k1 < 6 and 0 <= k2 < 6 and 0 <= k3 < 6]
                    G.add_node(f'L3_{i}-{j}-{k}_{k4}', preds)

            for i in range(100):
                preds = [f'L3_{k1}-{k2}-{k3}_{k4}' for k1, k2, k3 in itertools.product(range(4), range(4), range(4)) for k4 in [1, 2, 3]]
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

    kqi = TestAvgPool3d().KQI(torch.randn(1, 8, 8, 8))
    true = TestAvgPool3d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


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
                    preds = [f'L2_{k1}-{k2}_{k3}' for k1, k2 in itertools.product([i*2-1, i*2], [j*2-1, j*2]) if 0 <= k1 < 26 and 0 <= k2 < 26]
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
                # 1x8x8x8
                kqinn.Conv3d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x6x6x6
                kqinn.MaxPool3d(kernel_size=2, stride=2, padding=1, dilation=1)
            )
            self.layers2 = kqinn.Sequential(
                # 3x4x4x4
                kqinn.Linear(in_features=3 * 4 * 4 * 4, out_features=100, bias=False),
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
            volume = volume.reshape(3, 4, 4, 4)
            volume = self.layers1.KQIbackward(volume, volume_backward)

            return volume

        def true_kqi(self):
            G = kqitool.DiGraph()
            for i, j, k in itertools.product(range(8), range(8), range(8)):
                G.add_node(f'L1_{i}-{j}-{k}', [])
            for i, j, k in itertools.product(range(6), range(6), range(6)):
                preds = [f'L1_{k1}-{k2}-{k3}' for k1, k2, k3 in itertools.product([i, i+1, i+2], [j, j+1, j+2], [k, k+1, k+2])]
                G.add_node(f'L2_{i}-{j}-{k}_1', preds)
                G.add_node(f'L2_{i}-{j}-{k}_2', preds)
                G.add_node(f'L2_{i}-{j}-{k}_3', preds)
            for i, j, k in itertools.product(range(4), range(4), range(4)):
                for k4 in [1, 2, 3]:
                    preds = [f'L2_{k1}-{k2}-{k3}_{k4}' for k1, k2, k3 in itertools.product([i*2-1, i*2], [j*2-1, j*2], [k*2-1, k*2]) if 0 <= k1 < 6 and 0 <= k2 < 6 and 0 <= k3 < 6]
                    G.add_node(f'L3_{i}-{j}-{k}_{k4}', preds)

            for i in range(100):
                preds = [f'L3_{k1}-{k2}-{k3}_{k4}' for k1, k2, k3 in itertools.product(range(4), range(4), range(4)) for k4 in [1, 2, 3]]
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

    kqi = TestMaxPool3d().KQI(torch.randn(1, 8, 8, 8))
    true = TestMaxPool3d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_AdaptiveAvgPool1d():
    class TestAdaptiveAvgPool1d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28
                kqinn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26
                kqinn.AdaptiveAvgPool1d(output_size=13)
            )
            self.layers2 = kqinn.Sequential(
                # 3x13
                kqinn.Linear(in_features=3 * 13, out_features=100, bias=False),
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
            volume = volume.reshape(3, 13)
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
            for i in range(13):
                for k2 in [1, 2, 3]:
                    preds = [f'L2_{k1}_{k2}' for k1 in [i*2, i*2+1]]
                    G.add_node(f'L3_{i}_{k2}', preds)

            for i in range(100):
                preds = [f'L3_{k1}_{k2}' for k1 in range(13) for k2 in [1, 2, 3]]
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

    kqi = TestAdaptiveAvgPool1d().KQI(torch.randn(1, 28))
    true = TestAdaptiveAvgPool1d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_AdaptiveAvgPool2d():
    class TestAdaptiveAvgPool2d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26x26
                kqinn.AdaptiveAvgPool2d(output_size=13)
            )
            self.layers2 = kqinn.Sequential(
                # 3x13x13
                kqinn.Linear(in_features=3 * 13 * 13, out_features=100, bias=False),
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
            volume = volume.reshape(3, 13, 13)
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
            for i, j in itertools.product(range(13), range(13)):
                for k3 in [1, 2, 3]:
                    preds = [f'L2_{k1}-{k2}_{k3}' for k1, k2 in itertools.product([i*2, i*2+1], [j*2, j*2+1])]
                    G.add_node(f'L3_{i}-{j}_{k3}', preds)

            for i in range(100):
                preds = [f'L3_{k1}-{k2}_{k3}' for k1, k2 in itertools.product(range(13), range(13)) for k3 in [1, 2, 3]]
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

    kqi = TestAdaptiveAvgPool2d().KQI(torch.randn(1, 28, 28))
    true = TestAdaptiveAvgPool2d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_AdaptiveAvgPool3d():
    class TestAdaptiveAvgPool3d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x8x8x8
                kqinn.Conv3d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x6x6x6
                kqinn.AdaptiveAvgPool3d(output_size=3)
            )
            self.layers2 = kqinn.Sequential(
                # 3x3x3x3
                kqinn.Linear(in_features=3 * 3 * 3 * 3, out_features=100, bias=False),
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
            volume = volume.reshape(3, 3, 3, 3)
            volume = self.layers1.KQIbackward(volume, volume_backward)

            return volume

        def true_kqi(self):
            G = kqitool.DiGraph()
            for i, j, k in itertools.product(range(8), range(8), range(8)):
                G.add_node(f'L1_{i}-{j}-{k}', [])
            for i, j, k in itertools.product(range(6), range(6), range(6)):
                preds = [f'L1_{k1}-{k2}-{k3}' for k1, k2, k3 in itertools.product([i, i+1, i+2], [j, j+1, j+2], [k, k+1, k+2])]
                G.add_node(f'L2_{i}-{j}-{k}_1', preds)
                G.add_node(f'L2_{i}-{j}-{k}_2', preds)
                G.add_node(f'L2_{i}-{j}-{k}_3', preds)
            for i, j, k in itertools.product(range(3), range(3), range(3)):
                for k4 in [1, 2, 3]:
                    preds = [f'L2_{k1}-{k2}-{k3}_{k4}' for k1, k2, k3 in itertools.product([i*2, i*2+1], [j*2, j*2+1], [k*2, k*2+1])]
                    G.add_node(f'L3_{i}-{j}-{k}_{k4}', preds)

            for i in range(100):
                preds = [f'L3_{k1}-{k2}-{k3}_{k4}' for k1, k2, k3 in itertools.product(range(3), range(3), range(3)) for k4 in [1, 2, 3]]
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

    kqi = TestAdaptiveAvgPool3d().KQI(torch.randn(1, 8, 8, 8))
    true = TestAdaptiveAvgPool3d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_AdaptiveMaxPool1d():
    class TestAdaptiveMaxPool1d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28
                kqinn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26
                kqinn.AdaptiveMaxPool1d(output_size=13)
            )
            self.layers2 = kqinn.Sequential(
                # 3x13
                kqinn.Linear(in_features=3 * 13, out_features=100, bias=False),
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
            volume = volume.reshape(3, 13)
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
            for i in range(13):
                for k2 in [1, 2, 3]:
                    preds = [f'L2_{k1}_{k2}' for k1 in [i*2, i*2+1]]
                    G.add_node(f'L3_{i}_{k2}', preds)

            for i in range(100):
                preds = [f'L3_{k1}_{k2}' for k1 in range(13) for k2 in [1, 2, 3]]
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

    kqi = TestAdaptiveMaxPool1d().KQI(torch.randn(1, 28))
    true = TestAdaptiveMaxPool1d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_AdaptiveMaxPool2d():
    class TestAdaptiveMaxPool2d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26x26
                kqinn.AdaptiveMaxPool2d(output_size=13)
            )
            self.layers2 = kqinn.Sequential(
                # 3x13x13
                kqinn.Linear(in_features=3 * 13 * 13, out_features=100, bias=False),
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
            volume = volume.reshape(3, 13, 13)
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
            for i, j in itertools.product(range(13), range(13)):
                for k3 in [1, 2, 3]:
                    preds = [f'L2_{k1}-{k2}_{k3}' for k1, k2 in itertools.product([i*2, i*2+1], [j*2, j*2+1])]
                    G.add_node(f'L3_{i}-{j}_{k3}', preds)

            for i in range(100):
                preds = [f'L3_{k1}-{k2}_{k3}' for k1, k2 in itertools.product(range(13), range(13)) for k3 in [1, 2, 3]]
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

    kqi = TestAdaptiveMaxPool2d().KQI(torch.randn(1, 28, 28))
    true = TestAdaptiveMaxPool2d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_AdaptiveMaxPool3d():
    class TestAdaptiveMaxPool3d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x8x8x8
                kqinn.Conv3d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x6x6x6
                kqinn.AdaptiveMaxPool3d(output_size=3)
            )
            self.layers2 = kqinn.Sequential(
                # 3x3x3x3
                kqinn.Linear(in_features=3 * 3 * 3 * 3, out_features=100, bias=False),
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
            volume = volume.reshape(3, 3, 3, 3)
            volume = self.layers1.KQIbackward(volume, volume_backward)

            return volume

        def true_kqi(self):
            G = kqitool.DiGraph()
            for i, j, k in itertools.product(range(8), range(8), range(8)):
                G.add_node(f'L1_{i}-{j}-{k}', [])
            for i, j, k in itertools.product(range(6), range(6), range(6)):
                preds = [f'L1_{k1}-{k2}-{k3}' for k1, k2, k3 in itertools.product([i, i+1, i+2], [j, j+1, j+2], [k, k+1, k+2])]
                G.add_node(f'L2_{i}-{j}-{k}_1', preds)
                G.add_node(f'L2_{i}-{j}-{k}_2', preds)
                G.add_node(f'L2_{i}-{j}-{k}_3', preds)
            for i, j, k in itertools.product(range(3), range(3), range(3)):
                for k4 in [1, 2, 3]:
                    preds = [f'L2_{k1}-{k2}-{k3}_{k4}' for k1, k2, k3 in itertools.product([i*2, i*2+1], [j*2, j*2+1], [k*2, k*2+1])]
                    G.add_node(f'L3_{i}-{j}-{k}_{k4}', preds)

            for i in range(100):
                preds = [f'L3_{k1}-{k2}-{k3}_{k4}' for k1, k2, k3 in itertools.product(range(3), range(3), range(3)) for k4 in [1, 2, 3]]
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

    kqi = TestAdaptiveMaxPool3d().KQI(torch.randn(1, 8, 8, 8))
    true = TestAdaptiveMaxPool3d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_LPPool1d():
    class TestLPPool1d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28
                kqinn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26
                kqinn.LPPool1d(2.3, kernel_size=2, stride=2)
            )
            self.layers2 = kqinn.Sequential(
                # 3x14
                kqinn.Linear(in_features=3 * 13, out_features=100, bias=False),
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
            volume = volume.reshape(3, 13)
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
            for i in range(13):
                for k2 in [1, 2, 3]:
                    preds = [f'L2_{k1}_{k2}' for k1 in [i*2, i*2+1]]
                    G.add_node(f'L3_{i}_{k2}', preds)

            for i in range(100):
                preds = [f'L3_{k1}_{k2}' for k1 in range(13) for k2 in [1, 2, 3]]
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

    kqi = TestLPPool1d().KQI(torch.randn(1, 28))
    true = TestLPPool1d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_LPPool2d():
    class TestLPPool2d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26x26
                kqinn.LPPool2d(1.2, kernel_size=2, stride=2)
            )
            self.layers2 = kqinn.Sequential(
                # 3x13x13
                kqinn.Linear(in_features=3 * 13 * 13, out_features=100, bias=False),
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
            volume = volume.reshape(3, 13, 13)
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
            for i, j in itertools.product(range(13), range(13)):
                for k3 in [1, 2, 3]:
                    preds = [f'L2_{k1}-{k2}_{k3}' for k1, k2 in itertools.product([i*2, i*2+1], [j*2, j*2+1])]
                    G.add_node(f'L3_{i}-{j}_{k3}', preds)

            for i in range(100):
                preds = [f'L3_{k1}-{k2}_{k3}' for k1, k2 in itertools.product(range(13), range(13)) for k3 in [1, 2, 3]]
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

    kqi = TestLPPool2d().KQI(torch.randn(1, 28, 28))
    true = TestLPPool2d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test_AvgPool1d()
    test_AvgPool2d()
    test_AvgPool3d()
    test_MaxPool1d()
    test_MaxPool2d()
    test_MaxPool3d()
    test_AdaptiveAvgPool1d()
    test_AdaptiveAvgPool2d()
    test_AdaptiveAvgPool3d()
    test_AdaptiveMaxPool1d()
    test_AdaptiveMaxPool2d()
    test_AdaptiveMaxPool3d()
    test_LPPool1d()
    test_LPPool2d()
