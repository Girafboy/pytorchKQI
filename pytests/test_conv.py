import torch
import kqinn
import kqitool
import itertools
import logging
import testtool


def test_Conv1d():
    class TestConv1d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 3x28
                torch.nn.Conv1d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                # 2x28
                torch.nn.Conv1d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8
                torch.nn.Linear(in_features=3 * 8, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestConv1d(), torch.randn(3, 28))


def test_Conv2d():
    class TestConv2d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                # 2x28x28
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
            )
            self.layers2 = kqinn.Sequential(
                # 3x8x8
                kqinn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
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
            volume = volume.reshape(3, 8, 8)
            volume = self.layers1.KQIbackward(volume, volume_backward)

            return volume

        def true_kqi(self):
            G = kqitool.DiGraph()
            for i, j in itertools.product(range(28), range(28)):
                G.add_node(f'L1_{i}-{j}_1', [])
                G.add_node(f'L1_{i}-{j}_2', [])
                G.add_node(f'L1_{i}-{j}_3', [])

            for i, j in itertools.product(range(28), range(28)):
                preds = [f'L1_{k1}-{k2}_{k3}' for k1, k2 in itertools.product([i - 1, i, i + 1], [j - 1, j, j + 1]) if k1 >= 0 and k1 < 28 and k2 >= 0 and k2 < 28 for k3 in [1, 2, 3]]
                G.add_node(f'L2_{i}-{j}_1', preds)
                G.add_node(f'L2_{i}-{j}_2', preds)

            for i, j in itertools.product(range(8), range(8)):
                preds = [f'L2_{k1}-{k2}_{k3}' for k1, k2 in itertools.product([i * 3, i * 3 + 2, i * 3 + 4], [j * 3, j * 3 + 2, j * 3 + 4]) for k3 in [1, 2]]
                G.add_node(f'L3_{i}-{j}_1', preds)
                G.add_node(f'L3_{i}-{j}_2', preds)
                G.add_node(f'L3_{i}-{j}_3', preds)

            for i in range(100):
                preds = [f'L3_{k1}-{k2}_{k3}' for k1, k2 in itertools.product(range(8), range(8)) for k3 in [1, 2, 3]]
                G.add_node(f'L4_{i}', preds)

            for i in range(10):
                preds = [f'L4_{k}' for k in range(100)]
                G.add_node(f'L5_{i}', preds)

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

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestConv2d().KQI(torch.randn(3, 28, 28))
    true = TestConv2d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_Conv3d():
    class TestConv3d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x9x9x9
                kqinn.Conv3d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                # 2x9x9x9
                kqinn.Conv3d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
            )
            self.layers2 = kqinn.Sequential(
                # 3x2x2x2
                kqinn.Linear(in_features=3 * 2 * 2 * 2, out_features=100, bias=False),
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
            volume = volume.reshape(3, 2, 2, 2)
            volume = self.layers1.KQIbackward(volume, volume_backward)

            return volume

        def true_kqi(self):
            G = kqitool.DiGraph()
            for i, j, k in itertools.product(range(9), range(9), range(9)):
                G.add_node(f'L1_{i}-{j}-{k}_1', [])
                G.add_node(f'L1_{i}-{j}-{k}_2', [])
                G.add_node(f'L1_{i}-{j}-{k}_3', [])

            for i, j, k in itertools.product(range(9), range(9), range(9)):
                preds = [f'L1_{m1}-{m2}-{m3}_{m4}' for m1, m2, m3 in itertools.product([i - 1, i, i + 1], [j - 1, j, j + 1], [k - 1, k, k + 1]) if m1 >= 0 and m1 < 9 and m2 >= 0 and m2 < 9 and m3 >= 0 and m3 < 9 for m4 in [1, 2, 3]]
                G.add_node(f'L2_{i}-{j}-{k}_1', preds)
                G.add_node(f'L2_{i}-{j}-{k}_2', preds)

            for i, j, k in itertools.product(range(2), range(2), range(2)):
                preds = [f'L2_{m1}-{m2}-{m3}_{m4}' for m1, m2, m3 in itertools.product([i * 3, i * 3 + 2, i * 3 + 4], [j * 3, j * 3 + 2, j * 3 + 4], [k * 3, k * 3 + 2, k * 3 + 4]) for m4 in [1, 2]]
                G.add_node(f'L3_{i}-{j}-{k}_1', preds)
                G.add_node(f'L3_{i}-{j}-{k}_2', preds)
                G.add_node(f'L3_{i}-{j}-{k}_3', preds)

            for i in range(100):
                preds = [f'L3_{m1}-{m2}-{m3}_{m4}' for m1, m2, m3 in itertools.product(range(2), range(2), range(2)) for m4 in [1, 2, 3]]
                G.add_node(f'L4_{i}', preds)

            for i in range(10):
                preds = [f'L4_{m}' for m in range(100)]
                G.add_node(f'L5_{i}', preds)

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

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestConv3d().KQI(torch.randn(3, 9, 9, 9))
    true = TestConv3d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test_Conv1d()
    test_Conv2d()
    test_Conv3d()
