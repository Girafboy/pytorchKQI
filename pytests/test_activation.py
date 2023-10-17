import torch
import kqinn
import kqitool
import itertools
import logging


def test_ReLU():
    class TestReLU(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.ReLU(inplace=True),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.ReLU(inplace=True),
            )
            self.layers2 = kqinn.Sequential(
                # 3*8*8
                kqinn.Linear(in_features=3*8*8, out_features=100, bias=False),
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
                G.add_node(f'L1_{i}-{j}', [])
            for i, j in itertools.product(range(26), range(26)):
                preds = [f'L1_{k1}-{k2}' for k1, k2 in itertools.product([i, i+1, i+2], [j, j+1, j+2])]
                G.add_node(f'L2_{i}-{j}_1', preds)
                G.add_node(f'L2_{i}-{j}_2', preds)
            for i, j in itertools.product(range(26), range(26)):
                G.add_node(f'L3_{i}-{j}_1', [f'L2_{i}-{j}_1'])
                G.add_node(f'L3_{i}-{j}_2', [f'L2_{i}-{j}_2'])
            for i, j in itertools.product(range(8), range(8)):
                preds = [f'L3_{k1}-{k2}_{k3}' for k1, k2 in itertools.product([i*3, i*3+2, i*3+4], [j*3, j*3+2, j*3+4]) for k3 in [1, 2]]
                G.add_node(f'L4_{i}-{j}_1', preds)
                G.add_node(f'L4_{i}-{j}_2', preds)
                G.add_node(f'L4_{i}-{j}_3', preds)
            for i, j in itertools.product(range(8), range(8)):
                G.add_node(f'L5_{i}-{j}_1', [f'L4_{i}-{j}_1'])
                G.add_node(f'L5_{i}-{j}_2', [f'L4_{i}-{j}_2'])
                G.add_node(f'L5_{i}-{j}_3', [f'L4_{i}-{j}_3'])

            for i in range(100):
                preds = [f'L5_{k1}-{k2}_{k3}' for k1, k2 in itertools.product(range(8), range(8)) for k3 in [1, 2, 3]]
                G.add_node(f'L6_{i}', preds)

            for i in range(10):
                preds = [f'L6_{k}' for k in range(100)]
                G.add_node(f'L7_{i}', preds)

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestReLU().KQI(torch.randn(1, 28, 28))
    true = TestReLU().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_Tanh():
    class TestTanh(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.Tanh(),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.Tanh(),
            )
            self.layers2 = kqinn.Sequential(
                # 3*8*8
                kqinn.Linear(in_features=3*8*8, out_features=100, bias=False),
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
                G.add_node(f'L1_{i}-{j}', [])
            for i, j in itertools.product(range(26), range(26)):
                preds = [f'L1_{k1}-{k2}' for k1, k2 in itertools.product([i, i+1, i+2], [j, j+1, j+2])]
                G.add_node(f'L2_{i}-{j}_1', preds)
                G.add_node(f'L2_{i}-{j}_2', preds)
            for i, j in itertools.product(range(26), range(26)):
                G.add_node(f'L3_{i}-{j}_1', [f'L2_{i}-{j}_1'])
                G.add_node(f'L3_{i}-{j}_2', [f'L2_{i}-{j}_2'])
            for i, j in itertools.product(range(8), range(8)):
                preds = [f'L3_{k1}-{k2}_{k3}' for k1, k2 in itertools.product([i*3, i*3+2, i*3+4], [j*3, j*3+2, j*3+4]) for k3 in [1, 2]]
                G.add_node(f'L4_{i}-{j}_1', preds)
                G.add_node(f'L4_{i}-{j}_2', preds)
                G.add_node(f'L4_{i}-{j}_3', preds)
            for i, j in itertools.product(range(8), range(8)):
                G.add_node(f'L5_{i}-{j}_1', [f'L4_{i}-{j}_1'])
                G.add_node(f'L5_{i}-{j}_2', [f'L4_{i}-{j}_2'])
                G.add_node(f'L5_{i}-{j}_3', [f'L4_{i}-{j}_3'])

            for i in range(100):
                preds = [f'L5_{k1}-{k2}_{k3}' for k1, k2 in itertools.product(range(8), range(8)) for k3 in [1, 2, 3]]
                G.add_node(f'L6_{i}', preds)

            for i in range(10):
                preds = [f'L6_{k}' for k in range(100)]
                G.add_node(f'L7_{i}', preds)

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestTanh().KQI(torch.randn(1, 28, 28))
    true = TestTanh().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_Softmax():
    class TestSoftmax(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                # 2x26x26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),

            )
            self.layers2 = kqinn.Sequential(
                kqinn.Softmax(dim=1),
                kqinn.Softmax(dim=2)
            )

        def forward(self, x):
            x = self.layers1(x)
            x = self.layers2(x)

            return x

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.layers1.KQIforward(x)
            x = self.layers2.KQIforward(x)

            return x

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            volume = self.layers2.KQIbackward(volume)
            volume = self.layers1.KQIbackward(volume, volume_backward)

            return volume

        def true_kqi(self):
            G = kqitool.DiGraph()
            for i, j in itertools.product(range(28), range(28)):
                G.add_node(f'L1_{i}-{j}', [])
            for i, j in itertools.product(range(26), range(26)):
                preds = [f'L1_{k1}-{k2}' for k1, k2 in itertools.product([i, i + 1, i + 2], [j, j + 1, j + 2])]
                G.add_node(f'L2_{i}-{j}_1', preds)
                G.add_node(f'L2_{i}-{j}_2', preds)
            for i, j in itertools.product(range(8), range(8)):
                preds = [f'L2_{k1}-{k2}_{k3}' for k1, k2 in
                         itertools.product([i * 3, i * 3 + 2, i * 3 + 4], [j * 3, j * 3 + 2, j * 3 + 4]) for k3 in
                         [1, 2]]
                G.add_node(f'L3_{i}-{j}_1', preds)
                G.add_node(f'L3_{i}-{j}_2', preds)
                G.add_node(f'L3_{i}-{j}_3', preds)
            for i, j in itertools.product(range(8), range(8)):
                preds = [f'L3_{k}-{j}_1' for k in range(8)]
                G.add_node(f'L4_{i}-{j}_1', preds)
                preds1 = [f'L3_{k}-{j}_2' for k in range(8)]
                G.add_node(f'L4_{i}-{j}_2', preds1)
                preds2 = [f'L3_{k}-{j}_3' for k in range(8)]
                G.add_node(f'L4_{i}-{j}_3', preds2)
            for i, j in itertools.product(range(8), range(8)):
                preds = [f'L4_{i}-{k}_1' for k in range(8)]
                G.add_node(f'L5_{i}-{j}_1', preds)
                preds1 = [f'L4_{i}-{k}_2' for k in range(8)]
                G.add_node(f'L5_{i}-{j}_2', preds1)
                preds2 = [f'L4_{i}-{k}_3' for k in range(8)]
                G.add_node(f'L5_{i}-{j}_3', preds2)

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestSoftmax().KQI(torch.randn(1, 28, 28))
    true = TestSoftmax().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001
    print(kqi)

def test_Softmax2d():
    class TestSoftmax2d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                # 2x26x26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),

            )
            self.layers2 = kqinn.Sequential(
                kqinn.Softmax2d()

            )

        def forward(self, x):
            x = self.layers1(x)
            x = self.layers2(x)

            return x

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.layers1.KQIforward(x)
            x = self.layers2.KQIforward(x)

            return x

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            volume = self.layers2.KQIbackward(volume)
            volume = self.layers1.KQIbackward(volume, volume_backward)

            return volume

        def true_kqi(self):
            G = kqitool.DiGraph()
            for i, j in itertools.product(range(28), range(28)):
                G.add_node(f'L1_{i}-{j}', [])
            for i, j in itertools.product(range(26), range(26)):
                preds = [f'L1_{k1}-{k2}' for k1, k2 in itertools.product([i, i + 1, i + 2], [j, j + 1, j + 2])]
                G.add_node(f'L2_{i}-{j}_1', preds)
                G.add_node(f'L2_{i}-{j}_2', preds)
            for i, j in itertools.product(range(8), range(8)):
                preds = [f'L2_{k1}-{k2}_{k3}' for k1, k2 in
                         itertools.product([i * 3, i * 3 + 2, i * 3 + 4], [j * 3, j * 3 + 2, j * 3 + 4]) for k3 in
                         [1, 2]]
                G.add_node(f'L3_{i}-{j}_1', preds)
                G.add_node(f'L3_{i}-{j}_2', preds)
                G.add_node(f'L3_{i}-{j}_3', preds)
            for i, j in itertools.product(range(8), range(8)):
                preds = [f'L3_{k}-{j}_1' for k in range(8)]
                G.add_node(f'L4_{i}-{j}_1', preds)
                preds1 = [f'L3_{k}-{j}_2' for k in range(8)]
                G.add_node(f'L4_{i}-{j}_2', preds1)
                preds2 = [f'L3_{k}-{j}_3' for k in range(8)]
                G.add_node(f'L4_{i}-{j}_3', preds2)
           
            kqi = sum(map(lambda k: G.kqi(k) if "L4_" in k else 0, G.nodes()))
            logging.debug(f'L4: KQI={kqi}, node={len([k for k in G.nodes() if "L4_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L4_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L3_" in k else 0, G.nodes()))
            logging.debug(f'L3: KQI={kqi}, node={len([k for k in G.nodes() if "L3_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L3_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L2_" in k else 0, G.nodes()))
            logging.debug(f'L2: KQI={kqi}, node={len([k for k in G.nodes() if "L2_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L2_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L1_" in k else 0, G.nodes()))
            logging.debug(f'L1: KQI={kqi}, node={len([k for k in G.nodes() if "L1_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L1_" in k])}')
            logging.debug(f'Total volume = {G.graph_volume()}')

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestSoftmax2d().KQI(torch.randn(1, 28, 28))
    true = TestSoftmax2d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001
    print(kqi)


if __name__ == '__main__':
    test_ReLU()
    test_Tanh()
    test_Softmax()
    test_Softmax2d()
