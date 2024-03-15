import torch
import kqinn
import kqitool
import itertools
import logging


def test_Threshold():
    class TestThreshold(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.Threshold(0.1, 20, inplace=True),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.Threshold(0.1, 20, inplace=True),
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

    kqi = TestThreshold().KQI(torch.randn(1, 28, 28))
    true = TestThreshold().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


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


def test_Hardtanh():
    class TestHardtanh(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=True),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=True),
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

    kqi = TestHardtanh().KQI(torch.randn(1, 28, 28))
    true = TestHardtanh().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_ReLU6():
    class TestReLU6(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.ReLU6(inplace=True),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.ReLU6(inplace=True),
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

    kqi = TestReLU6().KQI(torch.randn(1, 28, 28))
    true = TestReLU6().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_Sigmoid():
    class TestSigmoid(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.Sigmoid(),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.Sigmoid(),
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

    kqi = TestSigmoid().KQI(torch.randn(1, 28, 28))
    true = TestSigmoid().true_kqi()
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
                preds = [f'L2_{k1}-{k2}_{k3}' for k1, k2 in itertools.product([i * 3, i * 3 + 2, i * 3 + 4], [j * 3, j * 3 + 2, j * 3 + 4]) for k3 in [1, 2]]
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
                preds = [f'L2_{k1}-{k2}_{k3}' for k1, k2 in itertools.product([i * 3, i * 3 + 2, i * 3 + 4], [j * 3, j * 3 + 2, j * 3 + 4]) for k3 in [1, 2]]
                G.add_node(f'L3_{i}-{j}_1', preds)
                G.add_node(f'L3_{i}-{j}_2', preds)
                G.add_node(f'L3_{i}-{j}_3', preds)
            for i, j in itertools.product(range(8), range(8)):
                preds = [f'L3_{i}-{j}_{k}' for k in [1, 2, 3]]
                G.add_node(f'L4_{i}-{j}_1', preds)
                G.add_node(f'L4_{i}-{j}_2', preds)
                G.add_node(f'L4_{i}-{j}_3', preds)

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestSoftmax2d().KQI(torch.randn(1, 28, 28))
    true = TestSoftmax2d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001
    print(kqi)


def test_LogSoftmax():
    class TestLogSoftmax(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                # 2x26x26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),

            )
            self.layers2 = kqinn.Sequential(
                kqinn.LogSoftmax(dim=1),
                kqinn.LogSoftmax(dim=2)
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

    kqi = TestLogSoftmax().KQI(torch.randn(1, 28, 28))
    true = TestLogSoftmax().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_ELU():
    class TestELU(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.ELU(alpha=1.0, inplace=True),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.ELU(alpha=1.0, inplace=True),
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

    kqi = TestELU().KQI(torch.randn(1, 28, 28))
    true = TestELU().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_SELU():
    class TestSELU(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.SELU(inplace=True),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.SELU(inplace=True),
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

    kqi = TestSELU().KQI(torch.randn(1, 28, 28))
    true = TestSELU().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_CELU():
    class TestCELU(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.CELU(alpha=1.0, inplace=True),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.CELU(alpha=1.0, inplace=True),
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

    kqi = TestCELU().KQI(torch.randn(1, 28, 28))
    true = TestCELU().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_GELU():
    class TestGELU(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.GELU(approximate='none'),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.GELU(approximate='none'),
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

    kqi = TestGELU().KQI(torch.randn(1, 28, 28))
    true = TestGELU().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_Hardshrink():
    class TestHardshrink(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.Hardshrink(lambd=0.5),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.Hardshrink(lambd=0.5),
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

    kqi = TestHardshrink().KQI(torch.randn(1, 28, 28))
    true = TestHardshrink().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_LeakyReLU():
    class TestLeakyReLU(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.LeakyReLU(negative_slope=0.01, inplace=True),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.LeakyReLU(negative_slope=0.01, inplace=True),
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

    kqi = TestLeakyReLU().KQI(torch.randn(1, 28, 28))
    true = TestLeakyReLU().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_LogSigmoid():
    class TestLogSigmoid(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.LogSigmoid(),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.LogSigmoid(),
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

    kqi = TestLogSigmoid().KQI(torch.randn(1, 28, 28))
    true = TestLogSigmoid().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_Softplus():
    class TestSoftplus(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.Softplus(),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.Softplus(),
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

    kqi = TestSoftplus().KQI(torch.randn(1, 28, 28))
    true = TestSoftplus().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_Softshrink():
    class TestSoftshrink(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.Softshrink(),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.Softshrink(),
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

    kqi = TestSoftshrink().KQI(torch.randn(1, 28, 28))
    true = TestSoftshrink().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def MultiheadAttention_add_nodes(G, preds_q, preds_k, preds_v, head, head_dim, sequence_length, embedding_dim, name_in="MHA_in", name_out="MHA_out"):
    """
    :param G: The graph to add nodes to
    :param preds_q: The prefix name of the input tensor for Q
    :param preds_k: The prefix name of the input tensor for K
    :param preds_v: The prefix name of the input tensor for V
    :param head: The number of heads
    :param head_dim: The dimension of each head
    :param sequence_length: The length of the sequence
    :param embedding_dim: The dimension of the embedding, which is the same as head * head_dim
    :param name_in: Prefix names of nodes in the graph
    :param name_out: Prefix name of output nodes
    :return: The graph with nodes added
    """

    # linear
    for i in range(head):
        predsQ = [f'{preds_q}_{j}-{k}' for j, k in
                  itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim))]
        predsK = [f'{preds_k}_{j}-{k}' for j, k in
                  itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim))]
        predsV = [f'{preds_v}_{j}-{k}' for j, k in
                  itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim))]
        for j, k in itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim)):
            G.add_node(f'{name_in}_L1_Q_{j}-{k}', predsQ)
            G.add_node(f'{name_in}_L1_K_{j}-{k}', predsK)
            G.add_node(f'{name_in}_L1_V_{j}-{k}', predsV)

    # MatMul
    for i in range(head):
        for j in range(sequence_length):
            for k in range(sequence_length):
                preds = ([f'{name_in}_L1_Q_{j}-{i * head_dim + m}' for m in range(head_dim)]
                         + [f'{name_in}_L1_K_{k}-{i * head_dim + m}' for m in range(head_dim)])
                G.add_node(f'{name_in}_L2_{j}-{i * sequence_length + k}', preds)

    # Scale
    for i in range(head):
        for j in range(sequence_length):
            for k in range(sequence_length):
                preds = [f'{name_in}_L2_{j}-{i * sequence_length + k}']
                G.add_node(f'{name_in}_L3_{j}-{i * sequence_length + k}', preds)

    # Mask
    # for i in range(head):
    #     for j in range(sequence_length):
    #         for k in range(sequence_length):
    #             preds = [f'MultiheadAttention_L3_{j}-{i * sequence_length + k}']
    #             G.add_node(f'MultiheadAttention_L4_{j}-{i * sequence_length + k}', preds)

    # SoftMax
    for i in range(head):
        preds = [f'{name_in}_L3_{j}-{i * sequence_length + k}' for j, k in
                 itertools.product(range(sequence_length), range(sequence_length))]
        for j, k in itertools.product(range(sequence_length), range(sequence_length)):
            G.add_node(f'{name_in}_L5_{j}-{i * sequence_length + k}', preds)

    # MatMul
    for i in range(head):
        for j in range(sequence_length):
            for k in range(head_dim):
                preds = ([f'{name_in}_L5_{j}-{i * sequence_length + m}' for m in range(sequence_length)] +
                         [f'{name_in}_L1_V_{m}-{i * head_dim + k}' for m in range(sequence_length)])
                G.add_node(f'{name_in}_L6_{j}-{i * head_dim + k}', preds)

    # Linear
    preds = [f'{name_in}_L6_{j}-{k}' for j, k in itertools.product(range(sequence_length), range(embedding_dim))]
    for j, k in itertools.product(range(sequence_length), range(embedding_dim)):
        G.add_node(f'{name_out}_{j}-{k}', preds)

    return G


def test_MultiheadAttention():
    head = 8
    embedding_dim = 64
    sequence_length = 10
    head_dim = embedding_dim // head

    class TestMultiheadAttention(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()

            self.layerQKV = kqinn.Linear(in_features=embedding_dim * sequence_length,
                                         out_features=embedding_dim * sequence_length * 3, bias=False)
            self.layer = kqinn.MultiheadAttention(embed_dim=embedding_dim, num_heads=head)

        def forward(self, x):
            x = x.flatten()
            qkv = self.layerQKV(x).reshape(sequence_length, embedding_dim * 3)
            q = qkv[:, :embedding_dim].reshape(sequence_length, embedding_dim)
            k = qkv[:, embedding_dim:embedding_dim * 2].reshape(sequence_length, embedding_dim)
            v = qkv[:, embedding_dim * 2:].reshape(sequence_length, embedding_dim)
            attn_output, attn_output_weights = self.layer(k, q, v)
            return attn_output

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.flatten()
            qkv = self.layerQKV.KQIforward(x).reshape(sequence_length, embedding_dim * 3)
            q = qkv[:, :embedding_dim].reshape(sequence_length, embedding_dim)
            k = qkv[:, embedding_dim:embedding_dim * 2].reshape(sequence_length, embedding_dim)
            v = qkv[:, embedding_dim * 2:].reshape(sequence_length, embedding_dim)
            attn_output, attn_output_weights = self.layer.KQIforward(k, q, v)
            return attn_output

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            volume_backward_k, volume_backward_q, volume_backward_v = self.layer.KQIbackward(volume)
            volume_backward_qkv = torch.cat([volume_backward_q, volume_backward_k, volume_backward_v], dim=1)
            volume = self.layerQKV.KQIbackward(volume_backward_qkv, volume_backward)
            return volume.reshape(sequence_length, embedding_dim)

        def true_kqi(self):
            G = kqitool.DiGraph()
            # Construct Q, K, V
            for i, j in itertools.product(range(sequence_length), range(embedding_dim)):
                G.add_node(f'L_{i}-{j}', [])

            preds = [f'L_{i}-{j}' for i, j in itertools.product(range(sequence_length), range(embedding_dim))]
            for i, j in itertools.product(range(sequence_length), range(embedding_dim)):
                G.add_node(f'L0_Q_{i}-{j}', preds)
                G.add_node(f'L0_K_{i}-{j}', preds)
                G.add_node(f'L0_V_{i}-{j}', preds)

            G = MultiheadAttention_add_nodes(G, 'L0_Q', 'L0_K', 'L0_V', head, head_dim, sequence_length, embedding_dim)

            return sum(map(lambda m: G.kqi(m), G.nodes()))

    kqi = TestMultiheadAttention().KQI(torch.randn(sequence_length, embedding_dim))
    true = TestMultiheadAttention().true_kqi()
    print("true_kqi: ", true)
    print("kqi: ", kqi)
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_PReLU():
    class TestPReLU(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.PReLU(num_parameters=1, init=0.25),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.PReLU(num_parameters=1, init=0.25),
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

    kqi = TestPReLU().KQI(torch.randn(1, 28, 28))
    true = TestPReLU().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_Softsign():
    class TestSoftsign(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.Softsign(),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.Softsign(),
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

    kqi = TestSoftsign().KQI(torch.randn(1, 28, 28))
    true = TestSoftsign().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_Softmin():
    class TestSoftmin(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                # 2x26x26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),

            )
            self.layers2 = kqinn.Sequential(
                kqinn.Softmin(dim=1),
                kqinn.Softmin(dim=2)
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

    kqi = TestSoftmin().KQI(torch.randn(1, 28, 28))
    true = TestSoftmin().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_Tanhshrink():
    class TestTanhshrink(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.Tanhshrink(),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.Tanhshrink(),
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

    kqi = TestTanhshrink().KQI(torch.randn(1, 28, 28))
    true = TestTanhshrink().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_RReLU():
    class TestRReLU(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.RReLU(0.1, 0.3, inplace=True),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.RReLU(0.1, 0.3, inplace=True),
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

    kqi = TestRReLU().KQI(torch.randn(1, 28, 28))
    true = TestRReLU().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_GLU():
    class TestGLU(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                # 2x26x26
                kqinn.GLU(dim=-1),
                # 2x26x13
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                # 3x8x3
                kqinn.GLU(dim=-2),
            )
            self.layers2 = kqinn.Sequential(
                # 3x4x3
                kqinn.Linear(in_features=3*4*3, out_features=10, bias=False),
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
            volume = volume.reshape(3, 4, 3)
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
            for i, j in itertools.product(range(26), range(13)):
                G.add_node(f'L3_{i}-{j}_1', [f'L2_{i}-{j}_1', f'L2_{i}-{j+13}_1'])
                G.add_node(f'L3_{i}-{j}_2', [f'L2_{i}-{j}_2', f'L2_{i}-{j+13}_2'])
            for i, j in itertools.product(range(8), range(3)):
                preds = [f'L3_{k1}-{k2}_{k3}' for k1, k2 in itertools.product([i*3, i*3+2, i*3+4], [j*3, j*3+2, j*3+4]) for k3 in [1, 2]]
                G.add_node(f'L4_{i}-{j}_1', preds)
                G.add_node(f'L4_{i}-{j}_2', preds)
                G.add_node(f'L4_{i}-{j}_3', preds)
            for i, j in itertools.product(range(4), range(3)):
                G.add_node(f'L5_{i}-{j}_1', [f'L4_{i}-{j}_1', f'L4_{i+4}-{j}_1'])
                G.add_node(f'L5_{i}-{j}_2', [f'L4_{i}-{j}_2', f'L4_{i+4}-{j}_2'])
                G.add_node(f'L5_{i}-{j}_3', [f'L4_{i}-{j}_3', f'L4_{i+4}-{j}_3'])

            for i in range(10):
                preds = [f'L5_{k1}-{k2}_{k3}' for k1, k2 in itertools.product(range(4), range(3)) for k3 in [1, 2, 3]]
                G.add_node(f'L6_{i}', preds)

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestGLU().KQI(torch.randn(1, 28, 28))
    true = TestGLU().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_Hardsigmoid():
    class TestHardsigmoid(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.Hardsigmoid(),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.Hardsigmoid(),
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

    kqi = TestHardsigmoid().KQI(torch.randn(1, 28, 28))
    true = TestHardsigmoid().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_Hardswish():
    class TestHardswish(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.Hardswish(inplace=True),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.Hardswish(inplace=True),
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

    kqi = TestHardswish().KQI(torch.randn(1, 28, 28))
    true = TestHardswish().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_SiLU():
    class TestSiLU(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.SiLU(inplace=True),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.SiLU(inplace=True),
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

    kqi = TestSiLU().KQI(torch.randn(1, 28, 28))
    true = TestSiLU().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_Mish():
    class TestMish(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                kqinn.SiLU(inplace=True),
                # 2x26*26
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.SiLU(inplace=True),
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

    kqi = TestMish().KQI(torch.randn(1, 28, 28))
    true = TestMish().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test_Threshold()
    test_ReLU()
    test_Hardtanh()
    test_ReLU6()
    test_Sigmoid()
    test_Tanh()

    test_Softmax()
    test_Softmax2d()
    test_LogSoftmax()
    test_ELU()
    test_SELU()
    test_CELU()
    test_GELU()
    test_Hardshrink()
    test_LeakyReLU()
    test_LogSigmoid()

    test_Softplus()
    test_Softshrink()
    test_PReLU()
    test_Softsign()
    test_Softmin()
    test_Tanhshrink()
    test_RReLU()
    test_GLU()

    test_Hardsigmoid()
    test_MultiheadAttention()
    test_Hardswish()
    test_SiLU()
    test_Mish()
