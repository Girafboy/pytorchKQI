import torch
import kqinn
import kqitool
import logging
import itertools


def test_Dropout():
    class TestDropout(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = kqinn.Linear(in_features=784, out_features=512, bias=False)
            self.dropout1 = kqinn.Dropout(p=0.4)
            self.linear2 = kqinn.Linear(in_features=512, out_features=512, bias=False)
            self.dropout2 = kqinn.Dropout(p=0.3)
            self.linear3 = kqinn.Linear(in_features=512, out_features=10, bias=False)

        def forward(self, x):
            x = self.linear1(x)
            x = self.dropout1(x)
            x = self.linear2(x)
            x = self.dropout2(x)
            x = self.linear3(x)

            return x

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.linear1.KQIforward(x)
            x = self.dropout1.KQIforward(x)
            x = self.linear2.KQIforward(x)
            x = self.dropout2.KQIforward(x)
            x = self.linear3.KQIforward(x)

            return x

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            volume = self.linear3.KQIbackward(volume)
            volume = self.dropout2.KQIbackward(volume)
            volume = self.linear2.KQIbackward(volume)
            volume = self.dropout1.KQIbackward(volume)
            volume = self.linear1.KQIbackward(volume, volume_backward)

            return volume

        def true_kqi(self):
            G = kqitool.DiGraph()
            for i in range(0, 784):
                G.add_node(i, [])
            for i in range(784, 784+512):
                G.add_node(i, list(range(0, 784)))
            for i in range(784+512, 784+512+512):
                G.add_node(i, list(range(784, 784+512)))
            for i in range(784+512+512, 784+512+512+10):
                G.add_node(i, list(range(784+512, 784+512+512)))

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestDropout().KQI(torch.randn(1*28*28))
    true = TestDropout().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_Dropout1d():
    class TestDropout1d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 3x28
                kqinn.Conv1d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                kqinn.Dropout1d(p=0.5, inplace=True),
                kqinn.Conv1d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.Dropout1d(p=0.5, inplace=True),
            )
            self.layers2 = kqinn.Sequential(
                # 3x8
                kqinn.Linear(in_features=3*8, out_features=10, bias=False),
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
            volume = volume.reshape(3, 8)
            volume = self.layers1.KQIbackward(volume, volume_backward)

            return volume

        def true_kqi(self):
            G = kqitool.DiGraph()
            for i in range(28):
                G.add_node(f'L1_{i}_1', [])
                G.add_node(f'L1_{i}_2', [])
                G.add_node(f'L1_{i}_3', [])

            for i in range(28):
                preds = [f'L1_{k1}_{k2}' for k1 in [i-1, i, i+1] if k1 >= 0 and k1 < 28 for k2 in [1, 2, 3]]
                G.add_node(f'L2_{i}_1', preds)
                G.add_node(f'L2_{i}_2', preds)

            for i in range(8):
                preds = [f'L2_{k1}_{k2}' for k1 in [i*3, i*3+2, i*3+4] for k2 in [1, 2]]
                G.add_node(f'L3_{i}_1', preds)
                G.add_node(f'L3_{i}_2', preds)
                G.add_node(f'L3_{i}_3', preds)

            for i in range(10):
                preds = [f'L3_{k1}_{k2}' for k1 in range(8) for k2 in [1, 2, 3]]
                G.add_node(f'L4_{i}', preds)

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestDropout1d().KQI(torch.randn(3, 28))
    true = TestDropout1d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_Dropout2d():
    class TestDropout2d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x28x28
                kqinn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                kqinn.Dropout2d(p=0.4, inplace=True),
                kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.Dropout2d(p=0.5, inplace=True),
            )
            self.layers2 = kqinn.Sequential(
                # 3x8x8
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
                G.add_node(f'L1_{i}-{j}_1', [])
                G.add_node(f'L1_{i}-{j}_2', [])
                G.add_node(f'L1_{i}-{j}_3', [])

            for i, j in itertools.product(range(28), range(28)):
                preds = [f'L1_{k1}-{k2}_{k3}' for k1, k2 in itertools.product([i-1, i, i+1], [j-1, j, j+1]) if k1 >= 0 and k1 < 28 and k2 >= 0 and k2 < 28 for k3 in [1, 2, 3]]
                G.add_node(f'L2_{i}-{j}_1', preds)
                G.add_node(f'L2_{i}-{j}_2', preds)

            for i, j in itertools.product(range(8), range(8)):
                preds = [f'L2_{k1}-{k2}_{k3}' for k1, k2 in itertools.product([i*3, i*3+2, i*3+4], [j*3, j*3+2, j*3+4]) for k3 in [1, 2]]
                G.add_node(f'L3_{i}-{j}_1', preds)
                G.add_node(f'L3_{i}-{j}_2', preds)
                G.add_node(f'L3_{i}-{j}_3', preds)

            for i in range(100):
                preds = [f'L3_{k1}-{k2}_{k3}' for k1, k2 in itertools.product(range(8), range(8)) for k3 in [1, 2, 3]]
                G.add_node(f'L4_{i}', preds)

            for i in range(10):
                preds = [f'L4_{k}' for k in range(100)]
                G.add_node(f'L5_{i}', preds)

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestDropout2d().KQI(torch.randn(3, 28, 28))
    true = TestDropout2d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_Dropout3d():
    class TestDropout3d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x9x9x9
                kqinn.Conv3d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                kqinn.Dropout3d(p=0.4, inplace=True),
                kqinn.Conv3d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.Dropout3d(p=0.5, inplace=True),
            )
            self.layers2 = kqinn.Sequential(
                # 3x2x2x2
                kqinn.Linear(in_features=3*2*2*2, out_features=100, bias=False),
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
                preds = [f'L1_{m1}-{m2}-{m3}_{m4}' for m1, m2, m3 in itertools.product([i-1, i, i+1], [j-1, j, j+1], [k-1, k, k+1]) if m1 >= 0 and m1 < 9 and m2 >= 0 and m2 < 9 and m3 >= 0 and m3 < 9 for m4 in [1, 2, 3]]
                G.add_node(f'L2_{i}-{j}-{k}_1', preds)
                G.add_node(f'L2_{i}-{j}-{k}_2', preds)

            for i, j, k in itertools.product(range(2), range(2), range(2)):
                preds = [f'L2_{m1}-{m2}-{m3}_{m4}' for m1, m2, m3 in itertools.product([i*3, i*3+2, i*3+4], [j*3, j*3+2, j*3+4], [k*3, k*3+2, k*3+4]) for m4 in [1, 2]]
                G.add_node(f'L3_{i}-{j}-{k}_1', preds)
                G.add_node(f'L3_{i}-{j}-{k}_2', preds)
                G.add_node(f'L3_{i}-{j}-{k}_3', preds)

            for i in range(100):
                preds = [f'L3_{m1}-{m2}-{m3}_{m4}' for m1, m2, m3 in itertools.product(range(2), range(2), range(2)) for m4 in [1, 2, 3]]
                G.add_node(f'L4_{i}', preds)

            for i in range(10):
                preds = [f'L4_{m}' for m in range(100)]
                G.add_node(f'L5_{i}', preds)

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestDropout3d().KQI(torch.randn(3, 9, 9, 9))
    true = TestDropout3d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_AlphaDropout():
    class TestAlphaDropout(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = kqinn.Linear(in_features=784, out_features=512, bias=False)
            self.dropout1 = kqinn.AlphaDropout(p=0.4)
            self.linear2 = kqinn.Linear(in_features=512, out_features=512, bias=False)
            self.dropout2 = kqinn.AlphaDropout(p=0.3)
            self.linear3 = kqinn.Linear(in_features=512, out_features=10, bias=False)

        def forward(self, x):
            x = self.linear1(x)
            x = self.dropout1(x)
            x = self.linear2(x)
            x = self.dropout2(x)
            x = self.linear3(x)

            return x

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.linear1.KQIforward(x)
            x = self.dropout1.KQIforward(x)
            x = self.linear2.KQIforward(x)
            x = self.dropout2.KQIforward(x)
            x = self.linear3.KQIforward(x)

            return x

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            volume = self.linear3.KQIbackward(volume)
            volume = self.dropout2.KQIbackward(volume)
            volume = self.linear2.KQIbackward(volume)
            volume = self.dropout1.KQIbackward(volume)
            volume = self.linear1.KQIbackward(volume, volume_backward)

            return volume

        def true_kqi(self):
            G = kqitool.DiGraph()
            for i in range(0, 784):
                G.add_node(i, [])
            for i in range(784, 784+512):
                G.add_node(i, list(range(0, 784)))
            for i in range(784+512, 784+512+512):
                G.add_node(i, list(range(784, 784+512)))
            for i in range(784+512+512, 784+512+512+10):
                G.add_node(i, list(range(784+512, 784+512+512)))

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestAlphaDropout().KQI(torch.randn(1*28*28))
    true = TestAlphaDropout().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_FeatureAlphaDropout():
    class TestFeatureAlphaDropout(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                # 1x9x9x9
                kqinn.Conv3d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                kqinn.FeatureAlphaDropout(p=0.4, inplace=True),
                kqinn.Conv3d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                kqinn.FeatureAlphaDropout(p=0.5, inplace=True),
            )
            self.layers2 = kqinn.Sequential(
                # 3x2x2x2
                kqinn.Linear(in_features=3*2*2*2, out_features=100, bias=False),
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
                preds = [f'L1_{m1}-{m2}-{m3}_{m4}' for m1, m2, m3 in itertools.product([i-1, i, i+1], [j-1, j, j+1], [k-1, k, k+1]) if m1 >= 0 and m1 < 9 and m2 >= 0 and m2 < 9 and m3 >= 0 and m3 < 9 for m4 in [1, 2, 3]]
                G.add_node(f'L2_{i}-{j}-{k}_1', preds)
                G.add_node(f'L2_{i}-{j}-{k}_2', preds)

            for i, j, k in itertools.product(range(2), range(2), range(2)):
                preds = [f'L2_{m1}-{m2}-{m3}_{m4}' for m1, m2, m3 in itertools.product([i*3, i*3+2, i*3+4], [j*3, j*3+2, j*3+4], [k*3, k*3+2, k*3+4]) for m4 in [1, 2]]
                G.add_node(f'L3_{i}-{j}-{k}_1', preds)
                G.add_node(f'L3_{i}-{j}-{k}_2', preds)
                G.add_node(f'L3_{i}-{j}-{k}_3', preds)

            for i in range(100):
                preds = [f'L3_{m1}-{m2}-{m3}_{m4}' for m1, m2, m3 in itertools.product(range(2), range(2), range(2)) for m4 in [1, 2, 3]]
                G.add_node(f'L4_{i}', preds)

            for i in range(10):
                preds = [f'L4_{m}' for m in range(100)]
                G.add_node(f'L5_{i}', preds)

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestFeatureAlphaDropout().KQI(torch.randn(3, 9, 9, 9))
    true = TestFeatureAlphaDropout().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test_Dropout()
    test_Dropout1d()
    test_Dropout2d()
    test_Dropout3d()
    test_AlphaDropout()
    test_FeatureAlphaDropout()
