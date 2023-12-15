import torch
import kqinn
import kqitool
import logging
import itertools


def test_BatchNorm2d():
    class TestBatchNorm2d(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = kqinn.BatchNorm2d(6)
            self.layer2 = kqinn.Linear(in_features=1*6*10*10, out_features=1*6*10*10,
                                       bias=False)
            self.layer3 = kqinn.BatchNorm2d(6)
            self.layer4 = kqinn.Linear(in_features=1*6*10*10, out_features=1*6*10*10,
                                       bias=False)
            self.layer5 = kqinn.BatchNorm2d(6)

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

            for i, j, k in itertools.product(range(10), range(10), range(6)):
                preds = [f'L0_{i}-{j}-{k}' for i, j in itertools.product(range(10), range(10))]
                G.add_node(f'L1_{i}-{j}-{k}', preds)

            #linear
            preds = [f'L1_{i}-{j}-{k}' for i, j, k in itertools.product(range(10), range(10), range(6))]
            for i, j, k in itertools.product(range(10), range(10), range(6)):
                G.add_node(f'L2_{i}-{j}-{k}', preds)

            for i, j, k in itertools.product(range(10), range(10), range(6)):
                preds = [f'L2_{i}-{j}-{k}' for i, j in itertools.product(range(10), range(10))]
                G.add_node(f'L3_{i}-{j}-{k}', preds)

            # Linear
            preds = [f'L3_{i}-{j}-{k}' for i, j, k in itertools.product(range(10), range(10), range(6))]
            for i, j, k in itertools.product(range(10), range(10), range(6)):
                G.add_node(f'L4_{i}-{j}-{k}', preds)

            for i, j, k in itertools.product(range(10), range(10), range(6)):
                preds = [f'L4_{i}-{j}-{k}' for i, j in itertools.product(range(10), range(10))]
                G.add_node(f'L5_{i}-{j}-{k}', preds)

            return sum(map(lambda m: G.kqi(m), G.nodes()))

    kqi = TestBatchNorm2d().KQI(torch.randn(1, 6, 10, 10))
    true = TestBatchNorm2d().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001

if __name__ == '__main__':
    test_BatchNorm2d()
