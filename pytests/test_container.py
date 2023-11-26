import torch
import kqinn
import kqitool
import logging


def test_Sequential():
    class TestSequential(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = kqinn.Sequential(
                kqinn.Linear(in_features=784, out_features=512, bias=False),
                kqinn.Linear(in_features=512, out_features=512, bias=False),
            )
            self.layers2 = kqinn.Sequential(
                kqinn.Linear(in_features=512, out_features=10, bias=False),
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
            for i in range(0, 784):
                G.add_node(i, [])
            for i in range(784, 784+512):
                G.add_node(i, list(range(0, 784)))
            for i in range(784+512, 784+512+512):
                G.add_node(i, list(range(784, 784+512)))
            for i in range(784+512+512, 784+512+512+10):
                G.add_node(i, list(range(784+512, 784+512+512)))

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestSequential().KQI(torch.randn(1*28*28))
    true = TestSequential().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test_Sequential()
