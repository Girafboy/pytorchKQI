import torch
import kqinn
import kqitool
import logging


def test_Branch():
    class TestBranch(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = kqinn.Linear(in_features=784, out_features=512, bias=False)
            self.linear2 = kqinn.Linear(in_features=512, out_features=512, bias=False)
            self.linear3 = kqinn.Linear(in_features=512, out_features=10, bias=False)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x) + x
            x = self.linear3(x)

            return x

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.linear1.KQIforward(x)
            x1, x2 = kqinn.Branch(kqinn.Sequential(self.linear2, kqinn.SimplePass()), kqinn.SimplePass()).KQIforward(x)
            x = x1 + x2
            x = self.linear3.KQIforward(x)

            return x

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            volume = self.linear3.KQIbackward(volume)
            volume1, volume2 = volume/2, volume/2
            volume = kqinn.Branch(kqinn.Sequential(self.linear2, kqinn.SimplePass()), kqinn.SimplePass()).KQIbackward(volume1, volume2)
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
            for i in range(784+512+512, 784+512+512+512):
                G.add_node(i, [i-512, i-512-512])
            for i in range(784+512+512+512, 784+512+512+512+10):
                G.add_node(i, list(range(784+512+512, 784+512+512+512)))

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestBranch().KQI(torch.randn(1*28*28))
    true = TestBranch().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test_Branch()
