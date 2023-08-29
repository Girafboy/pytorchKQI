import torch
import kqinn
import kqitool
import logging


class MLP(torch.nn.Module, kqinn.KQI):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = kqinn.Linear(in_features = 784, out_features = 512, bias=False)
        self.linear2 = kqinn.Linear(in_features = 512, out_features = 512, bias=False)
        self.linear3 = kqinn.Linear(in_features = 512, out_features = 10, bias=False)


    def forward(self, x):
        x = self.linear1(x)
        x1 = self.linear2(x)
        x = x1 + x
        x = self.linear3(x)

        return x


    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1.KQIforward(x)
        x1, x = kqinn.Branch(self.linear2, kqinn.SimplePass()).KQIforward(x)
        x = kqinn.SimplePass().KQIforward(x1)
        x = self.linear3.KQIforward(x)
        
        return x


    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        volume = self.linear3.KQIbackward(volume)
        volume1 = kqinn.SimplePass().KQIbackward(volume/2)
        volume = kqinn.Branch(self.linear2, kqinn.SimplePass()).KQIbackward(volume1, volume/2)
        volume = self.linear1.KQIbackward(volume, volume_backward)
        
        return volume


def true_kqi():
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


def test():
    kqi = MLP().KQI(torch.randn(1*28*28))

    true = true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test()