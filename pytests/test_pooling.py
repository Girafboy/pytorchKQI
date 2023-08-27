import torch
import kqinn
import kqitool
import itertools
import logging


class CNN(torch.nn.Module, kqinn.KQI):
    def __init__(self) -> None:
        super().__init__()
        self.layers1 = kqinn.Sequential(
            # 1x28x28
            kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
            # 2x26x26
            kqinn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layers2 = kqinn.Sequential(
            kqinn.Linear(in_features=2 * 13 * 13, out_features=100, bias=False),
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

    def KQIbackward(self, volumes: torch.Tensor, kqi: float) -> (torch.Tensor, float):
        volumes, kqi = self.layers2.KQIbackward(volumes, kqi)
        volumes = volumes.reshape(2, 13, 13)
        volumes, kqi = self.layers1.KQIbackward(volumes, kqi)
        return volumes, kqi


def true_kqi():
    G = kqitool.DiGraph()
    for i,j in itertools.product(range(28), range(28)):
        G.add_node(f'L1_{i}-{j}', [])
    for i, j in itertools.product(range(26), range(26)):
        preds = [f'L1_{k1}-{k2}' for k1, k2 in itertools.product([i, i + 1, i + 2], [j, j + 1, j + 2])]
        G.add_node(f'L2_{i}-{j}_1', preds)
        G.add_node(f'L2_{i}-{j}_2', preds)
    for i, j in itertools.product(range(13), range(13)):
        for k3 in [1,2]:
            preds = [f'L2_{k1}-{k2}_{k3}' for k1, k2 in itertools.product([i * 2, i * 2 + 1], [j * 2, j * 2 + 1])]
            G.add_node(f'L3_{i}-{j}_{k3}', preds)

    for i in range(100):
        preds = [f'L3_{k1}-{k2}_{k3}' for k1,k2 in itertools.product(range(13), range(13)) for k3 in [1,2]]
        G.add_node(f'L4_{i}', preds)
    for i in range(10):
        preds = [f'L4_{k}' for k in range(100)]
        G.add_node(f'L5_{i}', preds)

    return sum(map(lambda k: G.kqi(k), G.nodes()))


def test():
    kqi = CNN().KQI(torch.randn(1,28,28))

    true = true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test()
