import torch
import kqinn
import kqitool
import itertools
import logging


class ResNet(torch.nn.Module, kqinn.KQI):
    def __init__(self) -> None:
        super().__init__()
        self.layers1 = kqinn.Sequential(
            # 1x28x28
            kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
            # 2x26*26
            kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
        )
        # 3*8*8
        self.layers2 = kqinn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.layers3 = kqinn.Sequential(
            # 3*8*8
            kqinn.Linear(in_features = 3*8*8, out_features = 100, bias=False),
            kqinn.Linear(in_features = 100, out_features = 10, bias=False),
        )


    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x) + x
        x = x.flatten()
        x = self.layers3(x)

        return x


    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers1.KQIforward(x)
        x0 = self.layers2.KQIforward(x)
        x = kqinn.kqi_add().KQIforward(x0, x)
        x = x.flatten()
        x = self.layers3.KQIforward(x)

        return x


    def KQIbackward(self, volumes: torch.Tensor, kqi: float) -> (torch.Tensor, float):
        volumes, kqi = self.layers3.KQIbackward(volumes, kqi)
        volumes = volumes.reshape(3,8,8)
        volumes0, kqi = kqinn.kqi_add().KQIbackward(volumes, kqi)

        volumes, kqi = kqinn.Combine(self.layers2.KQIbackward, kqinn.kqi_add().KQIbackward, volumes0, volumes, kqi)
        volumes, kqi = self.layers1.KQIbackward(volumes, kqi)

        return volumes, kqi


def true_kqi():
    G = kqitool.DiGraph()
    for i,j in itertools.product(range(28), range(28)):
        G.add_node(f'L1_{i}-{j}', [])
    for i,j in itertools.product(range(26), range(26)):
        preds = [f'L1_{k1}-{k2}' for k1, k2 in itertools.product([i, i+1, i+2], [j, j+1, j+2])]
        G.add_node(f'L2_{i}-{j}_1', preds)
        G.add_node(f'L2_{i}-{j}_2', preds)
    for i,j in itertools.product(range(8), range(8)):
        preds = [f'L2_{k1}-{k2}_{k3}' for k1, k2 in itertools.product([i*3, i*3+2, i*3+4], [j*3, j*3+2, j*3+4]) for k3 in [1,2]]
        G.add_node(f'L3_{i}-{j}_1', preds)
        G.add_node(f'L3_{i}-{j}_2', preds)
        G.add_node(f'L3_{i}-{j}_3', preds)

    for i in range(100):
        preds = [f'L3_{k1}-{k2}_{k3}' for k1,k2 in itertools.product(range(8), range(8)) for k3 in [1,2,3]]
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


def test():
    kqi = ResNet().KQI(torch.randn(1,28,28))
    print(kqi)
    true = true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test()