import itertools

import torch
import kqinn
import kqitool
import logging

# class SoftMax(torch.nn.Module, kqinn.KQI):
#     def __init__(self) -> None:
#         super().__init__()
#         self.layers1 = kqinn.Sequential(
#             # 1x28x28
#             kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
#             kqinn.ReLU(inplace=True),
#             # 2x26*26
#             kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
#             kqinn.ReLU(inplace=True),
#         )
#         self.layers2 = kqinn.Sequential(
#             kqinn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
#             kqinn.Linear(in_features=100, out_features=10, bias=False),
#             kqinn.SoftMax()
#         )
#
#     def forward(self, x):
#         x = self.layers1(x)
#         x = x.flatten()
#         x = self.layers2(x)
#         return x
#
#     def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.layers1.KQIforward(x)
#         x = x.flatten()
#         x = self.layers2.KQIforward(x)
#         return x
#
#     def KQIbackward(self, volumes: torch.Tensor, volume_backward: torch.Tensor = None) -> (torch.Tensor, float):
#         volumes, kqi = self.layers2.KQIbackward(volumes)
#         volumes = volumes.reshape(3, 8, 8)
#         volumes, kqi = self.layers1.KQIbackward(volumes, volume_backward)
#         return volumes, kqi

class softMax(torch.nn.Module, kqinn.KQI):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = kqinn.Linear(in_features=512, out_features=128, bias=False)
        self.linear2 = kqinn.Linear(in_features=128, out_features=10, bias=False)
        self.softMax = kqinn.SoftMax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.softMax(x)
        return x

    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1.KQIforward(x)
        x = self.linear2.KQIforward(x)
        x = self.softMax.KQIforward(x)
        return x

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        volume = self.softMax.KQIbackward(volume)
        volume = self.linear2.KQIbackward(volume)
        volume = self.linear1.KQIbackward(volume, volume_backward)
        return volume

def true_kqi():
    # G = kqitool.DiGraph()
    # for i in range(0, 256):
    #     G.add_node(i, [])
    # for i in range(256, 256+128):
    #     G.add_node(i, list(range(0, 256)))
    # for i in range(256+ 128, 258 + 128 + 10):
    #     G.add_node(i, list(range(256, 256 + 128)))
    G = kqitool.DiGraph()
    for i in range(0, 512):
        G.add_node(i, [])
    for i in range(512, 128+512):
        G.add_node(i, list(range(0, 512)))
    for i in range(512+128, 512 + 128 + 10):
        G.add_node(i, list(range(512, 512 + 128)))
    for i in range(512 + 128 + 10, 512 + 128 + 10 + 10):
        G.add_node(i, [i - 10])
    return sum(map(lambda k: G.kqi(k), G.nodes()))

# def true_kqi():
#     G = kqitool.DiGraph()
#     for i, j in itertools.product(range(28), range(28)):
#         G.add_node(f'L1_{i}-{j}', [])
#     for i, j in itertools.product(range(26), range(26)):
#         preds = [f'L1_{k1}-{k2}' for k1, k2 in itertools.product([i, i + 1, i + 2], [j, j + 1, j + 2])]
#         G.add_node(f'L2_{i}-{j}_1', preds)
#         G.add_node(f'L2_{i}-{j}_2', preds)
#     for i, j in itertools.product(range(26), range(26)):
#         G.add_node(f'L3_{i}-{j}_1', [f'L2_{i}-{j}_1'])
#         G.add_node(f'L3_{i}-{j}_2', [f'L2_{i}-{j}_2'])
#     for i, j in itertools.product(range(8), range(8)):
#         preds = [f'L3_{k1}-{k2}_{k3}' for k1, k2 in
#                  itertools.product([i * 3, i * 3 + 2, i * 3 + 4], [j * 3, j * 3 + 2, j * 3 + 4]) for k3 in [1, 2]]
#         G.add_node(f'L4_{i}-{j}_1', preds)
#         G.add_node(f'L4_{i}-{j}_2', preds)
#         G.add_node(f'L4_{i}-{j}_3', preds)
#     for i, j in itertools.product(range(8), range(8)):
#         G.add_node(f'L5_{i}-{j}_1', [f'L4_{i}-{j}_1'])
#         G.add_node(f'L5_{i}-{j}_2', [f'L4_{i}-{j}_2'])
#         G.add_node(f'L5_{i}-{j}_3', [f'L4_{i}-{j}_3'])
#
#     for i in range(100):
#         preds = [f'L5_{k1}-{k2}_{k3}' for k1, k2 in itertools.product(range(8), range(8)) for k3 in [1, 2, 3]]
#         G.add_node(f'L6_{i}', preds)
#
#     for i in range(10):
#         preds = [f'L6_{k}' for k in range(100)]
#         G.add_node(f'L7_{i}', preds)
#
#     for i in range(10):
#         G.add_node(f'L8_{i}', [f'L7_{i}']) # 添加softmax层 目前为一一对应
#
#     return sum(map(lambda k: G.kqi(k), G.nodes()))


def test():
    kqi = softMax().KQI(torch.randn(512))
    t = true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {t})')
    assert abs(kqi - t) / t < 0.0001


if __name__ == '__main__':
    test()
