import itertools

import torch
import kqinn
import kqitool

head = 5  # 注意力头数


class MHA(torch.nn.Module, kqinn.KQI):
    def __init__(self) -> None:
        super().__init__()

        for i in range(head):
            setattr(self, f'layer1_Q_{i}', kqinn.Linear(in_features=5 * 5, out_features=5 * 5, bias=False))
            setattr(self, f'layer1_K_{i}', kqinn.Linear(in_features=5 * 5, out_features=5 * 5, bias=False))
            setattr(self, f'layer1_V_{i}', kqinn.Linear(in_features=5 * 5, out_features=5 * 5, bias=False))
            setattr(self, f'layer2_{i}', kqinn.MatMul(featureM=5, featureN=5))
            setattr(self, f'layer3_{i}', kqinn.Scale(feature=5 * 5, scale=1))
            setattr(self, f'layer4_{i}', kqinn.Mask(feature=5 * 5, mask=1))
            setattr(self, f'layer5_{i}', kqinn.SoftMax(in_features=5 * 5, out_features=5 * 5))
            setattr(self, f'layer6_{i}', kqinn.MatMul(featureM=5, featureN=5))

        self.layer7 = kqinn.Linear(in_features=head * 5 * 5, out_features=head * 5 * 5, bias=False)

    # def forward(self, k, q, v):


def true_kqi():
    G = kqitool.DiGraph()
    # 构建 Q, K, V，均为 5*5*5 的张量
    for i, j, k in itertools.product(range(5), range(5), range(5)):
        G.add_node(f'L1_Q_{i}-{j}-{k}', [])
        G.add_node(f'L1_K_{i}-{j}-{k}', [])
        G.add_node(f'L1_V_{i}-{j}-{k}', [])

    # linear
    # 构建 Q, K, V 的线性层，均为 head*5*5 的张量
    # 线性层张量中的每个 5*5 的矩阵，与上一层中的一个 5*5 的矩阵全连接
    for i in range(head):
        preds = [f'L1_Q_{i}-{j}-{k}' for j, k in itertools.product(range(5), range(5))]
        for j, k in itertools.product(range(5), range(5)):
            G.add_node(f'L2_Q_{i}-{j}-{k}', preds)
        preds = [f'L1_K_{i}-{j}-{k}' for j, k in itertools.product(range(5), range(5))]
        for j, k in itertools.product(range(5), range(5)):
            G.add_node(f'L2_K_{i}-{j}-{k}', preds)
        preds = [f'L1_V_{i}-{j}-{k}' for j, k in itertools.product(range(5), range(5))]
        for j, k in itertools.product(range(5), range(5)):
            G.add_node(f'L2_V_{i}-{j}-{k}', preds)

    # MatMul
    # K, Q 线性层张量中的 5*5 矩阵相乘，形成 5*5*5 的张量 M
    for i in range(head):
        for j in range(5):
            for k in range(5):
                preds = [f'L2_Q_{i}-{j}-{l}' for l in range(5)] + [f'L2_K_{i}-{l}-{k}' for l in range(5)]
                G.add_node(f'L3{i}-{j}-{k}', preds)

    # Scale
    # 构建 M 的放缩层，为 5*5*5 的张量
    for i, j, k in itertools.product(range(head), range(5), range(5)):
        preds = [f'L3{i}-{j}-{k}']
        G.add_node(f'L4_{i}-{j}-{k}', preds)

    # Skip Mask

    # SoftMax
    # 构建 M 的 SoftMax 层，为 5*5*5 的张量
    for i in range(head):
        preds = [f'L4_{i}-{j}-{k}' for j, k in itertools.product(range(5), range(5))]
        for j, k in itertools.product(range(5), range(5)):
            G.add_node(f'L5_{i}-{j}-{k}', preds)

    # MatMul
    # SoftMax 层张量与 V 线性层张量中的 5*5 矩阵相乘，形成 5*5*5 的张量 M2
    for i in range(head):
        for j in range(5):
            for k in range(5):
                preds = [f'L5_{i}-{j}-{l}' for l in range(5)] + [f'L2_V_{i}-{l}-{k}' for l in range(5)]
                G.add_node(f'L6_{i}-{j}-{k}', preds)

    # Linear
    # 构建 M2 的线性层，为 head*5*5 的张量
    preds = [f'L6_{i}-{j}-{k}' for i, j, k in itertools.product(range(head), range(5), range(5))]
    for i, j, k in itertools.product(range(head), range(5), range(5)):
        G.add_node(f'L7_{i}-{j}-{k}', preds)

    return sum(map(lambda m: G.kqi(m), G.nodes()))


def test():
    # kqi = MHA().KQI(torch.randn(1 * 28 * 28))

    true = true_kqi()
    print("true_kqi: ", true)
    # logging.debug(f'KQI = {kqi} (True KQI = {true})')
    # assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test()
