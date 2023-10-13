import itertools

import torch
import kqinn
import kqitool


# class matMul(torch.nn.Module, kqinn.KQI):

def true_kqi():
    G = kqitool.DiGraph()
    # 构建 L, R, 均为 5*5 的矩阵
    for i, j in itertools.product(range(5), range(5)):
        G.add_node(f'L_{i}-{j}', [])
        G.add_node(f'R_{i}-{j}', [])

    # MatMul
    for i in range(5):
        for j in range(5):
            preds = [f'L_{i}-{k}' for k in range(5)] + [f'R_{k}-{j}' for k in range(5)]
            G.add_node(f'{i}-{j}', preds)

    return sum(map(lambda m: G.kqi(m), G.nodes()))


def test():
    # kqi = MHA().KQI(torch.randn(1 * 28 * 28))

    true = true_kqi()
    print("true_kqi: ", true)
    # logging.debug(f'KQI = {kqi} (True KQI = {true})')
    # assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test()
