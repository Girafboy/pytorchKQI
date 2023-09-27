import itertools

import torch
import kqinn
import kqitool
import logging


def true_kqi():
    G = kqitool.DiGraph()
    # 构建 Q, K, V，均为 5*5*5 的张量
    for i, j, k in itertools.product(range(5), range(5), range(5)):
        G.add_node(f'L1_Q_{i}-{j}-{k}', [])
        G.add_node(f'L1_K_{i}-{j}-{k}', [])
        G.add_node(f'L1_V_{i}-{j}-{k}', [])

    # linear
    # 构建 Q, K 的线性层，均为 5*5*5 的张量
    # 线性层张量中的每个 5*5 的矩阵，与上一层中的一个 5*5 的矩阵全连接
    for i in range(5):
        preds = [f'L1_Q_{i}-{j}-{k}' for j, k in itertools.product(range(5), range(5))]
        for j, k in itertools.product(range(5), range(5)):
            G.add_node(f'L2_Q_{i}-{j}-{k}', preds)
        preds = [f'L1_K_{i}-{j}-{k}' for j, k in itertools.product(range(5), range(5))]
        for j, k in itertools.product(range(5), range(5)):
            G.add_node(f'L2_K_{i}-{j}-{k}', preds)

    # MatMul
    # K, Q 线性层张量中的 5*5 矩阵相乘，形成 5*5*5 的张量 M
    for i in range(5):
        for j in range(5):
            for k in range(5):
                for l in range(5):
                    preds = [f'L2_Q_{i}-{j}-{l}', f'L2_K_{i}-{l}-{k}']
                G.add_node(f'L3{i}-{j}-{k}', preds)

    # Scale
    # 构建 M 的放缩层，为 5*5*5 的张量
    for i, j, k in itertools.product(range(5), range(5), range(5)):
        preds = [f'L3{i}-{j}-{k}']
        G.add_node(f'L4_{i}-{j}-{k}', preds)

    # Skip Mask

    # SoftMax
    # 构建 M 的 SoftMax 层，为 5*5*5 的张量
    for i in range(5):
        preds = [f'L4_{i}-{j}-{k}' for j, k in itertools.product(range(5), range(5))]
        for j, k in itertools.product(range(5), range(5)):
            G.add_node(f'L5_{i}-{j}-{k}', preds)

    # MatMul
    # SoftMax 层张量与 V 线性层张量中的 5*5 矩阵相乘，形成 5*5*5 的张量 M2
    for i in range(5):
        for j in range(5):
            for k in range(5):
                for l in range(5):
                    preds = [f'L5_{i}-{j}-{l}', f'L1_V_{i}-{l}-{k}']
                G.add_node(f'L6_{i}-{j}-{k}', preds)

    # Linear
    # 构建 M2 的线性层，为 5*5*5 的张量
    preds = [f'L6_{i}-{j}-{k}' for i, j, k in itertools.product(range(5), range(5), range(5))]
    for i, j, k in itertools.product(range(5), range(5), range(5)):
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
