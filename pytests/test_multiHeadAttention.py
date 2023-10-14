import itertools
import logging
from typing import Tuple

import torch
from torch import Tensor
import kqinn
import kqitool


def test_MultiHeadAttention():
    head = 8  # 注意力头数
    embedding_dim = 64  # 嵌入维度
    sequence_length = 10  # 序列长度
    head_dim = embedding_dim // head  # 每个头的维度

    class TestMultiHeadAttention(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()

            self.layer = kqinn.MultiheadAttention(embed_dim=embedding_dim, num_heads=head)

        def forward(self, x):
            return self.layer(x)

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer.KQIforward(x)

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            return self.layer.KQIbackward(volume)

        def true_kqi(self):
            G = kqitool.DiGraph()
            # 构建 Q, K, V, 均为 sequence_length * embedding_dim 的张量
            for i, j in itertools.product(range(sequence_length), range(embedding_dim)):
                G.add_node(f'L1_Q_{i}-{j}', [])
                G.add_node(f'L1_K_{i}-{j}', [])
                G.add_node(f'L1_V_{i}-{j}', [])

            # linear
            for i in range(head):
                predsQ = [f'L1_Q_{j}-{k}' for j, k in
                          itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim))]
                predsK = [f'L1_K_{j}-{k}' for j, k in
                          itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim))]
                predsV = [f'L1_V_{j}-{k}' for j, k in
                          itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim))]
                for j, k in itertools.product(range(sequence_length), range(head_dim)):
                    G.add_node(f'L2_Q_{j}-{i * head_dim + k}', predsQ)
                    G.add_node(f'L2_K_{j}-{i * head_dim + k}', predsK)
                    G.add_node(f'L2_V_{j}-{i * head_dim + k}', predsV)

            # MatMul
            for i in range(head):
                for j in range(sequence_length):
                    for k in range(sequence_length):
                        preds = ([f'L2_Q_{j}-{i * head_dim + m}' for m in range(head_dim)]
                                 + [f'L2_K_{k}-{i * head_dim + m}' for m in range(head_dim)])
                        G.add_node(f'L3_{j}-{i * sequence_length + k}', preds)

            # Scale
            for i in range(head):
                for j in range(sequence_length):
                    for k in range(sequence_length):
                        preds = [f'L3_{j}-{i * sequence_length + k}']
                        G.add_node(f'L4_{j}-{i * sequence_length + k}', preds)

            # Mask
            # for i in range(head):
            #     for j in range(sequence_length):
            #         for k in range(sequence_length):
            #             preds = [f'L4_{j}-{i * sequence_length + k}']
            #             G.add_node(f'L5_{j}-{i * sequence_length + k}', preds)

            # SoftMax
            for i in range(head):
                preds = [f'L4_{j}-{i * sequence_length + k}' for j, k in
                         itertools.product(range(sequence_length), range(sequence_length))]
                for j, k in itertools.product(range(sequence_length), range(sequence_length)):
                    G.add_node(f'L6_{j}-{i * sequence_length + k}', preds)

            # MatMul
            for i in range(head):
                for j in range(sequence_length):
                    for k in range(head_dim):
                        preds = ([f'L6_{j}-{i * sequence_length + m}' for m in range(sequence_length)] +
                                 [f'L2_V_{m}-{i * head_dim + k}' for m in range(sequence_length)])
                        G.add_node(f'L7_{j}-{i * head_dim + k}', preds)

            # Linear
            preds = [f'L7_{j}-{k}' for j, k in itertools.product(range(sequence_length), range(embedding_dim))]
            for j, k in itertools.product(range(sequence_length), range(embedding_dim)):
                G.add_node(f'L8_{j}-{k}', preds)

            return sum(map(lambda m: G.kqi(m), G.nodes()))

    kqi = TestMultiHeadAttention().KQI(torch.randn(3, sequence_length, embedding_dim))
    true = TestMultiHeadAttention().true_kqi()
    print("true_kqi: ", true)
    print("kqi: ", kqi)
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test_MultiHeadAttention()
