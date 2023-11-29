import itertools
import logging

import torch

import kqinn
import kqitool


def test_TransformerEncoderLayer():
    batch_size = 1
    sequence_length = 1
    d_model = 32
    head = 8
    dim_feedforward = 48

    class TestTransformerEncoderLayer(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()

            self.layer = kqinn.TransformerEncoderLayer(d_model=d_model, nhead=head, dim_feedforward=dim_feedforward,
                                                       norm_first=True)

        def forward(self, x):
            return self.layer(x)

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer.KQIforward(x)

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            return self.layer.KQIbackward(volume, volume_backward)

        def true_kqi(self):
            G = kqitool.DiGraph()

            for i in range(d_model):
                G.add_node(f'L0_{i}', [])

            # Norm1
            preds = [f'L0_{i}' for i in range(d_model)]
            for i in range(sequence_length):
                for j in range(d_model):
                    G.add_node(f'L1_{i}-{j}', preds)

            # ------------------------- MultiheadAttention -------------------------
            embedding_dim = d_model
            head_dim = embedding_dim // head
            # linear
            for i in range(head):
                predsQ = [f'L1_{j}-{k}' for j, k in
                          itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim))]
                predsK = [f'L1_{j}-{k}' for j, k in
                          itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim))]
                predsV = [f'L1_{j}-{k}' for j, k in
                          itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim))]
                for j, k in itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim)):
                    G.add_node(f'L2_Q_{j}-{k}', predsQ)
                    G.add_node(f'L2_K_{j}-{k}', predsK)
                    G.add_node(f'L2_V_{j}-{k}', predsV)
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

            # ------------------------- MultiheadAttention -------------------------

            # Add
            for i in range(sequence_length):
                for j in range(d_model):
                    preds = ([f'L0_{j}'] + [f'L8_{i}-{j}'])
                    G.add_node(f'L9_{j}', preds)

            # Norm2
            preds = [f'L9_{i}' for i in range(d_model)]
            for i in range(d_model):
                G.add_node(f'L10_{i}', preds)

            # Linear1
            preds = [f'L10_{i}' for i in range(d_model)]
            for i in range(dim_feedforward):
                G.add_node(f'L11_{i}', preds)

            # Linear2
            preds = [f'L11_{i}' for i in range(dim_feedforward)]
            for i in range(d_model):
                G.add_node(f'L12_{i}', preds)

            # Add
            for i in range(d_model):
                preds = ([f'L9_{i}'] + [f'L12_{i}'])
                G.add_node(f'L13_{i}', preds)

            kqi = sum(map(lambda k: G.kqi(k) if "L13_" in k else 0, G.nodes()))
            logging.debug(
                f'L13: KQI={kqi}, node={len([k for k in G.nodes() if "L13_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L13_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L12_" in k else 0, G.nodes()))
            logging.debug(
                f'L12: KQI={kqi}, node={len([k for k in G.nodes() if "L12_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L12_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L11_" in k else 0, G.nodes()))
            logging.debug(
                f'L11: KQI={kqi}, node={len([k for k in G.nodes() if "L11_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L11_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L10_" in k else 0, G.nodes()))
            logging.debug(
                f'L10: KQI={kqi}, node={len([k for k in G.nodes() if "L10_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L10_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L9_" in k else 0, G.nodes()))
            logging.debug(
                f'L9: KQI={kqi}, node={len([k for k in G.nodes() if "L9_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L9_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L8_" in k else 0, G.nodes()))
            logging.debug(
                f'L8: KQI={kqi}, node={len([k for k in G.nodes() if "L8_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L8_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L7_" in k else 0, G.nodes()))
            logging.debug(
                f'L7: KQI={kqi}, node={len([k for k in G.nodes() if "L7_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L7_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L6_" in k else 0, G.nodes()))
            logging.debug(
                f'L6: KQI={kqi}, node={len([k for k in G.nodes() if "L6_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L6_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L4_" in k else 0, G.nodes()))
            logging.debug(
                f'L4: KQI={kqi}, node={len([k for k in G.nodes() if "L4_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L4_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L3_" in k else 0, G.nodes()))
            logging.debug(
                f'L3: KQI={kqi}, node={len([k for k in G.nodes() if "L3_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L3_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L2_" in k else 0, G.nodes()))
            logging.debug(
                f'L2: KQI={kqi}, node={len([k for k in G.nodes() if "L2_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L2_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L1_" in k else 0, G.nodes()))
            logging.debug(
                f'L1: KQI={kqi}, node={len([k for k in G.nodes() if "L1_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L1_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L0_" in k else 0, G.nodes()))
            logging.debug(
                f'L0: KQI={kqi}, node={len([k for k in G.nodes() if "L0_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L0_" in k])}')
            logging.debug(f'Total volume = {G.graph_volume()}')

            return sum(map(lambda m: G.kqi(m), G.nodes()))

    kqi = TestTransformerEncoderLayer().KQI(torch.randn(sequence_length, batch_size, d_model))
    true = TestTransformerEncoderLayer().true_kqi()
    print("true_kqi: ", true)
    print("kqi: ", kqi)
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test_TransformerEncoderLayer()
