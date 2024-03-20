import itertools
import logging

import torch

import kqinn
import kqitool
from pytests.test_activation import MultiheadAttention_add_nodes


def test_Transformer():
    # Currently, we only support the case of sequence_length = 1
    sequence_length = 1

    d_model = 32
    head = 8
    dim_feedforward = 48
    num_encoder_layers = 3
    num_decoder_layers = 3

    class TestTransformer(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.linear = kqinn.Linear(in_features=d_model * sequence_length, out_features=d_model * sequence_length * 2, bias=False)
            self.layer = kqinn.Transformer(d_model=d_model, nhead=head, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

        def forward(self, x):
            x = x.flatten()
            x = self.linear(x).reshape(sequence_length, d_model * 2)
            tgt = x[:, :d_model].reshape(sequence_length, d_model)
            mem = x[:, d_model:].reshape(sequence_length, d_model)
            return self.layer(tgt, mem)

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.flatten()
            x = self.linear.KQIforward(x).reshape(sequence_length, d_model * 2)
            tgt = x[:, :d_model].reshape(sequence_length, d_model)
            mem = x[:, d_model:].reshape(sequence_length, d_model)
            return self.layer.KQIforward(tgt, mem)

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            volume_backward_tgt, volume_backward_mem = self.layer.KQIbackward(volume)
            volume_backward_tgt_mem = torch.cat([volume_backward_tgt, volume_backward_mem], dim=1)
            volume = self.linear.KQIbackward(volume_backward_tgt_mem, volume_backward)
            return volume.reshape(sequence_length, d_model)

        def true_kqi(self):
            G = kqitool.DiGraph()

            # Construct mem and tgt nodes
            for i, j in itertools.product(range(sequence_length), range(d_model)):
                G.add_node(f'start_{i}-{j}', [])
            preds = [f'start_{i}-{j}' for i, j in itertools.product(range(sequence_length), range(d_model))]
            for i, j in itertools.product(range(sequence_length), range(d_model)):
                G.add_node(f'src_{i}-{j}', preds)
                G.add_node(f'tgt_{i}-{j}', preds)

            G = TransformerEncoderLayer_add_nodes(G, "src", sequence_length, d_model, head, dim_feedforward, name_in="TEL1", name_out="TEL1_out")
            G = TransformerEncoderLayer_add_nodes(G, "TEL1_out", sequence_length, d_model, head, dim_feedforward, name_in="TEL2", name_out="TEL2_out")
            G = TransformerEncoderLayer_add_nodes(G, "TEL2_out", sequence_length, d_model, head, dim_feedforward, name_in="TEL3", name_out="TEL3_out")
            G = TransformerDecoderLayer_add_nodes(G, "tgt", "TEL3_out", sequence_length, d_model, head, dim_feedforward, name_in="TDL1", name_out="TDL1_out")
            G = TransformerDecoderLayer_add_nodes(G, "TDL1_out", "TEL3_out", sequence_length, d_model, head, dim_feedforward, name_in="TDL2", name_out="TDL2_out")
            G = TransformerDecoderLayer_add_nodes(G, "TDL2_out", "TEL3_out", sequence_length, d_model, head, dim_feedforward, name_in="TDL3", name_out="TDL3_out")

            return sum(map(lambda m: G.kqi(m), G.nodes()))

    kqi = TestTransformer().KQI(torch.randn(sequence_length, d_model))
    true = TestTransformer().true_kqi()
    print("true_kqi: ", true)
    print("kqi: ", kqi)
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_TransformerEncoder():
    # Currently, we only support the case of sequence_length = 1
    sequence_length = 1

    d_model = 32
    head = 8
    dim_feedforward = 48

    class TestTransformerEncoder(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()

            self.encoder_layer = kqinn.TransformerEncoderLayer(d_model=d_model, nhead=head,
                                                               dim_feedforward=dim_feedforward,
                                                               norm_first=True)
            self.layer = kqinn.TransformerEncoder(self.encoder_layer, num_layers=2)

        def forward(self, x):
            return self.layer(x)

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer.KQIforward(x)

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            return self.layer.KQIbackward(volume, volume_backward)

        def true_kqi(self):
            G = kqitool.DiGraph()
            for i in range(sequence_length):
                for j in range(d_model):
                    G.add_node(f'L0_{i}-{j}', [])
            G = TransformerEncoderLayer_add_nodes(G, "L0", sequence_length, d_model, head, dim_feedforward, name_in="TEL1", name_out="TEL1_out")
            G = TransformerEncoderLayer_add_nodes(G, "TEL1_out", sequence_length, d_model, head, dim_feedforward, name_in="TEL2", name_out="TEL2_out")

            return sum(map(lambda m: G.kqi(m), G.nodes()))

    kqi = TestTransformerEncoder().KQI(torch.randn(sequence_length, d_model))
    true = TestTransformerEncoder().true_kqi()
    print("true_kqi: ", true)
    print("kqi: ", kqi)
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_TransformerDecoder():
    # Currently, we only support the case of sequence_length = 1
    sequence_length = 1

    d_model = 32
    head = 8
    dim_feedforward = 48

    class TestTransformerDecoder(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.linear = kqinn.Linear(in_features=d_model * sequence_length, out_features=d_model * sequence_length * 2, bias=False)
            self.decoder_layer = kqinn.TransformerDecoderLayer(d_model=d_model, nhead=head, dim_feedforward=dim_feedforward, norm_first=True)
            self.layer = kqinn.TransformerDecoder(self.decoder_layer, num_layers=2)

        def forward(self, x):
            x = x.flatten()
            x = self.linear(x).reshape(sequence_length, d_model * 2)
            tgt = x[:, :d_model].reshape(sequence_length, d_model)
            mem = x[:, d_model:].reshape(sequence_length, d_model)
            return self.layer(tgt, mem)

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.flatten()
            x = self.linear.KQIforward(x).reshape(sequence_length, d_model * 2)
            tgt = x[:, :d_model].reshape(sequence_length, d_model)
            mem = x[:, d_model:].reshape(sequence_length, d_model)
            return self.layer.KQIforward(tgt, mem)

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            volume_backward_tgt, volume_backward_mem = self.layer.KQIbackward(volume)
            volume_backward_tgt_mem = torch.cat([volume_backward_tgt, volume_backward_mem], dim=1)
            volume = self.linear.KQIbackward(volume_backward_tgt_mem, volume_backward)
            return volume.reshape(sequence_length, d_model)

        def true_kqi(self):
            G = kqitool.DiGraph()

            for i, j in itertools.product(range(sequence_length), range(d_model)):
                G.add_node(f'start_{i}-{j}', [])
            preds = [f'start_{i}-{j}' for i, j in itertools.product(range(sequence_length), range(d_model))]
            for i, j in itertools.product(range(sequence_length), range(d_model)):
                G.add_node(f'mem_{i}-{j}', preds)
                G.add_node(f'tgt_{i}-{j}', preds)

            G = TransformerDecoderLayer_add_nodes(G, "tgt", "mem", sequence_length, d_model, head, dim_feedforward, name_in="TDL1", name_out="TDL1_out")
            G = TransformerDecoderLayer_add_nodes(G, "TDL1_out", "mem", sequence_length, d_model, head, dim_feedforward, name_in="TDL2", name_out="TDL2_out")

            return sum(map(lambda m: G.kqi(m), G.nodes()))

    kqi = TestTransformerDecoder().KQI(torch.randn(sequence_length, d_model))
    true = TestTransformerDecoder().true_kqi()
    print("true_kqi: ", true)
    print("kqi: ", kqi)
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_TransformerEncoderLayer():
    # Currently, we only support the case of sequence_length = 1
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
            for i in range(sequence_length):
                for j in range(d_model):
                    G.add_node(f'L0_{i}-{j}', [])

            G = TransformerEncoderLayer_add_nodes(G, "L0", sequence_length, d_model, head, dim_feedforward)

            return sum(map(lambda m: G.kqi(m), G.nodes()))

    kqi = TestTransformerEncoderLayer().KQI(torch.randn(sequence_length, d_model))
    true = TestTransformerEncoderLayer().true_kqi()
    print("true_kqi: ", true)
    print("kqi: ", kqi)
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_TransformerDecoderLayer():
    # Currently, we only support the case of sequence_length = 1
    sequence_length = 1

    d_model = 32
    head = 8
    dim_feedforward = 48

    class TestTransformerDecoderLayer(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            self.linear = kqinn.Linear(in_features=d_model * sequence_length, out_features=d_model * sequence_length * 2, bias=False)
            self.layer = kqinn.TransformerDecoderLayer(d_model=d_model, nhead=head, dim_feedforward=dim_feedforward, norm_first=True)

        def forward(self, x):
            x = x.flatten()
            x = self.linear(x).reshape(sequence_length, d_model * 2)
            tgt = x[:, :d_model].reshape(sequence_length, d_model)
            mem = x[:, d_model:].reshape(sequence_length, d_model)
            return self.layer(tgt, mem)

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.flatten()
            x = self.linear.KQIforward(x).reshape(sequence_length, d_model * 2)
            tgt = x[:, :d_model].reshape(sequence_length, d_model)
            mem = x[:, d_model:].reshape(sequence_length, d_model)
            return self.layer.KQIforward(tgt, mem)

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            volume_backward_tgt, volume_backward_mem = self.layer.KQIbackward(volume)
            volume_backward_tgt_mem = torch.cat([volume_backward_tgt, volume_backward_mem], dim=1)
            volume = self.linear.KQIbackward(volume_backward_tgt_mem, volume_backward)
            return volume.reshape(sequence_length, d_model)

        def true_kqi(self):
            G = kqitool.DiGraph()

            # Construct mem and tgt nodes
            for i, j in itertools.product(range(sequence_length), range(d_model)):
                G.add_node(f'start_{i}-{j}', [])
            preds = [f'start_{i}-{j}' for i, j in itertools.product(range(sequence_length), range(d_model))]
            for i, j in itertools.product(range(sequence_length), range(d_model)):
                G.add_node(f'mem_{i}-{j}', preds)
                G.add_node(f'tgt_{i}-{j}', preds)

            G = TransformerDecoderLayer_add_nodes(G, "tgt", "mem", sequence_length, d_model, head, dim_feedforward)

            return sum(map(lambda m: G.kqi(m), G.nodes()))

    kqi = TestTransformerDecoderLayer().KQI(torch.randn(sequence_length, d_model))
    true = TestTransformerDecoderLayer().true_kqi()
    print("true_kqi: ", true)
    print("kqi: ", kqi)
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def TransformerEncoderLayer_add_nodes(G, predecessors, sequence_length, d_model, head, dim_feedforward, name_in="TEL_in", name_out="TEL_out"):
    """
    :param G: The graph to add nodes to
    :param predecessors: The prefix name of predecessors of the first layer, which is a two-dimensional tensor, and the size is (sequence_length, d_model)
    :param sequence_length: The length of the sequence
    :param d_model: The dimension of the model
    :param head: The number of heads in the multiheadattention models
    :param dim_feedforward: The dimension of the feedforward network models
    :param name_in: Prefix names of nodes in the graph
    :param name_out: Prefix name of output nodes, which is a one-dimensional tensor
    :return: The graph with nodes added
    """

    # Norm1
    preds = [f'{predecessors}_{i}-{j}' for i, j in itertools.product(range(sequence_length), range(d_model))]
    for i in range(sequence_length):
        for j in range(d_model):
            G.add_node(f'{name_in}_L1_{i}-{j}', preds)

    # MultiheadAttention
    G = MultiheadAttention_add_nodes(G, name_in + "_L1", name_in + "_L1", name_in + "_L1", head, d_model // head, sequence_length,
                                     d_model, name_in=name_in + "_MHA_in", name_out=name_in + "_MHA_out")

    # Add
    for i in range(sequence_length):
        for j in range(d_model):
            preds = ([f'{predecessors}_{i}-{j}'] + [f'{name_in}_MHA_out_{i}-{j}'])
            G.add_node(f'{name_in}_L9_{i}-{j}', preds)

    # Norm2
    preds = [f'{name_in}_L9_{i}-{j}' for i, j in itertools.product(range(sequence_length), range(d_model))]
    for i in range(sequence_length):
        for j in range(d_model):
            G.add_node(f'{name_in}_L10_{i}-{j}', preds)

    # Linear1
    preds = [f'{name_in}_L10_{i}-{j}' for i, j in itertools.product(range(sequence_length), range(d_model))]
    for i in range(sequence_length):
        for j in range(dim_feedforward):
            G.add_node(f'{name_in}_L11_{i}-{j}', preds)

    # Linear2
    preds = [f'{name_in}_L11_{i}-{j}' for i, j in itertools.product(range(sequence_length), range(dim_feedforward))]
    for i in range(sequence_length):
        for j in range(d_model):
            G.add_node(f'{name_in}_L12_{i}-{j}', preds)

    # Add
    for i in range(sequence_length):
        for j in range(d_model):
            preds = ([f'{name_in}_L9_{i}-{j}'] + [f'{name_in}_L12_{i}-{j}'])
            G.add_node(f'{name_out}_{i}-{j}', preds)

    return G


def TransformerDecoderLayer_add_nodes(G, tgt, mem, sequence_length, d_model, head, dim_feedforward, name_in="TDL_in", name_out="TDL_out"):
    # Norm1
    preds = [f'{tgt}_{i}-{j}' for i, j in itertools.product(range(sequence_length), range(d_model))]
    for i in range(sequence_length):
        for j in range(d_model):
            G.add_node(f'{name_in}_TDL_norm1_{i}-{j}', preds)

    # Self-Attention
    G = MultiheadAttention_add_nodes(G, name_in + "_TDL_norm1", name_in + "_TDL_norm1", name_in + "_TDL_norm1", head, d_model // head, sequence_length, d_model,
                                     name_in=name_in + "_SA_in", name_out=name_in + "_SA_out")

    # Add1
    for i in range(sequence_length):
        for j in range(d_model):
            preds = ([f'{tgt}_{i}-{j}'] + [f'{name_in}_SA_out_{i}-{j}'])
            G.add_node(f'{name_in}_TDL_add1_{i}-{j}', preds)

    # Norm2
    preds = [f'{name_in}_TDL_add1_{i}-{j}' for i, j in itertools.product(range(sequence_length), range(d_model))]
    for i in range(sequence_length):
        for j in range(d_model):
            G.add_node(f'{name_in}_TDL_norm2_{i}-{j}', preds)

    # MultiheadAttention
    G = MultiheadAttention_add_nodes(G, name_in + "_TDL_norm2", mem, mem, head, d_model // head, sequence_length, d_model,
                                     name_in=name_in + "_MHA_in", name_out=name_in + "_MHA_out")

    # Add2
    for i in range(sequence_length):
        for j in range(d_model):
            preds = ([f'{name_in}_TDL_add1_{i}-{j}'] + [f'{name_in}_MHA_out_{i}-{j}'])
            G.add_node(f'{name_in}_TDL_add2_{i}-{j}', preds)

    # Norm3
    preds = [f'{name_in}_TDL_add2_{i}-{j}' for i, j in itertools.product(range(sequence_length), range(d_model))]
    for i in range(sequence_length):
        for j in range(d_model):
            G.add_node(f'{name_in}_TDL_norm3_{i}-{j}', preds)

    # Linear1
    preds = [f'{name_in}_TDL_norm3_{i}-{j}' for i, j in itertools.product(range(sequence_length), range(d_model))]
    for i in range(sequence_length):
        for j in range(dim_feedforward):
            G.add_node(f'{name_in}_TDL_linear1_{i}-{j}', preds)

    # Linear2
    preds = [f'{name_in}_TDL_linear1_{i}-{j}' for i, j in itertools.product(range(sequence_length), range(dim_feedforward))]
    for i in range(sequence_length):
        for j in range(d_model):
            G.add_node(f'{name_in}_TDL_linear2_{i}-{j}', preds)

    # Add3
    for i in range(sequence_length):
        for j in range(d_model):
            preds = ([f'{name_in}_TDL_add2_{i}-{j}'] + [f'{name_in}_TDL_linear2_{i}-{j}'])
            G.add_node(f'{name_out}_{i}-{j}', preds)

    return G


if __name__ == '__main__':
    test_Transformer()
    test_TransformerEncoder()
    test_TransformerDecoder()
    test_TransformerEncoderLayer()
    test_TransformerDecoderLayer()
