import itertools
import torch
import testtool


def MultiheadAttention_add_nodes(G, preds_q, preds_k, preds_v, head, head_dim, sequence_length, embedding_dim, name_in="MHA_in", name_out="MHA_out"):
    """
    :param G: The graph to add nodes to
    :param preds_q: The prefix name of the input tensor for Q
    :param preds_k: The prefix name of the input tensor for K
    :param preds_v: The prefix name of the input tensor for V
    :param head: The number of heads
    :param head_dim: The dimension of each head
    :param sequence_length: The length of the sequence
    :param embedding_dim: The dimension of the embedding, which is the same as head * head_dim
    :param name_in: Prefix names of nodes in the graph
    :param name_out: Prefix name of output nodes
    :return: The graph with nodes added
    """

    # linear
    for i in range(head):
        predsQ = [f'{preds_q}_{j}-{k}' for j, k in
                  itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim))]
        predsK = [f'{preds_k}_{j}-{k}' for j, k in
                  itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim))]
        predsV = [f'{preds_v}_{j}-{k}' for j, k in
                  itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim))]
        for j, k in itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim)):
            G.add_node(f'{name_in}_L1_Q_{j}-{k}', predsQ)
            G.add_node(f'{name_in}_L1_K_{j}-{k}', predsK)
            G.add_node(f'{name_in}_L1_V_{j}-{k}', predsV)

    # MatMul
    for i in range(head):
        for j in range(sequence_length):
            for k in range(sequence_length):
                preds = ([f'{name_in}_L1_Q_{j}-{i * head_dim + m}' for m in range(head_dim)] + [f'{name_in}_L1_K_{k}-{i * head_dim + m}' for m in range(head_dim)])
                G.add_node(f'{name_in}_L2_{j}-{i * sequence_length + k}', preds)

    # Scale
    for i in range(head):
        for j in range(sequence_length):
            for k in range(sequence_length):
                preds = [f'{name_in}_L2_{j}-{i * sequence_length + k}']
                G.add_node(f'{name_in}_L3_{j}-{i * sequence_length + k}', preds)

    # Mask
    # for i in range(head):
    #     for j in range(sequence_length):
    #         for k in range(sequence_length):
    #             preds = [f'MultiheadAttention_L3_{j}-{i * sequence_length + k}']
    #             G.add_node(f'MultiheadAttention_L4_{j}-{i * sequence_length + k}', preds)

    # SoftMax
    for i in range(head):
        preds = [f'{name_in}_L3_{j}-{i * sequence_length + k}' for j, k in
                 itertools.product(range(sequence_length), range(sequence_length))]
        for j, k in itertools.product(range(sequence_length), range(sequence_length)):
            G.add_node(f'{name_in}_L5_{j}-{i * sequence_length + k}', preds)

    # MatMul
    for i in range(head):
        for j in range(sequence_length):
            for k in range(head_dim):
                preds = ([f'{name_in}_L5_{j}-{i * sequence_length + m}' for m in range(sequence_length)] + [f'{name_in}_L1_V_{m}-{i * head_dim + k}' for m in range(sequence_length)])
                G.add_node(f'{name_in}_L6_{j}-{i * head_dim + k}', preds)

    # Linear
    preds = [f'{name_in}_L6_{j}-{k}' for j, k in itertools.product(range(sequence_length), range(embedding_dim))]
    for j, k in itertools.product(range(sequence_length), range(embedding_dim)):
        G.add_node(f'{name_out}_{j}-{k}', preds)

    return G


def test_Transformer():
    # Currently, we only support the case of sequence_length = 1
    sequence_length = 1

    d_model = 32
    head = 8
    dim_feedforward = 48
    num_encoder_layers = 3
    num_decoder_layers = 3

    class TestTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(in_features=d_model * sequence_length, out_features=d_model * sequence_length * 2, bias=False)
            self.layer = torch.nn.Transformer(d_model=d_model, nhead=head, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

        def forward(self, x):
            x = x.flatten()
            x = self.linear(x).reshape(sequence_length, d_model * 2)
            tgt = x[:, :d_model].reshape(sequence_length, d_model)
            mem = x[:, d_model:].reshape(sequence_length, d_model)
            return self.layer(tgt, mem)

    testtool.testKQI(TestTransformer(), torch.randn(sequence_length, d_model))


def test_TransformerEncoder():
    # Currently, we only support the case of sequence_length = 1
    sequence_length = 1

    d_model = 32
    head = 8
    dim_feedforward = 48

    class TestTransformerEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=head, dim_feedforward=dim_feedforward, norm_first=True)
            self.layer = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        def forward(self, x):
            return self.layer(x)

    testtool.testKQI(TestTransformerEncoder(), torch.randn(sequence_length, d_model))


def test_TransformerDecoder():
    # Currently, we only support the case of sequence_length = 1
    sequence_length = 1

    d_model = 32
    head = 8
    dim_feedforward = 48

    class TestTransformerDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(in_features=d_model * sequence_length, out_features=d_model * sequence_length * 2, bias=False)
            self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=head, dim_feedforward=dim_feedforward, norm_first=True)
            self.layer = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=2)

        def forward(self, x):
            x = x.flatten()
            x = self.linear(x).reshape(sequence_length, d_model * 2)
            tgt = x[:, :d_model].reshape(sequence_length, d_model)
            mem = x[:, d_model:].reshape(sequence_length, d_model)
            return self.layer(tgt, mem)

    testtool.testKQI(TestTransformerDecoder(), torch.randn(sequence_length, d_model))


def test_TransformerEncoderLayer():
    # Currently, we only support the case of sequence_length = 1
    sequence_length = 1

    d_model = 32
    head = 8
    dim_feedforward = 48

    class TestTransformerEncoderLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=head, dim_feedforward=dim_feedforward,
                                                       norm_first=True)

        def forward(self, x):
            return self.layer(x)

    testtool.testKQI(TestTransformerEncoderLayer(), torch.randn(sequence_length, d_model))


def test_TransformerDecoderLayer():
    # Currently, we only support the case of sequence_length = 1
    sequence_length = 1

    d_model = 32
    head = 8
    dim_feedforward = 48

    class TestTransformerDecoderLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(in_features=d_model * sequence_length, out_features=d_model * sequence_length * 2, bias=False)
            self.layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=head, dim_feedforward=dim_feedforward, norm_first=True)

        def forward(self, x):
            x = x.flatten()
            x = self.linear(x).reshape(sequence_length, d_model * 2)
            tgt = x[:, :d_model].reshape(sequence_length, d_model)
            mem = x[:, d_model:].reshape(sequence_length, d_model)
            return self.layer(tgt, mem)

    testtool.testKQI(TestTransformerDecoderLayer(), torch.randn(sequence_length, d_model))


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
