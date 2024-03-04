import itertools
import logging

import torch

import kqinn
import kqitool


# logging.basicConfig(level=logging.DEBUG)


def test_TransformerEncoder():
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

            # ----------------------------------------------------------
            # ------------------------- layer1 -------------------------
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # ------------------------- layer2 -------------------------
            # ----------------------------------------------------------

            # Norm1
            preds = [f'L13_{i}' for i in range(d_model)]
            for i in range(sequence_length):
                for j in range(d_model):
                    G.add_node(f'L14_{i}-{j}', preds)

            # ------------------------- MultiheadAttention -------------------------
            embedding_dim = d_model
            head_dim = embedding_dim // head
            # linear
            for i in range(head):
                predsQ = [f'L14_{j}-{k}' for j, k in
                          itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim))]
                predsK = [f'L14_{j}-{k}' for j, k in
                          itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim))]
                predsV = [f'L14_{j}-{k}' for j, k in
                          itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim))]
                for j, k in itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim)):
                    G.add_node(f'L15_Q_{j}-{k}', predsQ)
                    G.add_node(f'L15_K_{j}-{k}', predsK)
                    G.add_node(f'L15_V_{j}-{k}', predsV)
            # MatMul
            for i in range(head):
                for j in range(sequence_length):
                    for k in range(sequence_length):
                        preds = ([f'L15_Q_{j}-{i * head_dim + m}' for m in range(head_dim)]
                                 + [f'L15_K_{k}-{i * head_dim + m}' for m in range(head_dim)])
                        G.add_node(f'L16_{j}-{i * sequence_length + k}', preds)
            # Scale
            for i in range(head):
                for j in range(sequence_length):
                    for k in range(sequence_length):
                        preds = [f'L16_{j}-{i * sequence_length + k}']
                        G.add_node(f'L17_{j}-{i * sequence_length + k}', preds)
            # Mask
            # for i in range(head):
            #     for j in range(sequence_length):
            #         for k in range(sequence_length):
            #             preds = [f'L17_{j}-{i * sequence_length + k}']
            #             G.add_node(f'L18_{j}-{i * sequence_length + k}', preds)
            # SoftMax
            for i in range(head):
                preds = [f'L17_{j}-{i * sequence_length + k}' for j, k in
                         itertools.product(range(sequence_length), range(sequence_length))]
                for j, k in itertools.product(range(sequence_length), range(sequence_length)):
                    G.add_node(f'L19_{j}-{i * sequence_length + k}', preds)
            # MatMul
            for i in range(head):
                for j in range(sequence_length):
                    for k in range(head_dim):
                        preds = ([f'L19_{j}-{i * sequence_length + m}' for m in range(sequence_length)] +
                                 [f'L15_V_{m}-{i * head_dim + k}' for m in range(sequence_length)])
                        G.add_node(f'L20_{j}-{i * head_dim + k}', preds)
            # Linear
            preds = [f'L20_{j}-{k}' for j, k in itertools.product(range(sequence_length), range(embedding_dim))]
            for j, k in itertools.product(range(sequence_length), range(embedding_dim)):
                G.add_node(f'L21_{j}-{k}', preds)

            # ------------------------- MultiheadAttention -------------------------

            # Add
            for i in range(sequence_length):
                for j in range(d_model):
                    preds = ([f'L13_{j}'] + [f'L21_{i}-{j}'])
                    G.add_node(f'L22_{j}', preds)

            # Norm2
            preds = [f'L22_{i}' for i in range(d_model)]
            for i in range(d_model):
                G.add_node(f'L23_{i}', preds)

            # Linear1
            preds = [f'L23_{i}' for i in range(d_model)]
            for i in range(dim_feedforward):
                G.add_node(f'L24_{i}', preds)

            # Linear2
            preds = [f'L24_{i}' for i in range(dim_feedforward)]
            for i in range(d_model):
                G.add_node(f'L25_{i}', preds)

            # Add
            for i in range(d_model):
                preds = ([f'L22_{i}'] + [f'L25_{i}'])
                G.add_node(f'L26_{i}', preds)

            return sum(map(lambda m: G.kqi(m), G.nodes()))

    kqi = TestTransformerEncoder().KQI(torch.randn(sequence_length, d_model))
    true = TestTransformerEncoder().true_kqi()
    print("true_kqi: ", true)
    print("kqi: ", kqi)
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_TransformerEncoderLayer():
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

            return sum(map(lambda m: G.kqi(m), G.nodes()))

    kqi = TestTransformerEncoderLayer().KQI(torch.randn(sequence_length, d_model))
    true = TestTransformerEncoderLayer().true_kqi()
    print("true_kqi: ", true)
    print("kqi: ", kqi)
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_TransformerDecoderLayer():
    batch_size = 1
    sequence_length = 1
    d_model = 32
    head = 8
    dim_feedforward = 48

    class TestTransformerDecoderLayer(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()

            self.layer = kqinn.TransformerDecoderLayer(d_model=d_model, nhead=head, dim_feedforward=dim_feedforward,
                                                       norm_first=True)
            self.encoder_output = None  # Initialize a placeholder for memory

        def set_encoder_output(self, encoder_output):
            """ Set the encoder output to be used in the KQIforward method. """
            self.encoder_output = encoder_output

        def forward(self, x, encoder_output):
            return self.layer(x, encoder_output)

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            if self.encoder_output is None:
                raise ValueError("Encoder output not set")
            return self.layer(x, self.encoder_output)


        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            return self.layer.KQIbackward(volume, volume_backward)

        def true_kqi(self):
            G = kqitool.DiGraph()

            # Define Memory nodes representing the encoder output
            for i in range(sequence_length):
                for k in range(d_model):
                    G.add_node(f'M_{i}-{k}', [])  # No predecessors for encoder output nodes

            for i in range(d_model):
                G.add_node(f'L0_{i}', [])

            # Norm1
            preds = [f'L0_{i}' for i in range(d_model)]
            for i in range(sequence_length):
                for j in range(d_model):
                    G.add_node(f'L1_{i}-{j}', preds)

            # ------------------------- SelfAttention -------------------------
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
            for i in range(head):
                for j in range(sequence_length):
                    for k in range(sequence_length):
                        preds = [f'L4_{j}-{i * sequence_length + k}']
                        G.add_node(f'L5_{j}-{i * sequence_length + k}', preds)
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

            # ------------------------- SelfAttention -------------------------

            # Add
            for i in range(sequence_length):
                for j in range(d_model):
                    preds = ([f'L0_{j}'] + [f'L8_{i}-{j}'])
                    G.add_node(f'L9_{i}-{j}', preds)

            # Norm2
            preds = [f'L9_{i}-{j}' for i, j in itertools.product(range(sequence_length), range(embedding_dim))]
            # resize L10
            for i in range(d_model):
                for j in range(embedding_dim):
                    G.add_node(f'L10_{i}-{j}', preds)

            # ------------------------- MultiheadAttention -------------------------
            # embedding_dim = d_model
            # head_dim = embedding_dim // head
            # linear
            for i in range(head):
                predsQ = [f'L10_{j}-{k}' for j, k in
                          itertools.product(range(sequence_length), range(embedding_dim))]
                predsK = [f'M_{j}-{k}' for j, k in  # These preds come from the Memory (encoder output)
                          itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim))]
                predsV = [f'M_{j}-{k}' for j, k in  # These preds come from the Memory (encoder output)
                          itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim))]
                for j, k in itertools.product(range(sequence_length), range(i * head_dim, (i + 1) * head_dim)):
                    G.add_node(f'L11_Q_{j}-{k}', predsQ)
                    G.add_node(f'L11_K_{j}-{k}', predsK)
                    G.add_node(f'L11_V_{j}-{k}', predsV)
            # MatMul
            for i in range(head):
                for j in range(sequence_length):
                    for k in range(sequence_length):
                        preds = ([f'L11_Q_{j}-{i * head_dim + m}' for m in range(head_dim)]
                                 + [f'L11_K_{k}-{i * head_dim + m}' for m in range(head_dim)])
                        G.add_node(f'L12_{j}-{i * sequence_length + k}', preds)
            # Scale
            for i in range(head):
                for j in range(sequence_length):
                    for k in range(sequence_length):
                        preds = [f'L12_{j}-{i * sequence_length + k}']
                        G.add_node(f'L13_{j}-{i * sequence_length + k}', preds)
            # Mask
            # for i in range(head):
            #     for j in range(sequence_length):
            #         for k in range(sequence_length):
            #             preds = [f'L13_{j}-{i * sequence_length + k}']
            #             G.add_node(f'L14_{j}-{i * sequence_length + k}', preds)
            # SoftMax
            for i in range(head):
                preds = [f'L13_{j}-{i * sequence_length + k}' for j, k in
                         itertools.product(range(sequence_length), range(sequence_length))]
                for j, k in itertools.product(range(sequence_length), range(sequence_length)):
                    G.add_node(f'L15_{j}-{i * sequence_length + k}', preds)
            # MatMul
            for i in range(head):
                for j in range(sequence_length):
                    for k in range(head_dim):
                        preds = ([f'L15_{j}-{i * sequence_length + m}' for m in range(sequence_length)] +
                                 [f'L11_V_{m}-{i * head_dim + k}' for m in range(sequence_length)])
                        G.add_node(f'L16_{j}-{i * head_dim + k}', preds)
            # Linear
            preds = [f'L16_{j}-{k}' for j, k in itertools.product(range(sequence_length), range(embedding_dim))]
            for j, k in itertools.product(range(sequence_length), range(embedding_dim)):
                G.add_node(f'L17_{j}-{k}', preds)

            # ------------------------- MultiheadAttention -------------------------

            # Add
            for i in range(sequence_length):
                for j in range(d_model):
                    preds = ([f'L9_{i}-{j}'] + [f'L17_{i}-{j}'])
                    G.add_node(f'L18_{j}', preds)

            # Norm3
            preds = [f'L18_{i}' for i in range(d_model)]
            for i in range(d_model):
                G.add_node(f'L19_{i}', preds)

            # Linear1
            preds = [f'L19_{i}' for i in range(d_model)]
            for i in range(dim_feedforward):
                G.add_node(f'L20_{i}', preds)

            # Linear2
            preds = [f'L20_{i}' for i in range(dim_feedforward)]
            for i in range(d_model):
                G.add_node(f'L21_{i}', preds)

            # Add
            for i in range(d_model):
                preds = ([f'L18_{i}'] + [f'L21_{i}'])
                G.add_node(f'L22_{i}', preds)


            return sum(map(lambda m: G.kqi(m), G.nodes()))

    encoder_output_tensor = torch.randn(sequence_length, batch_size, d_model)
    test_layer = TestTransformerDecoderLayer()
    test_layer.set_encoder_output(encoder_output_tensor)

    kqi = test_layer.KQI(torch.randn(sequence_length, batch_size, d_model))
    true = test_layer.true_kqi()
    print("true_kqi: ", true)
    print("kqi: ", kqi)
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test_TransformerEncoder()
    test_TransformerEncoderLayer()
    test_TransformerDecoderLayer()
