import torch
import testtool


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

            self.layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=head, dim_feedforward=dim_feedforward, norm_first=True)

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


if __name__ == '__main__':
    test_Transformer()
    test_TransformerEncoder()
    test_TransformerDecoder()
    test_TransformerEncoderLayer()
    test_TransformerDecoderLayer()
