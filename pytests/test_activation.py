import torch
import testtool


def test_Threshold():
    class TestThreshold(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.Threshold(0.1, 20, inplace=True),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.Threshold(0.1, 20, inplace=True),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestThreshold(), torch.randn(1, 28, 28))


def test_ReLU():
    class TestReLU(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.ReLU(inplace=True),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.ReLU(inplace=True),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestReLU(), torch.randn(1, 28, 28))


def test_Hardtanh():
    class TestHardtanh(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=True),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=True),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestHardtanh(), torch.randn(1, 28, 28))


def test_ReLU6():
    class TestReLU6(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.ReLU6(inplace=True),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.ReLU6(inplace=True),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestReLU6(), torch.randn(1, 28, 28))


def test_Sigmoid():
    class TestSigmoid(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.Sigmoid(),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.Sigmoid(),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestSigmoid(), torch.randn(1, 28, 28))


def test_Tanh():
    class TestTanh(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.Tanh(),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.Tanh(),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestTanh(), torch.randn(1, 28, 28))


def test_Softmax():
    class TestSoftmax(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
            )
            self.layers2 = torch.nn.Sequential(
                torch.nn.Softmax(dim=1),
                torch.nn.Softmax(dim=2)
            )

        def forward(self, x):
            x = self.layers1(x)
            x = self.layers2(x)

            return x

    testtool.testKQI(TestSoftmax(), torch.randn(1, 28, 28))


def test_Softmax2d():
    class TestSoftmax2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
            )
            self.layers2 = torch.nn.Sequential(
                torch.nn.Softmax2d()
            )

        def forward(self, x):
            x = self.layers1(x)
            x = self.layers2(x)

            return x

    testtool.testKQI(TestSoftmax2d(), torch.randn(1, 28, 28))


def test_LogSoftmax():
    class TestLogSoftmax(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),

            )
            self.layers2 = torch.nn.Sequential(
                torch.nn.LogSoftmax(dim=1),
                torch.nn.LogSoftmax(dim=2)
            )

        def forward(self, x):
            x = self.layers1(x)
            x = self.layers2(x)

            return x

    testtool.testKQI(TestLogSoftmax(), torch.randn(1, 28, 28))


def test_ELU():
    class TestELU(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.ELU(alpha=1.0, inplace=True),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.ELU(alpha=1.0, inplace=True),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestELU(), torch.randn(1, 28, 28))


def test_SELU():
    class TestSELU(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.SELU(inplace=True),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.SELU(inplace=True),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestSELU(), torch.randn(1, 28, 28))


def test_CELU():
    class TestCELU(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.CELU(alpha=1.0, inplace=True),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.CELU(alpha=1.0, inplace=True),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestCELU(), torch.randn(1, 28, 28))


def test_GELU():
    class TestGELU(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.GELU(approximate='none'),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.GELU(approximate='none'),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.layers1.KQIforward(x)
            x = x.flatten()
            x = self.layers2.KQIforward(x)

            return x

    testtool.testKQI(TestGELU(), torch.randn(1, 28, 28))


def test_Hardshrink():
    class TestHardshrink(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.Hardshrink(lambd=0.5),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.Hardshrink(lambd=0.5),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestHardshrink(), torch.randn(1, 28, 28))


def test_LeakyReLU():
    class TestLeakyReLU(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.LeakyReLU(negative_slope=0.01, inplace=True),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.LeakyReLU(negative_slope=0.01, inplace=True),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestLeakyReLU(), torch.randn(1, 28, 28))


def test_LogSigmoid():
    class TestLogSigmoid(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.LogSigmoid(),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.LogSigmoid(),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestLogSigmoid(), torch.randn(1, 28, 28))


def test_Softplus():
    class TestSoftplus(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.Softplus(),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.Softplus(),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestSoftplus(), torch.randn(1, 28, 28))


def test_Softshrink():
    class TestSoftshrink(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.Softshrink(),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.Softshrink(),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestSoftshrink(), torch.randn(1, 28, 28))


# def test_MultiheadAttention():
#     head = 8
#     embedding_dim = 64
#     sequence_length = 10

#     class TestMultiheadAttention(torch.nn.Module):
#         def __init__(self) -> None:
#             super().__init__()

#             self.layerQKV = torch.nn.Linear(in_features=embedding_dim * sequence_length, out_features=embedding_dim * sequence_length * 3, bias=False)
#             self.layer = torch.nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=head)

#         def forward(self, x):
#             x = x.flatten()
#             qkv = self.layerQKV(x).reshape(sequence_length, embedding_dim * 3)
#             q = qkv[:, :embedding_dim].reshape(sequence_length, embedding_dim)
#             k = qkv[:, embedding_dim:embedding_dim * 2].reshape(sequence_length, embedding_dim)
#             v = qkv[:, embedding_dim * 2:].reshape(sequence_length, embedding_dim)
#             attn_output, attn_output_weights = self.layer(k, q, v)
#             return attn_output

#     testtool.testKQI(TestMultiheadAttention(), torch.randn(sequence_length, embedding_dim))


# def test_PReLU():
#     class TestPReLU(torch.nn.Module):
#         def __init__(self) -> None:
#             super().__init__()
#             self.layers1 = torch.nn.Sequential(
#                 # 1x28x28
#                 torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
#                 torch.nn.PReLU(num_parameters=1, init=0.25),
#                 # 2x26x26
#                 torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
#                 torch.nn.PReLU(num_parameters=1, init=0.25),
#             )
#             self.layers2 = torch.nn.Sequential(
#                 # 3x8x8
#                 torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
#                 torch.nn.Linear(in_features=100, out_features=10, bias=False),
#             )

#         def forward(self, x):
#             x = self.layers1(x)
#             x = x.flatten()
#             x = self.layers2(x)

#             return x

#     testtool.testKQI(TestPReLU(), torch.randn(1, 28, 28))


def test_Softsign():
    class TestSoftsign(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.Softsign(),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.Softsign(),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestSoftsign(), torch.randn(1, 28, 28))


def test_Softmin():
    class TestSoftmin(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),

            )
            self.layers2 = torch.nn.Sequential(
                torch.nn.Softmin(dim=1),
                torch.nn.Softmin(dim=2)
            )

        def forward(self, x):
            x = self.layers1(x)
            x = self.layers2(x)

            return x

    testtool.testKQI(TestSoftmin(), torch.randn(1, 28, 28))


def test_Tanhshrink():
    class TestTanhshrink(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.Tanhshrink(),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.Tanhshrink(),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestTanhshrink(), torch.randn(1, 28, 28))


def test_RReLU():
    class TestRReLU(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.RReLU(0.1, 0.3, inplace=True),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.RReLU(0.1, 0.3, inplace=True),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestRReLU(), torch.randn(1, 28, 28))


# def test_GLU():
#     class TestGLU(torch.nn.Module):
#         def __init__(self) -> None:
#             super().__init__()
#             self.layers1 = torch.nn.Sequential(
#                 # 1x28x28
#                 torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
#                 # 2x26x26
#                 torch.nn.GLU(dim=-1),
#                 # 2x26x13
#                 torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
#                 # 3x8x3
#                 torch.nn.GLU(dim=-2),
#             )
#             self.layers2 = torch.nn.Sequential(
#                 # 3x4x3
#                 torch.nn.Linear(in_features=3 * 4 * 3, out_features=10, bias=False),
#             )

#         def forward(self, x):
#             x = self.layers1(x)
#             x = x.flatten()
#             x = self.layers2(x)

#             return x

#     testtool.testKQI(TestGLU(), torch.randn(1, 28, 28))


def test_Hardsigmoid():
    class TestHardsigmoid(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.Hardsigmoid(),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.Hardsigmoid(),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestHardsigmoid(), torch.randn(1, 28, 28))


def test_Hardswish():
    class TestHardswish(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.Hardswish(inplace=True),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.Hardswish(inplace=True),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestHardswish(), torch.randn(1, 28, 28))


def test_SiLU():
    class TestSiLU(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.SiLU(inplace=True),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.SiLU(inplace=True),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestSiLU(), torch.randn(1, 28, 28))


def test_Mish():
    class TestMish(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                torch.nn.SiLU(inplace=True),
                # 2x26x26
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.SiLU(inplace=True),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8x8
                torch.nn.Linear(in_features=3 * 8 * 8, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestMish(), torch.randn(1, 28, 28))


if __name__ == '__main__':
    test_Threshold()
    test_ReLU()
    test_Hardtanh()
    test_ReLU6()
    test_Sigmoid()
    test_Tanh()

    test_Softmax()
    test_Softmax2d()
    test_LogSoftmax()
    test_ELU()
    test_SELU()
    test_CELU()
    test_GELU()
    test_Hardshrink()
    test_LeakyReLU()
    test_LogSigmoid()

    test_Softplus()
    test_Softshrink()
    # test_PReLU()
    test_Softsign()
    test_Softmin()
    test_Tanhshrink()
    test_RReLU()
    # test_GLU()

    test_Hardsigmoid()
    # test_MultiheadAttention()
    test_Hardswish()
    test_SiLU()
    test_Mish()
