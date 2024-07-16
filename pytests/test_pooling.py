import torch
import testtool


def test_AvgPool1d():
    class TestAvgPool1d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28
                torch.nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26
                torch.nn.AvgPool1d(kernel_size=2, stride=2, padding=1)
            )
            self.layers2 = torch.nn.Sequential(
                # 3x14
                torch.nn.Linear(in_features=3 * 14, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

    testtool.testKQI(TestAvgPool1d(), torch.randn(1, 28))


def test_AvgPool2d():
    class TestAvgPool2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26x26
                torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            )
            self.layers2 = torch.nn.Sequential(
                # 3x13x13
                torch.nn.Linear(in_features=3 * 13 * 13, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

    testtool.testKQI(TestAvgPool2d(), torch.randn(1, 28, 28))


def test_AvgPool3d():
    class TestAvgPool3d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x8x8x8
                torch.nn.Conv3d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x6x6x6
                torch.nn.AvgPool3d(kernel_size=2, stride=2, padding=1)
            )
            self.layers2 = torch.nn.Sequential(
                # 3x4x4x4
                torch.nn.Linear(in_features=3 * 4 * 4 * 4, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

    testtool.testKQI(TestAvgPool3d(), torch.randn(1, 8, 8, 8))


def test_MaxPool1d():
    class TestMaxPool1d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28
                torch.nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26
                torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=1, dilation=1)
            )
            self.layers2 = torch.nn.Sequential(
                # 3x14
                torch.nn.Linear(in_features=3 * 14, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

    testtool.testKQI(TestMaxPool1d(), torch.randn(1, 28))


def test_MaxPool2d():
    class TestMaxPool2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26x26
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1)
            )
            self.layers2 = torch.nn.Sequential(
                # 3x14x14
                torch.nn.Linear(in_features=3 * 14 * 14, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

    testtool.testKQI(TestMaxPool2d(), torch.randn(1, 28, 28))


def test_MaxPool3d():
    class TestMaxPool3d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x8x8x8
                torch.nn.Conv3d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x6x6x6
                torch.nn.MaxPool3d(kernel_size=2, stride=2, padding=1, dilation=1)
            )
            self.layers2 = torch.nn.Sequential(
                # 3x4x4x4
                torch.nn.Linear(in_features=3 * 4 * 4 * 4, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

    testtool.testKQI(TestMaxPool3d(), torch.randn(1, 8, 8, 8))


def test_AdaptiveAvgPool1d():
    class TestAdaptiveAvgPool1d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28
                torch.nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26
                torch.nn.AdaptiveAvgPool1d(output_size=13)
            )
            self.layers2 = torch.nn.Sequential(
                # 3x13
                torch.nn.Linear(in_features=3 * 13, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

    testtool.testKQI(TestAdaptiveAvgPool1d(), torch.randn(1, 28))


def test_AdaptiveAvgPool2d():
    class TestAdaptiveAvgPool2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26x26
                torch.nn.AdaptiveAvgPool2d(output_size=13)
            )
            self.layers2 = torch.nn.Sequential(
                # 3x13x13
                torch.nn.Linear(in_features=3 * 13 * 13, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

    testtool.testKQI(TestAdaptiveAvgPool2d(), torch.randn(1, 28, 28))


def test_AdaptiveAvgPool3d():
    class TestAdaptiveAvgPool3d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x8x8x8
                torch.nn.Conv3d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x6x6x6
                torch.nn.AdaptiveAvgPool3d(output_size=3)
            )
            self.layers2 = torch.nn.Sequential(
                # 3x3x3x3
                torch.nn.Linear(in_features=3 * 3 * 3 * 3, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

    testtool.testKQI(TestAdaptiveAvgPool3d(), torch.randn(1, 8, 8, 8))


def test_AdaptiveMaxPool1d():
    class TestAdaptiveMaxPool1d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28
                torch.nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26
                torch.nn.AdaptiveMaxPool1d(output_size=13)
            )
            self.layers2 = torch.nn.Sequential(
                # 3x13
                torch.nn.Linear(in_features=3 * 13, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

    testtool.testKQI(TestAdaptiveMaxPool1d(), torch.randn(1, 28))


def test_AdaptiveMaxPool2d():
    class TestAdaptiveMaxPool2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26x26
                torch.nn.AdaptiveMaxPool2d(output_size=13)
            )
            self.layers2 = torch.nn.Sequential(
                # 3x13x13
                torch.nn.Linear(in_features=3 * 13 * 13, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

    testtool.testKQI(TestAdaptiveMaxPool2d(), torch.randn(1, 28, 28))


def test_AdaptiveMaxPool3d():
    class TestAdaptiveMaxPool3d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x8x8x8
                torch.nn.Conv3d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x6x6x6
                torch.nn.AdaptiveMaxPool3d(output_size=3)
            )
            self.layers2 = torch.nn.Sequential(
                # 3x3x3x3
                torch.nn.Linear(in_features=3 * 3 * 3 * 3, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

    testtool.testKQI(TestAdaptiveMaxPool3d(), torch.randn(1, 8, 8, 8))


def test_LPPool1d():
    class TestLPPool1d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28
                torch.nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26
                torch.nn.LPPool1d(2.3, kernel_size=2, stride=2)
            )
            self.layers2 = torch.nn.Sequential(
                # 3x14
                torch.nn.Linear(in_features=3 * 13, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

    testtool.testKQI(TestLPPool1d(), torch.randn(1, 28))


def test_LPPool2d():
    class TestLPPool2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, bias=False),
                # 3x26x26
                torch.nn.LPPool2d(1.2, kernel_size=2, stride=2)
            )
            self.layers2 = torch.nn.Sequential(
                # 3x13x13
                torch.nn.Linear(in_features=3 * 13 * 13, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

    testtool.testKQI(TestLPPool2d(), torch.randn(1, 28, 28))


def test_FractionalMaxPool2d():
    class TestFractionalMaxPool2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.FractionalMaxPool2d(2, 5)
            self.layers2 = torch.nn.Linear(in_features=25, out_features=10, bias=False)

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

    testtool.testKQI(TestFractionalMaxPool2d(), torch.randn(1, 10, 10))


def test_FractionalMaxPool3d():
    class TestFractionalMaxPool3d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.FractionalMaxPool3d(kernel_size=2, output_size=5)
            self.layers2 = torch.nn.Linear(in_features=125, out_features=10, bias=False)

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)
            return x

    testtool.testKQI(TestFractionalMaxPool3d(), torch.randn(1, 10, 10, 10))


def test_MaxUnpool1d():
    class TestMaxUnpool1d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.MaxPool1d(2, return_indices=True)
            self.layers2 = torch.nn.MaxUnpool1d(kernel_size=2, stride=2)
            self.layers3 = torch.nn.Linear(in_features=28, out_features=10, bias=False)

        def forward(self, x):
            x, ind = self.layers1(x)
            x = self.layers2(x, ind)
            x = x.flatten()
            x = self.layers3(x)
            return x

    testtool.testKQI(TestMaxUnpool1d(), torch.randn(1, 28))


def test_MaxUnpool2d():
    class TestMaxUnpool2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.MaxPool2d(2, return_indices=True)
            self.layers2 = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.layers3 = torch.nn.Linear(in_features=28 * 28, out_features=100, bias=False)

        def forward(self, x):
            x, ind = self.layers1(x)
            x = self.layers2(x, ind)
            x = x.flatten()
            x = self.layers3(x)
            return x

    testtool.testKQI(TestMaxUnpool2d(), torch.randn(1, 28, 28))


def test_MaxUnpool3d():
    class TestMaxUnpool3d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.MaxPool3d(2, return_indices=True)
            self.layers2 = torch.nn.MaxUnpool3d(kernel_size=2, stride=2)
            self.layers3 = torch.nn.Linear(in_features=10 * 10 * 10, out_features=100, bias=False)

        def forward(self, x):
            x, ind = self.layers1(x)
            x = self.layers2(x, ind)
            x = x.flatten()
            x = self.layers3(x)
            return x

    testtool.testKQI(TestMaxUnpool3d(), torch.randn(1, 10, 10, 10))


if __name__ == '__main__':
    test_AvgPool1d()
    test_AvgPool2d()
    test_AvgPool3d()
    test_MaxPool1d()
    test_MaxPool2d()
    test_MaxPool3d()
    test_AdaptiveAvgPool1d()
    test_AdaptiveAvgPool2d()
    test_AdaptiveAvgPool3d()
    test_AdaptiveMaxPool1d()
    test_AdaptiveMaxPool2d()
    test_AdaptiveMaxPool3d()
    test_LPPool1d()
    test_LPPool2d()
    test_FractionalMaxPool2d()
    test_FractionalMaxPool3d()
    test_MaxUnpool1d()
    test_MaxUnpool2d()
    test_MaxUnpool3d()
