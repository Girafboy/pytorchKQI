import torch
import kqinn
import kqitool
import itertools
import logging
import testtool


def test_Conv1d():
    class TestConv1d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 3x28
                torch.nn.Conv1d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                # 2x28
                torch.nn.Conv1d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x8
                torch.nn.Linear(in_features=3 * 8, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestConv1d(), torch.randn(3, 28))


def test_Conv2d():
    class TestConv2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 1x28x28
                torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                # 2x28x28
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
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

    testtool.testKQI(TestConv2d(), torch.randn(3, 28, 28))
        


def test_Conv3d():
    class TestConv3d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 3x9x9x9
                torch.nn.Conv3d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                # 2x9x9x9
                torch.nn.Conv3d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
            )
            self.layers2 = torch.nn.Sequential(
                # 3x2x2x2
                torch.nn.Linear(in_features=3 * 2 * 2 * 2, out_features=100, bias=False),
                torch.nn.Linear(in_features=100, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = x.flatten()
            x = self.layers2(x)

            return x

    testtool.testKQI(TestConv3d(), torch.randn(3, 9, 9, 9))


if __name__ == '__main__':
    test_Conv2d()
    test_Conv3d()
