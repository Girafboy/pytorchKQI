import torch
import testtool


def test_Sequential():
    class TestSequential(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                torch.nn.Linear(in_features=64, out_features=32, bias=False),
                torch.nn.Linear(in_features=32, out_features=32, bias=False),
            )
            self.layers2 = torch.nn.Sequential(
                torch.nn.Linear(in_features=32, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = self.layers2(x)

            return x

    testtool.testKQI(TestSequential(), torch.randn(1 * 8 * 8))


if __name__ == '__main__':
    test_Sequential()
