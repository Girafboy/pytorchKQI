import torch
import testtool


def test_Branch():
    class TestBranch(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = torch.nn.Linear(in_features=64, out_features=32, bias=False)
            self.linear2 = torch.nn.Linear(in_features=32, out_features=32, bias=False)
            self.linear3 = torch.nn.Linear(in_features=32, out_features=10, bias=False)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x) + x
            x = self.linear3(x)

            return x

    testtool.testKQI(TestBranch(), torch.randn(1, 8 * 8))


if __name__ == '__main__':
    test_Branch()
