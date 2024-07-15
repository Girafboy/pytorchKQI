import torch
import testtool


def test_Identity():
    class TestIdentity(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(28, 10)
            self.fc2 = torch.nn.Linear(10, 5)
            self.optional_layer = torch.nn.Identity()

        def forward(self, x):
            x = self.fc1(x)
            x = self.optional_layer(x)
            x = self.fc2(x)
            return x

    testtool.testKQI(TestIdentity(), torch.randn(1, 28))


def test_Linear():
    class TestLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = torch.nn.Linear(in_features=64, out_features=32, bias=False)
            self.linear2 = torch.nn.Linear(in_features=32, out_features=32, bias=False)
            self.linear3 = torch.nn.Linear(in_features=32, out_features=10, bias=False)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)

            return x

    testtool.testKQI(TestLinear(), torch.randn(1, 8 * 8))


def test_LazyLinear():
    class TestLazyLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = torch.nn.LazyLinear(out_features=32, bias=False)
            self.linear2 = torch.nn.LazyLinear(out_features=32, bias=False)
            self.linear3 = torch.nn.LazyLinear(out_features=10, bias=False)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            return x
    
    testtool.testKQI(TestLazyLinear(), torch.randn(1, 8 * 8))


if __name__ == '__main__':
    test_Identity()
    test_Linear()
    test_LazyLinear()
