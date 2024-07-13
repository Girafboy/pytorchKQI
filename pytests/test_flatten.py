import torch
import testtool

def test_Flatten():
    class TestFlatten(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = torch.nn.Linear(in_features=64, out_features=32, bias=False)
            self.linear2 = torch.nn.Linear(in_features=32, out_features=32, bias=False)
            self.linear3 = torch.nn.Linear(in_features=32, out_features=10, bias=False)
            self.flatten = torch.nn.Flatten()   

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            x = self.flatten(x)

            return x

    testtool.testKQI(TestFlatten(), torch.randn(1, 8 * 8))


def test_Unflatten():
    class TestUnflatten(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = torch.nn.Linear(in_features=64, out_features=32, bias=False)
            self.linear2 = torch.nn.Linear(in_features=32, out_features=32, bias=False)
            self.linear3 = torch.nn.Linear(in_features=32, out_features=10, bias=False)
            self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size=(2, 5))

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            x = self.unflatten(x)

            return x

    testtool.testKQI(TestUnflatten(), torch.randn(1, 8 * 8))


if __name__ == '__main__':
    test_Flatten()
    test_Unflatten()