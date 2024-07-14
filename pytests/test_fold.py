import torch
import testtool

def test_Fold():
    class TestFold(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = torch.nn.Linear(in_features=64, out_features=32, bias=False)
            self.linear2 = torch.nn.Linear(in_features=32, out_features=32, bias=False)
            self.linear3 = torch.nn.Linear(in_features=32, out_features=10, bias=False)
            self.fold = torch.nn.Fold(output_size=(5, 2), kernel_size=(1, 1))  

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            x = self.fold(x)

            return x

    testtool.testKQI(TestFold(), torch.randn(1, 8 * 8))


def test_Unfold():
    class TestUnfold(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = torch.nn.Linear(in_features=64, out_features=32, bias=False)
            self.linear2 = torch.nn.Linear(in_features=32, out_features=32, bias=False)
            self.linear3 = torch.nn.Linear(in_features=32, out_features=10, bias=False)
            self.unfold = torch.nn.Unfold(kernel_size=(2, 2), stride=2)  

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            x = x.view(1, 1, 5, 2)
            x = self.unfold(x)

            return x

    testtool.testKQI(TestUnfold(), torch.randn(1, 8 * 8))


if __name__ == '__main__':
    # test_Fold()
    test_Unfold()
