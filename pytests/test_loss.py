import torch
import testtool


def test_L1Loss():
    class TestL1Loss(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = torch.nn.Linear(8 * 8, 25)
            self.linear2 = torch.nn.Linear(25, 10)
            self.l1_loss = torch.nn.L1Loss()

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            loss = self.l1_loss(x, torch.randn(1, 10))
            return loss

    testtool.testKQI(TestL1Loss(), torch.randn(1, 8 * 8))


def test_NLLLoss():
    class TestNLLLoss(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = torch.nn.Linear(8 * 8, 25)
            self.linear2 = torch.nn.Linear(25, 10)
            self.log_softmax = torch.nn.LogSoftmax(dim=1)
            self.nll_loss = torch.nn.NLLLoss()

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.log_softmax(x.reshape(2, 5))
            targets = torch.tensor([1, 3])
            loss = self.nll_loss(x, targets)
            return loss

    testtool.testKQI(TestNLLLoss(), torch.randn(1, 8 * 8))



if __name__ == "__main__":
    # test_L1Loss()
    test_NLLLoss()
