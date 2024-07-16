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


if __name__ == "__main__":
    test_L1Loss()
