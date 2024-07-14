import torch
import testtool


def test_Dropout():
    class TestDropout(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = torch.nn.Linear(in_features=64, out_features=32, bias=False)
            self.dropout1 = torch.nn.Dropout(p=0.4)
            self.linear2 = torch.nn.Linear(in_features=32, out_features=32, bias=False)
            self.dropout2 = torch.nn.Dropout(p=0.3)
            self.linear3 = torch.nn.Linear(in_features=32, out_features=10, bias=False)

        def forward(self, x):
            x = self.linear1(x)
            x = self.dropout1(x)
            x = self.linear2(x)
            x = self.dropout2(x)
            x = self.linear3(x)

            return x

    testtool.testKQI(TestDropout(), torch.randn(1 * 8 * 8))


def test_Dropout1d():
    class TestDropout1d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 3x28
                torch.nn.Conv1d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                torch.nn.Dropout1d(p=0.5),
                torch.nn.Conv1d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.Dropout1d(p=0.5),
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

    testtool.testKQI(TestDropout1d(), torch.randn(3, 28))


def test_Dropout2d():
    class TestDropout2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 3x28x28
                torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                torch.nn.Dropout2d(p=0.4),
                torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.Dropout2d(p=0.5),
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

    testtool.testKQI(TestDropout2d(), torch.randn(1, 3, 28, 28))


def test_Dropout3d():
    class TestDropout3d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 3x9x9x9
                torch.nn.Conv3d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                torch.nn.Dropout3d(p=0.4),
                torch.nn.Conv3d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.Dropout3d(p=0.5),
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

    testtool.testKQI(TestDropout3d(), torch.randn(3, 9, 9, 9))


def test_AlphaDropout():
    class TestAlphaDropout(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = torch.nn.Linear(in_features=784, out_features=256, bias=False)
            self.dropout1 = torch.nn.AlphaDropout(p=0.4)
            self.linear2 = torch.nn.Linear(in_features=256, out_features=128, bias=False)
            self.dropout2 = torch.nn.AlphaDropout(p=0.3)
            self.linear3 = torch.nn.Linear(in_features=128, out_features=10, bias=False)

        def forward(self, x):
            x = self.linear1(x)
            x = self.dropout1(x)
            x = self.linear2(x)
            x = self.dropout2(x)
            x = self.linear3(x)

            return x

    testtool.testKQI(TestAlphaDropout(), torch.randn(1 * 28 * 28))


def test_FeatureAlphaDropout():
    class TestFeatureAlphaDropout(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                # 3x9x9x9
                torch.nn.Conv3d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                torch.nn.FeatureAlphaDropout(p=0.4, inplace=True),
                torch.nn.Conv3d(in_channels=2, out_channels=3, kernel_size=3, stride=3, padding=0, dilation=2, bias=False),
                torch.nn.FeatureAlphaDropout(p=0.5, inplace=True),
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

    testtool.testKQI(TestFeatureAlphaDropout(), torch.randn(3, 9, 9, 9))


if __name__ == '__main__':
    test_Dropout()
    test_Dropout1d()
    test_Dropout2d()
    test_Dropout3d()
    test_AlphaDropout()
    test_FeatureAlphaDropout()
