import torch
import testtool


def test_InstanceNorm1d():
    class TestInstanceNorm1d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = torch.nn.InstanceNorm1d(6)
            self.layer2 = torch.nn.Linear(in_features=1 * 6 * 10, out_features=1 * 6 * 10,
                                          bias=False)
            self.layer3 = torch.nn.InstanceNorm1d(6)
            self.layer4 = torch.nn.Linear(in_features=1 * 6 * 10, out_features=1 * 6 * 10,
                                          bias=False)
            self.layer5 = torch.nn.InstanceNorm1d(6)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(1, 6, 10))
            x = self.layer4(x.flatten())
            x = self.layer5(x.reshape(1, 6, 10))
            return x

    testtool.testKQI(TestInstanceNorm1d(), torch.randn(1, 6, 10))



def test_InstanceNorm2d():
    class TestInstanceNorm2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = torch.nn.InstanceNorm2d(6)
            self.layer2 = torch.nn.Linear(in_features=1 * 6 * 5 * 5, out_features=1 * 6 * 5 * 5,
                                          bias=False)
            self.layer3 = torch.nn.InstanceNorm2d(6)
            self.layer4 = torch.nn.Linear(in_features=1 * 6 * 5 * 5, out_features=1 * 6 * 5 * 5,
                                          bias=False)
            self.layer5 = torch.nn.InstanceNorm2d(6)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(1, 6, 5, 5))
            x = self.layer4(x.flatten())
            x = self.layer5(x.reshape(1, 6, 5, 5))
            return x

    testtool.testKQI(TestInstanceNorm2d(), torch.randn(1, 6, 5, 5))


def test_InstanceNorm3d():
    class TestInstanceNorm3d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = torch.nn.InstanceNorm3d(3)
            self.layer2 = torch.nn.Linear(in_features=1 * 3 * 5 * 5 * 5, out_features=1 * 3 * 5 * 5 * 5,
                                          bias=False)
            self.layer3 = torch.nn.InstanceNorm3d(3)
            self.layer4 = torch.nn.Linear(in_features=1 * 3 * 5 * 5 * 5, out_features=1 * 3 * 5 * 5 * 5,
                                          bias=False)
            self.layer5 = torch.nn.InstanceNorm3d(3)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(1, 3, 5, 5, 5))
            x = self.layer4(x.flatten())
            x = self.layer5(x.reshape(1, 3, 5, 5, 5))
            return x

    testtool.testKQI(TestInstanceNorm3d(), torch.randn(1, 3, 5, 5, 5))


if __name__ == '__main__':
    test_InstanceNorm1d()
    test_InstanceNorm2d()
    test_InstanceNorm3d()