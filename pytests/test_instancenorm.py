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
            self.layer2 = torch.nn.Linear(in_features=1 * 6 * 3 * 3, out_features=1 * 6 * 3 * 3,
                                          bias=False)
            self.layer3 = torch.nn.InstanceNorm2d(6)
            self.layer4 = torch.nn.Linear(in_features=1 * 6 * 3 * 3, out_features=1 * 6 * 3 * 3,
                                          bias=False)
            self.layer5 = torch.nn.InstanceNorm2d(6)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(1, 6, 3, 3))
            x = self.layer4(x.flatten())
            x = self.layer5(x.reshape(1, 6, 3, 3))
            return x

    testtool.testKQI(TestInstanceNorm2d(), torch.randn(1, 6, 3, 3))


def test_InstanceNorm3d():
    class TestInstanceNorm3d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = torch.nn.InstanceNorm3d(3)
            self.layer2 = torch.nn.Linear(in_features=1 * 3 * 2 * 2 * 2, out_features=1 * 3 * 2 * 2 * 2,
                                          bias=False)
            self.layer3 = torch.nn.InstanceNorm3d(3)
            self.layer4 = torch.nn.Linear(in_features=1 * 3 * 2 * 2 * 2, out_features=1 * 3 * 2 * 2 * 2,
                                          bias=False)
            self.layer5 = torch.nn.InstanceNorm3d(3)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(1, 3, 2, 2, 2))
            x = self.layer4(x.flatten())
            x = self.layer5(x.reshape(1, 3, 2, 2, 2))
            return x

    testtool.testKQI(TestInstanceNorm3d(), torch.randn(1, 3, 2, 2, 2))


def test_LazyInstanceNorm1d():
    class TestLazyInstanceNorm1d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = torch.nn.LazyInstanceNorm1d()
            self.layer2 = torch.nn.Linear(in_features=1 * 6 * 10, out_features=1 * 6 * 10,
                                          bias=False)
            self.layer3 = torch.nn.LazyInstanceNorm1d()
            self.layer4 = torch.nn.Linear(in_features=1 * 6 * 10, out_features=1 * 6 * 10,
                                          bias=False)
            self.layer5 = torch.nn.LazyInstanceNorm1d()

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(1, 6, 10))
            x = self.layer4(x.flatten())
            x = self.layer5(x.reshape(1, 6, 10))
            return x

    testtool.testKQI(TestLazyInstanceNorm1d(), torch.randn(1, 6, 10))



def test_LazyInstanceNorm2d():
    class TestLazyInstanceNorm2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = torch.nn.LazyInstanceNorm2d()
            self.layer2 = torch.nn.Linear(in_features=1 * 6 * 3 * 3, out_features=1 * 6 * 3 * 3,
                                          bias=False)
            self.layer3 = torch.nn.LazyInstanceNorm2d()
            self.layer4 = torch.nn.Linear(in_features=1 * 6 * 3 * 3, out_features=1 * 6 * 3 * 3,
                                          bias=False)
            self.layer5 = torch.nn.LazyInstanceNorm2d()

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(1, 6, 3, 3))
            x = self.layer4(x.flatten())
            x = self.layer5(x.reshape(1, 6, 3, 3))
            return x

    testtool.testKQI(TestLazyInstanceNorm2d(), torch.randn(1, 6, 3, 3))


def test_LazyInstanceNorm3d():
    class TestLazyInstanceNorm3d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = torch.nn.LazyInstanceNorm3d()
            self.layer2 = torch.nn.Linear(in_features=1 * 3 * 2 * 2 * 2, out_features=1 * 3 * 2 * 2 * 2,
                                          bias=False)
            self.layer3 = torch.nn.LazyInstanceNorm3d()
            self.layer4 = torch.nn.Linear(in_features=1 * 3 * 2 * 2 * 2, out_features=1 * 3 * 2 * 2 * 2,
                                          bias=False)
            self.layer5 = torch.nn.LazyInstanceNorm3d()

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(1, 3, 2, 2, 2))
            x = self.layer4(x.flatten())
            x = self.layer5(x.reshape(1, 3, 2, 2, 2))
            return x

    testtool.testKQI(TestLazyInstanceNorm3d(), torch.randn(1, 3, 2, 2, 2))



if __name__ == '__main__':
    test_InstanceNorm1d()
    test_InstanceNorm2d()
    test_InstanceNorm3d()
    test_LazyInstanceNorm1d()
    test_LazyInstanceNorm2d()
    test_LazyInstanceNorm3d()