import torch
import testtool


def test_BatchNorm1d():
    class TestBatchNorm1d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = torch.nn.BatchNorm1d(6)
            self.layer2 = torch.nn.Linear(in_features=1 * 6 * 10, out_features=1 * 6 * 10,
                                          bias=False)
            self.layer3 = torch.nn.BatchNorm1d(6)
            self.layer4 = torch.nn.Linear(in_features=1 * 6 * 10, out_features=1 * 6 * 10,
                                          bias=False)
            self.layer5 = torch.nn.BatchNorm1d(6)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(1, 6, 10))
            x = self.layer4(x.flatten())
            x = self.layer5(x.reshape(1, 6, 10))
            return x

    testtool.testKQI(TestBatchNorm1d(), torch.randn(1, 6, 10))


def test_BatchNorm2d():
    class TestBatchNorm2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = torch.nn.BatchNorm2d(6)
            self.layer2 = torch.nn.Linear(in_features=1 * 6 * 5 * 5, out_features=1 * 6 * 5 * 5,
                                          bias=False)
            self.layer3 = torch.nn.BatchNorm2d(6)
            self.layer4 = torch.nn.Linear(in_features=1 * 6 * 5 * 5, out_features=1 * 6 * 5 * 5,
                                          bias=False)
            self.layer5 = torch.nn.BatchNorm2d(6)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(1, 6, 5, 5))
            x = self.layer4(x.flatten())
            x = self.layer5(x.reshape(1, 6, 5, 5))
            return x

    testtool.testKQI(TestBatchNorm2d(), torch.randn(1, 6, 5, 5))


def test_BatchNorm3d():
    class TestBatchNorm3d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = torch.nn.BatchNorm3d(3)
            self.layer2 = torch.nn.Linear(in_features=1 * 3 * 5 * 5 * 5, out_features=1 * 3 * 5 * 5 * 5,
                                          bias=False)
            self.layer3 = torch.nn.BatchNorm3d(3)
            self.layer4 = torch.nn.Linear(in_features=1 * 3 * 5 * 5 * 5, out_features=1 * 3 * 5 * 5 * 5,
                                          bias=False)
            self.layer5 = torch.nn.BatchNorm3d(3)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(1, 3, 5, 5, 5))
            x = self.layer4(x.flatten())
            x = self.layer5(x.reshape(1, 3, 5, 5, 5))
            return x

    testtool.testKQI(TestBatchNorm3d(), torch.randn(1, 3, 5, 5, 5))


def test_SyncBatchNorm():
    class TestSyncBatchNorm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = torch.nn.SyncBatchNorm(6)
            self.layer2 = torch.nn.Linear(in_features=1 * 6 * 5 * 5, out_features=1 * 6 * 5 * 5,
                                          bias=False)
            self.layer3 = torch.nn.SyncBatchNorm(6)
            self.layer4 = torch.nn.Linear(in_features=1 * 6 * 5 * 5, out_features=1 * 6 * 5 * 5,
                                          bias=False)
            self.layer5 = torch.nn.SyncBatchNorm(6)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(1, 6, 5, 5))
            x = self.layer4(x.flatten())
            x = self.layer5(x.reshape(1, 6, 5, 5))
            return x

    testtool.testKQI(TestSyncBatchNorm(), torch.randn(1, 6, 5, 5))


def test_LazyBatchNorm1d():
    class TestLazyBatchNorm1d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = torch.nn.LazyBatchNorm1d()
            self.layer2 = torch.nn.Linear(in_features=1 * 6 * 10, out_features=1 * 6 * 10,
                                          bias=False)
            self.layer3 = torch.nn.LazyBatchNorm1d()
            self.layer4 = torch.nn.Linear(in_features=1 * 6 * 10, out_features=1 * 6 * 10,
                                          bias=False)
            self.layer5 = torch.nn.LazyBatchNorm1d()

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(1, 6, 10))
            x = self.layer4(x.flatten())
            x = self.layer5(x.reshape(1, 6, 10))
            return x

    testtool.testKQI(TestLazyBatchNorm1d(), torch.randn(1, 6, 10))


if __name__ == '__main__':
    test_BatchNorm1d()
    test_BatchNorm2d()
    test_BatchNorm3d()
    # test_SyncBatchNorm()
    test_LazyBatchNorm1d()