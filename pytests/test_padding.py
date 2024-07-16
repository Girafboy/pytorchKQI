import torch
import testtool


def test_ReflectionPad1d():
    class TestReflectionPad1d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.reflection_pad = torch.nn.ReflectionPad1d(padding=(1, 1))
            self.linear1 = torch.nn.Linear(in_features=66, out_features=32, bias=False)  # Adjusted input size
            self.linear2 = torch.nn.Linear(in_features=32, out_features=32, bias=False)
            self.linear3 = torch.nn.Linear(in_features=32, out_features=10, bias=False)

        def forward(self, x):
            x = self.reflection_pad(x)
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)

            return x

    testtool.testKQI(TestReflectionPad1d(), torch.randn(1, 8 * 8))


def test_ReflectionPad2d():
    class TestReflectionPad2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.reflection_pad = torch.nn.ReflectionPad2d(padding=(1, 1, 1, 1))
            self.linear1 = torch.nn.Linear(in_features=100, out_features=32, bias=False)  # Adjusted input size
            self.linear2 = torch.nn.Linear(in_features=32, out_features=32, bias=False)
            self.linear3 = torch.nn.Linear(in_features=32, out_features=10, bias=False)

        def forward(self, x):
            x = self.reflection_pad(x)
            x = self.linear1(x.flatten())
            x = self.linear2(x)
            x = self.linear3(x)

            return x

    testtool.testKQI(TestReflectionPad2d(), torch.randn(1, 1, 8, 8))


def test_ReflectionPad3d():
    class TestReflectionPad3d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.reflection_pad = torch.nn.ReflectionPad3d(padding=(1, 1, 1, 1, 1, 1))
            self.linear1 = torch.nn.Linear(in_features=64, out_features=32, bias=False)  # Adjusted input size
            self.linear2 = torch.nn.Linear(in_features=32, out_features=16, bias=False)
            self.linear3 = torch.nn.Linear(in_features=16, out_features=10, bias=False)

        def forward(self, x):
            x = self.reflection_pad(x)
            x = self.linear1(x.flatten())
            x = self.linear2(x)
            x = self.linear3(x)

            return x

    testtool.testKQI(TestReflectionPad3d(), torch.randn(1, 1, 2, 2, 2))


def test_ReplicationPad1d():
    class TestReplicationPad1d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.replication_pad = torch.nn.ReplicationPad1d(padding=(1, 1))
            self.linear1 = torch.nn.Linear(in_features=66, out_features=32, bias=False)  # Adjusted input size
            self.linear2 = torch.nn.Linear(in_features=32, out_features=32, bias=False)
            self.linear3 = torch.nn.Linear(in_features=32, out_features=10, bias=False)

        def forward(self, x):
            x = self.replication_pad(x)
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)

            return x

    testtool.testKQI(TestReplicationPad1d(), torch.randn(1, 8 * 8))


def test_ReplicationPad2d():
    class TestReplicationPad2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.replication_pad = torch.nn.ReplicationPad2d(padding=(1, 1, 1, 1))
            self.linear1 = torch.nn.Linear(in_features=100, out_features=32, bias=False)  # Adjusted input size
            self.linear2 = torch.nn.Linear(in_features=32, out_features=32, bias=False)
            self.linear3 = torch.nn.Linear(in_features=32, out_features=10, bias=False)

        def forward(self, x):
            x = self.replication_pad(x)
            x = self.linear1(x.flatten())
            x = self.linear2(x)
            x = self.linear3(x)

            return x

    testtool.testKQI(TestReplicationPad2d(), torch.randn(1, 1, 8, 8))


def test_ReplicationPad3d():
    class TestReplicationPad3d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.replication_pad = torch.nn.ReplicationPad3d(padding=(1, 1, 1, 1, 1, 1))
            self.linear1 = torch.nn.Linear(in_features=64, out_features=32, bias=False)  # Adjusted input size
            self.linear2 = torch.nn.Linear(in_features=32, out_features=16, bias=False)
            self.linear3 = torch.nn.Linear(in_features=16, out_features=10, bias=False)

        def forward(self, x):
            x = self.replication_pad(x)
            x = self.linear1(x.flatten())
            x = self.linear2(x)
            x = self.linear3(x)

            return x

    testtool.testKQI(TestReplicationPad3d(), torch.randn(1, 1, 2, 2, 2))


def test_ZeroPad2d():
    class TestZeroPad2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.zero_pad = torch.nn.ZeroPad2d(padding=(1, 1, 1, 1))
            self.linear1 = torch.nn.Linear(in_features=100, out_features=32, bias=False)  # Adjusted input size
            self.linear2 = torch.nn.Linear(in_features=32, out_features=32, bias=False)
            self.linear3 = torch.nn.Linear(in_features=32, out_features=10, bias=False)

        def forward(self, x):
            x = self.zero_pad(x)
            x = self.linear1(x.flatten())
            x = self.linear2(x)
            x = self.linear3(x)

            return x

    testtool.testKQI(TestZeroPad2d(), torch.randn(1, 1, 8, 8))


def test_ConstantPad1d():
    class TestConstantPad1d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.constant_pad = torch.nn.ConstantPad1d(padding=(1, 1), value=0)
            self.linear1 = torch.nn.Linear(in_features=66, out_features=32, bias=False)  # Adjusted input size
            self.linear2 = torch.nn.Linear(in_features=32, out_features=32, bias=False)
            self.linear3 = torch.nn.Linear(in_features=32, out_features=10, bias=False)

        def forward(self, x):
            x = self.constant_pad(x)
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)

            return x

    testtool.testKQI(TestConstantPad1d(), torch.randn(1, 8 * 8))


def test_ConstantPad2d():
    class TestConstantPad2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.constant_pad = torch.nn.ConstantPad2d(padding=(1, 1, 1, 1), value=0)
            self.linear1 = torch.nn.Linear(in_features=100, out_features=32, bias=False)  # Adjusted input size
            self.linear2 = torch.nn.Linear(in_features=32, out_features=32, bias=False)
            self.linear3 = torch.nn.Linear(in_features=32, out_features=10, bias=False)

        def forward(self, x):
            x = self.constant_pad(x)
            x = self.linear1(x.flatten())
            x = self.linear2(x)
            x = self.linear3(x)

            return x

    testtool.testKQI(TestConstantPad2d(), torch.randn(1, 1, 8, 8))


def test_ConstantPad3d():
    class TestConstantPad3d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.constant_pad = torch.nn.ConstantPad3d(padding=(1, 1, 1, 1, 1, 1), value=0)
            self.linear1 = torch.nn.Linear(in_features=64, out_features=32, bias=False)  # Adjusted input size
            self.linear2 = torch.nn.Linear(in_features=32, out_features=16, bias=False)
            self.linear3 = torch.nn.Linear(in_features=16, out_features=10, bias=False)

        def forward(self, x):
            x = self.constant_pad(x)
            x = self.linear1(x.flatten())
            x = self.linear2(x)
            x = self.linear3(x)

            return x

    testtool.testKQI(TestConstantPad3d(), torch.randn(1, 1, 2, 2, 2))


if __name__ == '__main__':
    test_ReflectionPad1d()
    test_ReflectionPad2d()
    test_ReflectionPad3d()
    test_ReplicationPad1d()
    test_ReplicationPad2d()
    test_ReplicationPad3d()
    test_ZeroPad2d()
    test_ConstantPad1d()
    test_ConstantPad2d()
    test_ConstantPad3d()
