import torch
import testtool

def test_Upsample():
    class TestUpsample(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(48, 48)
            self.upsample1 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.upsample2 = torch.nn.Upsample(scale_factor=1.25, mode='bilinear', align_corners=False)

        def forward(self, x):
            x = self.linear(x.flatten())
            x = x.reshape(1, 3, 4, 4)
            x = self.upsample1(x)
            x = self.upsample2(x)
            return x

    testtool.testKQI(TestUpsample(), torch.randn(1, 3, 4, 4))


def test_UpsamplingBilinear2d():
    class TestUpsamplingBilinear2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(48, 48)
            self.upsample1 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
            self.upsample2 = torch.nn.UpsamplingBilinear2d(scale_factor=1.25)

        def forward(self, x):
            x = self.linear(x.flatten())
            x = x.reshape(1, 3, 4, 4)
            x = self.upsample1(x)
            x = self.upsample2(x)
            return x

    testtool.testKQI(TestUpsamplingBilinear2d(), torch.randn(1, 3, 4, 4))


def test_UpsamplingNearest2d():
    class TestUpsamplingNearest2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(48, 48)
            self.upsample1 = torch.nn.UpsamplingNearest2d(scale_factor=2)
            self.upsample2 = torch.nn.UpsamplingNearest2d(scale_factor=1.25)

        def forward(self, x):
            x = self.linear(x.flatten())
            x = x.reshape(1, 3, 4, 4)
            x = self.upsample1(x)
            x = self.upsample2(x)
            return x

    testtool.testKQI(TestUpsamplingNearest2d(), torch.randn(1, 3, 4, 4))


if __name__ == '__main__':
    test_Upsample()
    test_UpsamplingBilinear2d()
    test_UpsamplingNearest2d()
