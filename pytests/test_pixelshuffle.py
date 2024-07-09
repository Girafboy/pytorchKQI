import torch
import testtool


def test_PixelShuffle():
    class TestPixelShuffle(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer1 = torch.nn.Linear(in_features=1 * 4 * 5 * 5, out_features=1 * 4 * 5 * 5, bias=False)
            self.layer2 = torch.nn.Linear(in_features=1 * 4 * 5 * 5, out_features=1 * 4 * 5 * 5, bias=False)
            self.pixel_shuffle = torch.nn.PixelShuffle(upscale_factor=2)

        def forward(self, x):
            x = self.layer1(x.flatten())
            x = x.reshape(1, 4, 5, 5)
            x = self.layer2(x.flatten())
            x = x.reshape(1, 4, 5, 5)
            x = self.pixel_shuffle(x)
            return x

    testtool.testKQI(TestPixelShuffle(), torch.randn(1, 4, 5, 5))


def test_PixelUnshuffle():
    class TestPixelUnshuffle(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer1 = torch.nn.Linear(in_features=1 * 1 * 12 * 12, out_features=1 * 1 * 12 * 12, bias=False)
            self.layer2 = torch.nn.Linear(in_features=1 * 1 * 12 * 12, out_features=1 * 1 * 12 * 12, bias=False)
            self.pixel_shuffle = torch.nn.PixelUnshuffle(downscale_factor=3)

        def forward(self, x):
            x = self.layer1(x.flatten())
            x = x.reshape(1, 1, 12, 12)
            x = self.layer2(x.flatten())
            x = x.reshape(1, 1, 12, 12)
            x = self.pixel_shuffle(x)
            return x

    testtool.testKQI(TestPixelUnshuffle(), torch.randn(1, 1, 12, 12))


if __name__ == '__main__':
    # test_PixelShuffle()
    test_PixelUnshuffle()
