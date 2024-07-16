import torch
import testtool


def test_Sequential():
    class TestSequential(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.Sequential(
                torch.nn.Linear(in_features=64, out_features=32, bias=False),
                torch.nn.Linear(in_features=32, out_features=32, bias=False),
            )
            self.layers2 = torch.nn.Sequential(
                torch.nn.Linear(in_features=32, out_features=10, bias=False),
            )

        def forward(self, x):
            x = self.layers1(x)
            x = self.layers2(x)

            return x

    testtool.testKQI(TestSequential(), torch.randn(1 * 8 * 8))


def test_ModuleList():
    class TestModuleList(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers1 = torch.nn.ModuleList([
                torch.nn.Linear(in_features=64, out_features=32, bias=False),
                torch.nn.Linear(in_features=32, out_features=32, bias=False)
            ])
            self.layers2 = torch.nn.ModuleList([
                torch.nn.Linear(in_features=32, out_features=10, bias=False)
            ])

        def forward(self, x):
            for layer in self.layers1:
                x = layer(x)
            for layer in self.layers2:
                x = layer(x)

            return x

    testtool.testKQI(TestModuleList(), torch.randn(1 * 8 * 8))


def test_ModuleDict():
    class TestModuleDict(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleDict({
                'layer1': torch.nn.Linear(in_features=64, out_features=32, bias=False),
                'layer2': torch.nn.Linear(in_features=32, out_features=32, bias=False),
                'layer3': torch.nn.Linear(in_features=32, out_features=10, bias=False),
            })

        def forward(self, x):
            x = self.layers['layer1'](x)
            x = self.layers['layer2'](x)
            x = self.layers['layer3'](x)
            return x

    testtool.testKQI(TestModuleDict(), torch.randn(1 * 8 * 8))


if __name__ == '__main__':
    test_Sequential()
    test_ModuleList()
    test_ModuleDict()
