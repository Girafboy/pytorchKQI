import torch
import testtool


# def test_LayerNorm():
#     dim_1 = 3
#     dim_2 = 4
#     dim_3 = 5

#     class TestLayerNorm(torch.nn.Module):
#         def __init__(self) -> None:
#             super().__init__()

#             self.layer1 = torch.nn.LayerNorm(normalized_shape=dim_3)
#             self.layer2 = torch.nn.Linear(in_features=dim_1 * dim_2 * dim_3, out_features=dim_1 * dim_2 * dim_3,
#                                           bias=False)
#             self.layer3 = torch.nn.LayerNorm(normalized_shape=[dim_2, dim_3])
#             self.layer4 = torch.nn.Linear(in_features=dim_1 * dim_2 * dim_3, out_features=dim_1 * dim_2 * dim_3,
#                                           bias=False)
#             self.layer5 = torch.nn.LayerNorm(normalized_shape=[dim_1, dim_2, dim_3])

#         def forward(self, x):
#             x = self.layer1(x)
#             x = self.layer2(x.flatten())
#             x = self.layer3(x.reshape(dim_1, dim_2, dim_3))
#             x = self.layer4(x.flatten())
#             x = self.layer5(x.reshape(dim_1, dim_2, dim_3))
#             return x

#     testtool.testKQI(TestLayerNorm(), torch.randn(dim_1, dim_2, dim_3))


def test_GroupNorm():
    class TestGroupNorm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = torch.nn.GroupNorm(num_groups=3, num_channels=6, affine=False)
            self.layer2 = torch.nn.Linear(in_features=1 * 6 * 5 * 5, out_features=1 * 6 * 5 * 5,
                                          bias=False)
            self.layer3 = torch.nn.GroupNorm(num_groups=2, num_channels=6, affine=False)
            self.layer4 = torch.nn.Linear(in_features=1 * 6 * 5 * 5, out_features=1 * 6 * 5 * 5,
                                          bias=False)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(1, 6, 5, 5))
            x = self.layer4(x.flatten())
            return x

    testtool.testKQI(TestGroupNorm(), torch.randn(1, 6, 5, 5))


def test_LocalResponseNorm():
    class TestLocalResponseNorm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.layer1 = torch.nn.LocalResponseNorm(3)
            self.layer2 = torch.nn.Linear(in_features=1 * 2 * 3 * 3, out_features=1 * 2 * 3 * 3, bias=False)
            self.layer3 = torch.nn.LocalResponseNorm(2)
            self.layer4 = torch.nn.Linear(in_features=1 * 2 * 3 * 3, out_features=1 * 2 * 3 * 3, bias=False)
            self.layer5 = torch.nn.LocalResponseNorm(4)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x.flatten())
            x = self.layer3(x.reshape(1, 2, 3, 3))
            x = self.layer4(x.flatten())
            x = self.layer5(x.reshape(1, 2, 3, 3))
            return x

    testtool.testKQI(TestLocalResponseNorm(), torch.randn(1, 2, 3, 3))


if __name__ == '__main__':
    # test_LayerNorm()
    test_GroupNorm()
    test_LocalResponseNorm()
