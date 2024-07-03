import torch
import testtool


def test_Embedding():
    class TestEmbedding(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # 2*1*4
            self.layers1 = torch.nn.Embedding(10, 3)
            # 2*4*3
            self.layers2 = torch.nn.Linear(in_features=2 * 4 * 3, out_features=2 * 4 * 3, bias=False)

        def forward(self, x):
            x = self.layers1(x)
            x = self.layers2(x.flatten())
            x = x.reshape(2, 4, 3)

    testtool.testKQI(TestEmbedding(), torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]))


if __name__ == '__main__':
    test_Embedding()
