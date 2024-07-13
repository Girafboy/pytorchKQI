import torch
import testtool

def test_ChannelShuffle():
    class TestChannelShuffle(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = torch.nn.Linear(in_features=64, out_features=32, bias=False)
            self.linear2 = torch.nn.Linear(in_features=32, out_features=32, bias=False)
            self.linear3 = torch.nn.Linear(in_features=32, out_features=10, bias=False)
            self.channel_shuffle = torch.nn.ChannelShuffle(groups=4)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = x.view(x.size(0), x.size(1), 1, 1)
            x = self.channel_shuffle(x)  
            x = x.view(x.size(0), -1)
            x = self.linear3(x)

            return x

    testtool.testKQI(TestChannelShuffle(), torch.randn(1, 8 * 8))

if __name__ == '__main__':
    test_ChannelShuffle()