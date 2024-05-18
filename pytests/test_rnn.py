import torch
import testtool


def test_RNN():
    class TestRNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.RNN(input_size=28, hidden_size=32, num_layers=2, bias=False)
            self.fc = torch.nn.Linear(32, 10)

        def forward(self, x):
            out, _ = self.rnn(x)
            out = out[-1, :]
            out = self.fc(out)
            return out

    testtool.testKQI(TestRNN(), torch.randn(3, 28))


def test_LSTM():
    class TestLSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.LSTM(input_size=28, hidden_size=32, num_layers=2, bias=False)
            self.fc = torch.nn.Linear(32, 10)

        def forward(self, x):
            out, _ = self.rnn(x)
            out = out[-1, :]
            out = self.fc(out)
            return out

    testtool.testKQI(TestLSTM(), torch.randn(3, 28))


def test_GRU():
    class TestGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.GRU(input_size=28, hidden_size=32, num_layers=2, bias=False)
            self.fc = torch.nn.Linear(32, 10)

        def forward(self, x):
            out, _ = self.rnn(x)
            out = out[-1, :]
            out = self.fc(out)
            return out

    testtool.testKQI(TestGRU(), torch.randn(3, 28))


if __name__ == '__main__':
    test_RNN()
    test_LSTM()
    test_GRU()
