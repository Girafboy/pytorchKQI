import torch
import kqinn
import kqitool
import logging
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
    class TestLSTM(torch.nn.Module, kqinn.KQI):
        def __init__(self):
            super().__init__()
            self.rnn = kqinn.LSTM(input_size=28, hidden_size=32, num_layers=2, bias=False)
            self.fc = kqinn.Linear(32, 10)

        def forward(self, x):
            out, _ = self.rnn(x)
            out = out[-1, :]
            out = self.fc(out)
            return out

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            self.length = x.shape[0]
            x, _ = self.rnn.KQIforward(x)
            x = x[-1, :]
            x = self.fc.KQIforward(x)
            return x

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            volume = self.fc.KQIbackward(volume)

            volume_output = torch.zeros(self.length, self.rnn.hidden_size)
            volume_output[-1] += volume
            volume_backward = self.rnn.KQIbackward(volume_output, volume_backward)

            return volume_backward

        def true_kqi(self):
            G = kqitool.DiGraph()
            input_size = 28
            hidden_size = 32
            # layer1
            for t in range(1, 4):
                for i in range(input_size):
                    G.add_node(f'x_{t}_{i}', [])

                if t == 1:
                    preds = [f'x_{t}_{k}' for k in range(input_size)]
                else:
                    preds = [f'x_{t}_{k}' for k in range(input_size)] + [f'l1_h_{t-1}_{k}' for k in range(hidden_size)]
                for i in range(hidden_size):
                    # f_t
                    G.add_node(f'l1_f_{t}_linear_{i}', preds)
                    G.add_node(f'l1_f_{t}_{i}', [f'l1_f_{t}_linear_{i}'])
                    # i_t
                    G.add_node(f'l1_i_{t}_linear_{i}', preds)
                    G.add_node(f'l1_i_{t}_{i}', [f'l1_i_{t}_linear_{i}'])
                    # C~_t
                    G.add_node(f'l1_C~{t}_linear_{i}', preds)
                    G.add_node(f'l1_C~{t}_{i}', [f'l1_C~{t}_linear_{i}'])
                    # o_t
                    G.add_node(f'l1_o_{t}_linear_{i}', preds)
                    G.add_node(f'l1_o_{t}_{i}', [f'l1_o_{t}_linear_{i}'])
                # C_t and tanh(C_t)
                for i in range(hidden_size):
                    if t == 1:
                        G.add_node(f'l1_C_{t}_tmp1_{i}', [f'l1_f_{t}_{i}'])
                    else:
                        G.add_node(f'l1_C_{t}_tmp1_{i}', [f'l1_f_{t}_{i}', f'l1_C_{t-1}_{i}'])
                    G.add_node(f'l1_C_{t}_tmp2_{i}', [f'l1_i_{t}_{i}', f'l1_C~{t}_{i}'])
                    G.add_node(f'l1_C_{t}_{i}', [f'l1_C_{t}_tmp1_{i}', f'l1_C_{t}_tmp2_{i}'])
                    G.add_node(f'l1_tanh_C_{t}_{i}', [f'l1_C_{t}_{i}'])
                # h_t
                for i in range(hidden_size):
                    G.add_node(f'l1_h_{t}_{i}', [f'l1_o_{t}_{i}', f'l1_tanh_C_{t}_{i}'])

            # layer2
            for t in range(1, 4):
                if t == 1:
                    preds = [f'l1_h_{t}_{k}' for k in range(hidden_size)]
                else:
                    preds = [f'l1_h_{t}_{k}' for k in range(hidden_size)] + [f'l2_h_{t-1}_{k}' for k in range(hidden_size)]
                for i in range(hidden_size):
                    # f_t
                    G.add_node(f'l2_f_{t}_linear_{i}', preds)
                    G.add_node(f'l2_f_{t}_{i}', [f'l2_f_{t}_linear_{i}'])
                    # i_t
                    G.add_node(f'l2_i_{t}_linear_{i}', preds)
                    G.add_node(f'l2_i_{t}_{i}', [f'l2_i_{t}_linear_{i}'])
                    # C~_t
                    G.add_node(f'l2_C~{t}_linear_{i}', preds)
                    G.add_node(f'l2_C~{t}_{i}', [f'l2_C~{t}_linear_{i}'])
                    # o_t
                    G.add_node(f'l2_o_{t}_linear_{i}', preds)
                    G.add_node(f'l2_o_{t}_{i}', [f'l2_o_{t}_linear_{i}'])
                # C_t and tanh(C_t)
                for i in range(hidden_size):
                    if t == 1:
                        G.add_node(f'l2_C_{t}_tmp1_{i}', [f'l2_f_{t}_{i}'])
                    else:
                        G.add_node(f'l2_C_{t}_tmp1_{i}', [f'l2_f_{t}_{i}', f'l2_C_{t-1}_{i}'])
                    G.add_node(f'l2_C_{t}_tmp2_{i}', [f'l2_i_{t}_{i}', f'l2_C~{t}_{i}'])
                    G.add_node(f'l2_C_{t}_{i}', [f'l2_C_{t}_tmp1_{i}', f'l2_C_{t}_tmp2_{i}'])
                    G.add_node(f'l2_tanh_C_{t}_{i}', [f'l2_C_{t}_{i}'])
                # h_t
                for i in range(hidden_size):
                    G.add_node(f'l2_h_{t}_{i}', [f'l2_o_{t}_{i}', f'l2_tanh_C_{t}_{i}'])

            for i in range(10):
                preds = [f'l2_h_3_{k}' for k in range(hidden_size)]
                G.add_node(f'out_{i}', preds)

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestLSTM().KQI(torch.randn(3, 28))
    true = TestLSTM().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


def test_GRU():
    class TestGRU(torch.nn.Module, kqinn.KQI):
        def __init__(self):
            super().__init__()
            self.rnn = kqinn.GRU(input_size=28, hidden_size=32, num_layers=2, bias=False)
            self.fc = kqinn.Linear(32, 10)

        def forward(self, x):
            out, _ = self.rnn(x)
            out = out[-1, :]
            out = self.fc(out)
            return out

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            self.length = x.shape[0]
            x, _ = self.rnn.KQIforward(x)
            x = x[-1, :]
            x = self.fc.KQIforward(x)
            return x

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            volume = self.fc.KQIbackward(volume)

            volume_output = torch.zeros(self.length, self.rnn.hidden_size)
            volume_output[-1] += volume
            volume_backward = self.rnn.KQIbackward(volume_output, volume_backward)

            return volume_backward

        def true_kqi(self):
            G = kqitool.DiGraph()
            input_size = 28
            hidden_size = 32

            for t in range(1, 4):
                for i in range(input_size):
                    G.add_node(f'x_{t}_{i}', [])
            for layer in range(1, 3):
                for t in range(1, 4):
                    x_preds = [f'x_{t}_{k}' for k in range(input_size)] if layer == 1 else [f'l{layer-1}_h_{t}_{k}' for k in range(hidden_size)]
                    preds = x_preds if t == 1 else x_preds + [f'l{layer}_h_{t-1}_{k}' for k in range(hidden_size)]

                    for i in range(hidden_size):
                        G.add_node(f'l{layer}_hr_{t}_{i}', preds)
                        G.add_node(f'l{layer}_r_{t}_{i}', [f'l{layer}_hr_{t}_{i}'])  # rt
                        G.add_node(f'l{layer}_hz_{t}_{i}', preds)
                        G.add_node(f'l{layer}_z_{t}_{i}', [f'l{layer}_hz_{t}_{i}'])  # zt
                        G.add_node(f'l{layer}_1_z_{t}_{i}', [f'l{layer}_z_{t}_{i}'])  # 1-zt

                        if t == 1:
                            G.add_node(f'l{layer}_r_{t}_hn_{i}', [f'l{layer}_r_{t}_{i}'])
                            G.add_node(f'l{layer}_h_{t}_pre_right_{i}', [f'l{layer}_z_{t}_{i}'])  # ht_pre_right
                        else:
                            G.add_node(f'l{layer}_hn_{t}_{i}', [f'l{layer}_h_{t-1}_{k}' for k in range(hidden_size)])
                            G.add_node(f'l{layer}_r_{t}_hn_{i}', [f'l{layer}_r_{t}_{i}', f'l{layer}_hn_{t}_{i}'])
                            G.add_node(f'l{layer}_h_{t}_pre_right_{i}', [f'l{layer}_z_{t}_{i}', f'l{layer}_h_{t-1}_{i}'])  # ht_pre_right

                        G.add_node(f'l{layer}_n_{t}_pre_{i}', x_preds + [f'l{layer}_r_{t}_hn_{i}'])
                        G.add_node(f'l{layer}_n_{t}_{i}', [f'l{layer}_n_{t}_pre_{i}'])  # nt

                        G.add_node(f'l{layer}_h_{t}_pre_left_{i}', [f'l{layer}_n_{t}_{i}', f'l{layer}_1_z_{t}_{i}'])  # ht_pre_left

                        G.add_node(f'l{layer}_h_{t}_{i}', [f'l{layer}_h_{t}_pre_left_{i}', f'l{layer}_h_{t}_pre_right_{i}'])  # ht

            for i in range(10):
                G.add_node(f'out_{i}', [f'l2_h_3_{k}' for k in range(hidden_size)])

            return sum(map(lambda k: G.kqi(k), G.nodes()))

    kqi = TestGRU().KQI(torch.randn(3, 28))
    true = TestGRU().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test_RNN()
    test_LSTM()
    test_GRU()
