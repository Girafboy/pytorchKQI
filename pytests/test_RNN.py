
import torch
import kqinn
import kqitool
import logging

class RNN(torch.nn.Module, kqinn.KQI):
    def init(self, input_size, hidden_size, num_layers, batch_first=False)->None:super().init():
        self.rnn=kqinn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.linear=kqinn.Linear(in_features=hidden_size, out_features=10, bias=False)

    def forward(self,x):
        output, _ = self.rnn(x)
        x = self.linear(output[:, -1, :])
        return x

    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn.KQIforward(x)
        x = self.linear.KQIforward(output[:, -1, :])
        return x

    def KQIbackward(self, volumes: torch.Tensor, kqi: float) -> (torch.Tensor, float):
        volumes, kqi = self.linear.KQIbackward(volumes, kqi)
        volumes, kqi = self.rnn.KQIbackward(volumes, kqi)
        return volumes, kqi
    def true_kqi():
        G = kqitool.DiGraph()
        for i in range(0, 784):
        G.add_node(i, [])
        for i in range(784, 784 + 512):
        G.add_node(i, list(range(0, 784)))
        for i in range(784 + 512, 784 + 512 + 512):
        G.add_node(i, list(range(784, 784 + 512)))
        for i in range(784 + 512 + 512, 784 + 512 + 512 + 10):
        G.add_node(i, list(range(784 + 512, 784 + 512 + 512)))

    returgin sum(map(lambda k: G.kqi(k), G.nodes()))
    def test():
        rnn = RNN(input_size=28, hidden_size=256, num_layers=2, batch_first=True)
        kqi = rnn.KQI(torch.randn(1, 28, 28))
        true = true_kqi()
        logging.debug(f'KQI = {kqi} (True KQI = {true})')
        assert abs(kqi - true) / true < 0.0001
if name == 'main':
    test()