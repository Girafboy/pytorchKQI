import logging
import torch
import kqinn
import kqitool


class DropoutKQI(torch.nn.Module, kqinn.KQI):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = kqinn.Linear(in_features = 784, out_features = 512, bias=False)
        self.dropout1=kqinn.Dropout(p=0.4)
        self.linear2 = kqinn.Linear(in_features = 512, out_features = 512, bias=False)
        self.dropout2=kqinn.Dropout(p=0.3)
        self.linear3 = kqinn.Linear(in_features = 512, out_features = 10, bias=False)


    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.linear3(x)

        return x


    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1.KQIforward(x)
        x = self.dropout1.KQIforward(x)
        x = self.linear2.KQIforward(x)
        x = self.dropout2.KQIforward(x)
        x = self.linear3.KQIforward(x)
        
        return x
    

    def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
        volume = self.linear3.KQIbackward(volume)
        volume=self.dropout2.KQIbackward(volume)
        volume = self.linear2.KQIbackward(volume)
        volume=self.dropout1.KQIbackward(volume)
        volume = self.linear1.KQIbackward(volume, volume_backward)
        
        return volume


def true_kqi():
    G = kqitool.DiGraph()
    for i in range(0, 784):
        G.add_node(i, [])
    for i in range(784, 784+512):
        G.add_node(i, list(range(0, 784)))
    for i in range(784+512, 784+512+512):
        G.add_node(i, list(range(784, 784+512)))
    for i in range(784+512+512, 784+512+512+10):
        G.add_node(i, list(range(784+512, 784+512+512)))

    return sum(map(lambda k: G.kqi(k), G.nodes()))


def test():
    kqi = DropoutKQI().KQI(torch.randn(1*28*28))

    true = true_kqi()
    
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test()