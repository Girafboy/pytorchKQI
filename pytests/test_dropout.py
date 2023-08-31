import torch
import kqinn
import kqitool
import logging


class Dropout(torch.nn.Module, kqinn.KQI):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = kqinn.Linear(in_features = 100, out_features = 100, bias=False)
        self.dropout1=torch.nn.Dropout(p=0.5)
        self.linear2 = kqinn.Linear(in_features = 100, out_features = 100, bias=False)
        self.dropout2=torch.nn.Dropout(p=0.5)
        self.linear3 = kqinn.Linear(in_features = 100, out_features = 10, bias=False)


    def forward(self, x):
        x = self.linear1(x)
        x=self.dropout1(x)
        x = self.linear2(x)
        x=self.dropout2(x)
        x = self.linear3(x)

        return x


    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1.KQIforward(x)
        x = self.linear2.KQIforward(x)
        x = self.linear3.KQIforward(x)
        
        return x
    

    def KQIbackward(self, volumes: torch.Tensor, kqi: float) -> (torch.Tensor, float):
        volumes, kqi = self.linear3.KQIbackward(volumes, kqi)
        volumes, kqi = self.linear2.KQIbackward(volumes, kqi)
        volumes, kqi = self.linear1.KQIbackward(volumes, kqi)
        
        return volumes, kqi


def true_kqi():
    G = kqitool.DiGraph()
    for i in range(0, 100):
        G.add_node(i, [])
    for i in range(100, 100+100):
        G.add_node(i, list(range(0, 100)))
    for i in range(100+100, 100+100+100):
        G.add_node(i, list(range(100, 100+100)))
    for i in range(100+100+100, 100+100+100+10):
        G.add_node(i, list(range(100+100, 100+100+100)))

    return sum(map(lambda k: G.kqi(k), G.nodes()))


def test():
    kqi = Dropout().KQI(torch.randn(1*28*28))

    true = true_kqi()
    
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test()