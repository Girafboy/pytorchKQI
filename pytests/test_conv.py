import torch
import kqinn
import kqitool
import itertools

class CNN(torch.nn.Module, kqinn.KQI):
    def __init__(self) -> None:
        super().__init__()
        self.layers1 = kqinn.Sequential(
            # 1x28x28
            kqinn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False),
            kqinn.ReLU(inplace=True),
            kqinn.MaxPool2d(kernel_size=2, stride=2),

            kqinn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
            kqinn.ReLU(inplace=True),
            kqinn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.layers2 = kqinn.Sequential(
            kqinn.Linear(in_features = 3*7*7, out_features = 100, bias=False),
            kqinn.Linear(in_features = 100, out_features = 10, bias=False),
        )


    def forward(self, x):
        x = self.layers1(x)
        x = x.flatten()
        x = self.layers2(x)

        return x


    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers1.KQIforward(x)
        x = x.flatten()
        x = self.layers2.KQIforward(x)

        return x


    def KQIbackward(self, volumes: torch.Tensor, kqi: float) -> (torch.Tensor, float):
        volumes, kqi = self.layers2.Kqi(volumes, kqi)
        volumes, kqi = self.layers1.Kqi(volumes, kqi)

        return volumes, kqi


def true_kqi():
    G = kqitool.DiGraph()
    for i,j in itertools.product(range(28), range(28)):
        G.add_node(f'1_{i}-{j}', [])
    for i,j in itertools.product(range(28), range(28)):
        preds = [f'1_{k1}-{k2}' for k1, k2 in itertools.product([i-1, i, i+1], [j-1, j, j+1]) if k1>=0 and k1<28 and k2>=0 and k2<28]
        G.add_node(f'2_{i}-{j}_1', preds)
        G.add_node(f'2_{i}-{j}_2', preds)
    for i,j in itertools.product(range(28), range(28)):
        G.add_node(f'3_{i}-{j}_1', [f'2_{i}-{j}_1'])
        G.add_node(f'3_{i}-{j}_2', [f'2_{i}-{j}_2'])
    for i,j in itertools.product(range(14), range(14)):
        G.add_node(f'4_{i}-{j}_1', [f'3_{i*2}-{j*2}_1', f'3_{i*2+1}-{j*2}_1', f'3_{i*2}-{j*2+1}_1', f'3_{i*2+1}-{j*2+1}_1'])
        G.add_node(f'4_{i}-{j}_2', [f'3_{i*2}-{j*2}_2', f'3_{i*2+1}-{j*2}_2', f'3_{i*2}-{j*2+1}_2', f'3_{i*2+1}-{j*2+1}_2'])
    
    for i,j in itertools.product(range(14), range(14)):
        preds = [f'4_{k1}-{k2}_{k3}' for k1, k2 in itertools.product([i-1, i, i+1], [j-1, j, j+1]) if k1>=0 and k1<14 and k2>=0 and k2<14 for k3 in [1,2]]
        G.add_node(f'5_{i}-{j}_1', preds)
        G.add_node(f'5_{i}-{j}_2', preds)
        G.add_node(f'5_{i}-{j}_3', preds)
    for i,j in itertools.product(range(14), range(14)):
        G.add_node(f'6_{i}-{j}_1', [f'5_{i}-{j}_1'])
        G.add_node(f'6_{i}-{j}_2', [f'5_{i}-{j}_2'])
        G.add_node(f'6_{i}-{j}_3', [f'5_{i}-{j}_3'])
    for i,j in itertools.product(range(7), range(7)):
        G.add_node(f'7_{i}-{j}_1', [f'6_{i*2}-{j*2}_1', f'6_{i*2+1}-{j*2}_1', f'6_{i*2}-{j*2+1}_1', f'6_{i*2+1}-{j*2+1}_1'])
        G.add_node(f'7_{i}-{j}_2', [f'6_{i*2}-{j*2}_2', f'6_{i*2+1}-{j*2}_2', f'6_{i*2}-{j*2+1}_2', f'6_{i*2+1}-{j*2+1}_2'])
        G.add_node(f'7_{i}-{j}_3', [f'6_{i*2}-{j*2}_3', f'6_{i*2+1}-{j*2}_3', f'6_{i*2}-{j*2+1}_3', f'6_{i*2+1}-{j*2+1}_3'])

    for i in range(100):
        preds = [f'7_{k1}-{k2}_{k3}' for k1,k2 in itertools.product(range(7), range(7)) for k3 in [1,2,3]]
        G.add_node(f'8_{i}', preds)

    for i in range(10):
        preds = [f'8_{k}' for k in range(100)]
        G.add_node(f'9_{i}', preds)

    return sum(map(lambda k: G.kqi(k), G.nodes()))


def test():
    kqi = CNN().KQI(torch.randn(1*28*28))

    true = true_kqi()
    print(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.01


if __name__ == '__main__':
    test()