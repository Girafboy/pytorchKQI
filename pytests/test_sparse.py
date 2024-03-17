import torch
import kqinn
import kqitool
import logging
import itertools


def test_Embedding():
    class TestEmbedding(torch.nn.Module, kqinn.KQI):
        def __init__(self) -> None:
            super().__init__()
            # 2*1*4
            self.layers1 = kqinn.Embedding(10, 3)
            # 2*4*3
            self.layers2 = kqinn.Linear(in_features=2 * 4 * 3, out_features=2 * 4 * 3, bias=False)

        def forward(self, x):
            x = self.layers1(x)
            x = self.layers2(x.flatten())
            x = x.reshape(2, 4, 3)

        def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.layers1.KQIforward(x)
            x = self.layers2.KQIforward(x.flatten())
            x = x.reshape(2, 4, 3)
            return x

        def KQIbackward(self, volume: torch.Tensor, volume_backward: torch.Tensor = None) -> torch.Tensor:
            volume = self.layers2.KQIbackward(volume.flatten())
            volume = self.layers1.KQIbackward(volume.reshape(2, 4, 3), volume_backward)
            return volume

        def true_kqi(self):
            G = kqitool.DiGraph()

            for i, j, k in itertools.product(range(1), range(4), range(2)):
                G.add_node(f'L0_{i}-{j}-{k}', [])

            # embedding
            for i, j, k in itertools.product(range(4), range(3), range(2)):
                preds = [f'L0_{0}-{i}-{k}']
                G.add_node(f'L1_{i}-{j}-{k}', preds)

            # linear
            preds = [f'L1_{i}-{j}-{k}' for i, j, k in itertools.product(range(4), range(3), range(2))]
            for i, j, k in itertools.product(range(4), range(3), range(2)):
                G.add_node(f'L2_{i}-{j}-{k}', preds)

            kqi = sum(map(lambda k: G.kqi(k) if "L2_" in k else 0, G.nodes()))
            logging.debug(f'L2: KQI={kqi}, node={len([k for k in G.nodes() if "L2_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L2_" in k])}')
            kqi += sum(map(lambda k: G.kqi(k) if "L1_" in k else 0, G.nodes()))
            logging.debug(f'L1: KQI={kqi}, node={len([k for k in G.nodes() if "L1_" in k])}, volume={sum([G.volume(k) for k in G.nodes() if "L1_" in k])}')
            logging.debug(f'Total volume = {G.graph_volume()}')

            return sum(map(lambda m: G.kqi(m), G.nodes()))

    kqi = TestEmbedding().KQI(torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]))
    true = TestEmbedding().true_kqi()
    logging.debug(f'KQI = {kqi} (True KQI = {true})')
    assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test_Embedding()
