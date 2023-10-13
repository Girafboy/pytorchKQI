import itertools

import torch
import kqinn
import kqitool


# class mask(torch.nn.Module, kqinn.KQI):

def true_kqi():
    G = kqitool.DiGraph()
    for i in range(50):
        G.add_node(i, [])

    for i in range(50, 100):
        G.add_node(i, [i - 50])

    return sum(map(lambda m: G.kqi(m), G.nodes()))


def test():
    # kqi = MHA().KQI(torch.randn(1 * 28 * 28))

    true = true_kqi()
    print("true_kqi: ", true)
    # logging.debug(f'KQI = {kqi} (True KQI = {true})')
    # assert abs(kqi - true) / true < 0.0001


if __name__ == '__main__':
    test()
