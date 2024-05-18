import torch
import networkx as nx

import logging
from . import functions
from typing import Tuple, Iterator, Union, Dict
from itertools import zip_longest

logging.basicConfig(level=logging.DEBUG)


def __construct_compute_graph(grad_fn):
    G = nx.DiGraph()
    stack = [grad_fn]
    access = {grad_fn}
    G.add_node(grad_fn)
    while stack:
        cur = stack.pop()
        for next_fn, _ in cur.next_functions:
            if next_fn is not None:
                G.add_edge(next_fn, cur)
                if next_fn not in access:
                    stack.append(next_fn)
                    access.add(next_fn)
    del stack, access
    return G


def __intermediate_result_generator(model_output: torch.Tensor, return_graph: bool = False) -> Union[Iterator[Tuple[object, Tuple[torch.Tensor], Tuple[torch.Tensor]]], Iterator[Tuple[object, Tuple[torch.Tensor], Tuple[torch.Tensor], Tuple[torch.Tensor], Dict[int, Tuple[int]]]]]:
    grad_fn = model_output.grad_fn

    G = __construct_compute_graph(grad_fn)
    waiting = {}  # Dict[torch.autograd.graph.Node, int]
    volumes = {grad_fn: (torch.zeros_like(model_output),)}  # Dict[torch.autograd.graph.Node, Tuple[torch.Tensor]]
    garbage_counter = {}  # Dict[torch.autograd.graph.Node, int]
    if return_graph:
        increID = 1
        nodeIDs = {grad_fn: (torch.arange(increID, increID + model_output.numel(), dtype=torch.float32).reshape_as(model_output),)}  # Dict[torch.autograd.graph.Node, Tuple[torch.Tensor]]
        increID += model_output.numel()

    for cur in reversed(list(nx.topological_sort(G))):
        garbage_counter[cur] = G.out_degree(cur) + G.in_degree(cur)
        inputs = functions.backward_mapper(cur).cell_Volume(cur, volumes[cur])
        for (next_fn, i), vI in zip(cur.next_functions, inputs):
            if next_fn is not None:
                waiting[cur] = waiting.get(cur, 0) + 1
                volumes[next_fn] = tuple(v_old + vI for v_old, vI in zip_longest(volumes.get(next_fn, tuple()), (0,) * i + (vI,), fillvalue=0))
                if return_graph:
                    nodeIDs[next_fn] = tuple(v_old if v_old is not None or vI is None else torch.arange(increID, increID + vI.numel(), dtype=torch.float32).reshape_as(vI) for v_old, vI in zip_longest(nodeIDs.get(next_fn, tuple()), (None,) * i + (vI,), fillvalue=None))
                    increID += vI.numel()

        for succ in G.successors(cur):
            waiting[succ] -= 1
            if waiting[succ] == 0:
                if return_graph:
                    yield succ, functions.backward_mapper(succ).cell_KQI(succ, tuple(volumes[next_fn][i] if next_fn is not None else None for next_fn, i in succ.next_functions), volumes[succ]), volumes[succ], nodeIDs[succ], functions.backward_mapper(succ).cell_Graph(succ, tuple(nodeIDs[next_fn][i] if next_fn is not None else None for next_fn, i in succ.next_functions), nodeIDs[succ])
                else:
                    yield succ, functions.backward_mapper(succ).cell_KQI(succ, tuple(volumes[next_fn][i] if next_fn is not None else None for next_fn, i in succ.next_functions), volumes[succ]), volumes[succ]
                del waiting[succ]
                for pred in G.predecessors(succ):
                    garbage_counter[succ] -= 1
                    garbage_counter[pred] -= 1
                    if garbage_counter[pred] == 0 and G.in_degree(pred) != 0:
                        del volumes[pred]
                        del garbage_counter[pred]
                        if return_graph:
                            del nodeIDs[pred]
                if garbage_counter[succ] == 0:
                    del volumes[succ]
                    del garbage_counter[succ]
                    if return_graph:
                        del nodeIDs[succ]

    W = sum((1 + V).sum() for Vs in volumes.values() for V in Vs)
    for grad_fn, Vs in volumes.items():
        if return_graph:
            yield grad_fn, functions.backward_mapper(grad_fn).cell_KQI(grad_fn, (W,), Vs), Vs, nodeIDs[grad_fn], functions.backward_mapper(grad_fn).cell_Graph(grad_fn, tuple(), nodeIDs[grad_fn])
        else:
            yield grad_fn, functions.backward_mapper(grad_fn).cell_KQI(grad_fn, (W,), Vs), Vs


def KQI(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    for param in model.parameters():
        param.requires_grad_(False)
    x.requires_grad_()
    model_output = model(x)

    kqi = torch.tensor(0, dtype=float)
    W = torch.tensor(0, dtype=float)
    for grad_fn, ks, Vs in __intermediate_result_generator(model_output):
        kqi += sum(map(lambda k: k.sum(), ks))
        if 'torch::autograd::AccumulateGrad' in grad_fn.name():
            W += sum((1 + V).sum() for V in Vs)
    kqi /= W
    logging.debug(f'W = {W}, KQI = {kqi}')
    return kqi


def Graph(model: torch.nn.Module, x: torch.Tensor) -> Iterator[Tuple[int, Tuple[int], str, float, float]]:
    for param in model.parameters():
        param.requires_grad_(False)
    x.requires_grad_()
    model_output = model(x)

    W = sum((1 + V).sum() for grad_fn, _, Vs in __intermediate_result_generator(model_output) if 'torch::autograd::AccumulateGrad' in grad_fn.name() for V in Vs)

    for grad_fn, kqis, volumes, node_ids, adj in __intermediate_result_generator(model_output, return_graph=True):
        for kqi, volume, node_id in zip(kqis, volumes, node_ids):
            for k, v, i in zip(kqi.flatten(), volume.flatten(), node_id.flatten()):
                yield int(i), adj[int(i)], grad_fn.name(), float(k / W), float(v)
