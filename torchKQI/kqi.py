import torch
import networkx as nx
import numpy as np

import logging
import itertools
from . import functions, function_base
from typing import Tuple, Iterator, Union, Dict, Callable
from matplotlib import cm, colors, pyplot as plt


def __construct_compute_graph(grad_fn):
    G = nx.MultiDiGraph()
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


__W = torch.tensor(0, dtype=float)


@torch.no_grad()
def __intermediate_result_generator(model_output: torch.Tensor, return_graph: bool = False) -> Union[Iterator[Tuple[object, Tuple[torch.Tensor], Tuple[torch.Tensor]]], Iterator[Tuple[object, Tuple[torch.Tensor], Tuple[torch.Tensor], Tuple[torch.Tensor], Dict[int, Tuple[int]]]]]:
    grad_fn = model_output.grad_fn
    G = __construct_compute_graph(grad_fn)
    function_base.Context.bar.total = G.number_of_nodes() * (3 if return_graph else 2)

    waiting = {}  # Dict[torch.autograd.graph.Node, int]
    volumes = {grad_fn: (torch.zeros_like(model_output),)}  # Dict[torch.autograd.graph.Node, Tuple[torch.Tensor]]
    garbage_counter = {}  # Dict[torch.autograd.graph.Node, int]
    pending = []
    if return_graph:
        increID = 1
        nodeIDs = {grad_fn: (torch.arange(increID, increID + model_output.numel(), dtype=torch.float64).reshape_as(model_output),)}  # Dict[torch.autograd.graph.Node, Tuple[torch.Tensor]]
        increID += model_output.numel()

    for cur in reversed(list(nx.topological_sort(G))):
        garbage_counter[cur] = G.out_degree(cur) + G.in_degree(cur)
        inputs = functions.backward_mapper(cur).cell_Volume(cur, volumes[cur])
        for (next_fn, i), vI in zip(cur.next_functions, inputs):
            if next_fn is not None:
                waiting[cur] = waiting.get(cur, 0) + 1
                volumes[next_fn] = tuple(v_old + vI for v_old, vI in itertools.zip_longest(volumes.get(next_fn, tuple()), (0,) * i + (vI,), fillvalue=0))
                if return_graph:
                    nodeIDs[next_fn] = tuple(v_old if v_old is not None or vI is None else torch.arange(increID, increID + vI.numel(), dtype=torch.float64).reshape_as(vI) for v_old, vI in itertools.zip_longest(nodeIDs.get(next_fn, tuple()), (None,) * i + (vI,), fillvalue=None))
                    if any(nodeID is not None and nodeID.eq(increID).any() for nodeID in nodeIDs[next_fn]):
                        increID += vI.numel()

        for _, succ in G.out_edges(cur):
            waiting[succ] -= 1
            if waiting[succ] == 0:
                kqis = functions.backward_mapper(succ).cell_KQI(succ, tuple(volumes[next_fn][i] if next_fn is not None else None for next_fn, i in succ.next_functions), volumes[succ])
                if any(kqi.isnan().any() for kqi in kqis):
                    if return_graph:
                        pending.append((succ, kqis, volumes[succ], nodeIDs[succ], functions.backward_mapper(succ).cell_Graph(succ, tuple(nodeIDs[next_fn][i] if next_fn is not None else None for next_fn, i in succ.next_functions), nodeIDs[succ])))
                    else:
                        pending.append((succ, kqis, volumes[succ]))
                else:
                    if return_graph:
                        yield succ, kqis, volumes[succ], nodeIDs[succ], functions.backward_mapper(succ).cell_Graph(succ, tuple(nodeIDs[next_fn][i] if next_fn is not None else None for next_fn, i in succ.next_functions), nodeIDs[succ])
                    else:
                        yield succ, kqis, volumes[succ]
                del waiting[succ]
                for pred, _ in G.in_edges(succ):
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

    for grad_fn, Vs in volumes.items():
        kqis = functions.backward_mapper(grad_fn).cell_KQI(grad_fn, (), Vs)
        if any(kqi.isnan().any() for kqi in kqis):
            if return_graph:
                pending.append((grad_fn, kqis, Vs, nodeIDs[grad_fn], functions.backward_mapper(grad_fn).cell_Graph(grad_fn, tuple(), nodeIDs[grad_fn])))
            else:
                pending.append((grad_fn, kqis, Vs))
        else:
            if return_graph:
                yield grad_fn, kqis, Vs, nodeIDs[grad_fn], functions.backward_mapper(grad_fn).cell_Graph(grad_fn, tuple(), nodeIDs[grad_fn])
            else:
                yield grad_fn, kqis, Vs

    global __W
    __W = sum(K.isnan().sum() + V.masked_select(K.isnan()).sum() for _, Ks, Vs, *_ in pending for K, V in zip(Ks, Vs))
    for grad_fn, kqis, vols, *args in pending:
        yield grad_fn, tuple(kqi.masked_scatter(kqi.isnan(), torch.masked_select(functions.FB.temporary_KQI(vol, __W), kqi.isnan())) for kqi, vol in zip(kqis, vols)), vols, *args


def __prepare(model: torch.nn.Module, x: torch.Tensor, callback_func: Callable, device: Union[torch.device, Tuple[torch.device]]) -> torch.Tensor:
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch.backends.mkldnn.enabled = False
    except Exception:
        pass

    function_base.Context.bar.desc = model.__class__.__name__
    function_base.Context.bar.reset()
    function_base.Context.grad_fn_info.clear()
    function_base.Context.device = [device] if isinstance(device, torch.device) else device
    function_base.Context.init_pool()

    model.eval()
    callback_func(model, x)  # Initialize the lazy model if any

    model.requires_grad_(True)
    if isinstance(x, dict):
        for key, tensor in x.items():
            tensor.requires_grad_(False)
    else:
        x.requires_grad_(False)
    model_output = callback_func(model, x)

    for grad_fn in __construct_compute_graph(model_output.grad_fn).nodes:
        grad_fn.register_hook(function_base.Context.hook_factory(grad_fn))
    model_output.backward(model_output, retain_graph=True)
    model.zero_grad()
    return model_output


def KQI(model: torch.nn.Module, x: torch.Tensor, callback_func: Callable = lambda model, x: model(x), device: Union[torch.device, Tuple[torch.device]] = torch.device('cpu')) -> torch.Tensor:
    model_output = __prepare(model, x, callback_func, device)

    kqi = torch.tensor(0, dtype=float)
    for _, ks, _ in __intermediate_result_generator(model_output):
        kqi += sum(map(lambda k: k.sum(), ks))
    kqi /= __W
    logging.debug(f'W = {__W}, KQI = {kqi}')
    return kqi


def Graph(model: torch.nn.Module, x: torch.Tensor, callback_func: Callable = lambda model, x: model(x), device: Union[torch.device, Tuple[torch.device]] = torch.device('cpu')) -> Iterator[Tuple[int, Tuple[int], str, float, float]]:
    model_output = __prepare(model, x, callback_func, device)

    for grad_fn, kqis, volumes, node_ids, adj in __intermediate_result_generator(model_output, return_graph=True):
        for kqi, volume, node_id in zip(kqis, volumes, node_ids):
            for k, v, i in zip(kqi.flatten(), volume.flatten(), node_id.flatten()):
                yield int(i), adj[int(i)], grad_fn.name(), float(k / __W), float(v)


def KQI_generator(model: torch.nn.Module, x: torch.Tensor, callback_func: Callable = lambda model, x: model(x), device: Union[torch.device, Tuple[torch.device]] = torch.device('cpu')) -> Iterator[Tuple[object, Tuple[torch.Tensor]]]:
    model_output = __prepare(model, x, callback_func, device)
    for grad_fn, ks, _ in __intermediate_result_generator(model_output):
        yield grad_fn, ks


def VisualKQI(model: torch.nn.Module, x: torch.Tensor, callback_func: Callable = lambda model, x: model(x), device: Union[torch.device, Tuple[torch.device]] = torch.device('cpu'), filename: str = None, dots_per_unit: int = 4, fontsize=7):
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.bottom'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.right'] = False
    plt.rcParams['ytick.labelleft'] = False
    plt.rcParams['ytick.labelright'] = False
    plt.rcParams['xtick.top'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['axes.titlesize'] = fontsize
    plt.rcParams['axes.labelsize'] = fontsize
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize
    plt.rcParams['savefig.bbox'] = 'tight'
    INTERVAL = 5
    SCALE_INCH_PT = 72
    PADDING = 10

    def compact_to_2d(x):
        if x.dim() == 0:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        else:
            pad = 0
            while x.dim() > 2:
                xs = x.unbind(dim=-3)
                interval = tuple(torch.zeros(k.shape[:-1] + (int(pad / 2 + 1),) if pad % 2 == 0 else k.shape[:-2] + (int(pad / 2 + 1),) + k.shape[-1:]) * float('nan') for k in xs)
                x = torch.concat(list(itertools.chain.from_iterable(zip(xs, interval)))[:-1], -1 if pad % 2 == 0 else -2)
                pad += 1

        return x.detach().numpy()

    model_params = {var: name for name, var in model.named_parameters()}

    def get_name(grad_fn):
        if 'AccumulateGrad' in grad_fn.name():
            return model_params[grad_fn.variable]
        return grad_fn.name()

    model_output = __prepare(model, x, callback_func, device)
    G = __construct_compute_graph(model_output.grad_fn)
    kqi_min, kqi_max = np.inf, -np.inf
    for grad_fn, kqis, _ in __intermediate_result_generator(model_output):
        kqis_compact = [compact_to_2d(kqi) for kqi in kqis]
        G.nodes[grad_fn]['width'] = (sum(map(lambda k: k.shape[1], kqis_compact)) + INTERVAL * (len(kqis_compact) - 1) + PADDING * 2) / SCALE_INCH_PT
        G.nodes[grad_fn]['height'] = (max(map(lambda k: k.shape[0], kqis_compact)) + PADDING * 2) / SCALE_INCH_PT
        G.nodes[grad_fn]['shape'] = 'box'
        G.nodes[grad_fn]['fontsize'] = 9
        G.nodes[grad_fn]['label'] = get_name(grad_fn)
        G.nodes[grad_fn]['labelloc'] = 't'
        kqi_min, kqi_max = min(*map(lambda k: k.min(), kqis), kqi_min), max(*map(lambda k: k.max(), kqis), kqi_max)
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')

    x_min, x_max = min(map(lambda k: k[1][0] - G.nodes[k[0]]['width'] * SCALE_INCH_PT / 2, pos.items())), max(map(lambda k: k[1][0] + G.nodes[k[0]]['width'] * SCALE_INCH_PT / 2, pos.items()))
    y_min, y_max = min(map(lambda k: k[1][1] - G.nodes[k[0]]['height'] * SCALE_INCH_PT / 2, pos.items())), max(map(lambda k: k[1][1] + G.nodes[k[0]]['height'] * SCALE_INCH_PT / 2, pos.items()))
    plt.figure(figsize=((x_max - x_min) / SCALE_INCH_PT, (y_max - y_min) / SCALE_INCH_PT), dpi=SCALE_INCH_PT * dots_per_unit)

    posx_transform, posy_transform = lambda x: (x - x_min) / (x_max - x_min), lambda y: (y - y_min) / (y_max - y_min)

    plt.axes([0, 0, 1, 1])
    for grad_fn in G.nodes():
        for pred in G.predecessors(grad_fn):
            plt.plot([posx_transform(pos[grad_fn][0]), posx_transform(pos[pred][0])],
                     [posy_transform(pos[grad_fn][1] + G.nodes[grad_fn]['height'] * SCALE_INCH_PT / 2 + 13), posy_transform(pos[pred][1] - G.nodes[pred]['height'] * SCALE_INCH_PT / 2)], color='black')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    for grad_fn, kqis, _ in __intermediate_result_generator(model_output):
        plt.axes([posx_transform(pos[grad_fn][0] - G.nodes[grad_fn]['width'] * SCALE_INCH_PT / 2 + PADDING),
                  posy_transform(pos[grad_fn][1] - G.nodes[grad_fn]['height'] * SCALE_INCH_PT / 2 + PADDING),
                  posx_transform(G.nodes[grad_fn]['width'] * SCALE_INCH_PT - PADDING * 2 + x_min),
                  posy_transform(G.nodes[grad_fn]['height'] * SCALE_INCH_PT - PADDING * 2 + y_min)])
        plt.title(get_name(grad_fn), pad=13, bbox=dict(facecolor='white', linewidth=0, boxstyle='Square, pad=0'))
        offset = 0
        for kqi in kqis:
            kqi_compact = compact_to_2d(kqi)
            plt.axes([posx_transform(offset + pos[grad_fn][0] - G.nodes[grad_fn]['width'] * SCALE_INCH_PT / 2 + PADDING),
                      posy_transform(pos[grad_fn][1] - G.nodes[grad_fn]['height'] * SCALE_INCH_PT / 2 + PADDING),
                      posx_transform(kqi_compact.shape[1] + x_min),
                      posy_transform(kqi_compact.shape[0] + y_min)])
            plt.imshow(kqi_compact, cmap='turbo', norm=colors.Normalize(vmin=kqi_min, vmax=kqi_max))
            plt.title("$\\times$".join(map(str, kqi.shape)), pad=3, bbox=dict(facecolor='white', linewidth=0, boxstyle='Square, pad=0'))
            offset += kqi_compact.shape[1] + INTERVAL

    plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=kqi_min / __W, vmax=kqi_max / __W), cmap='turbo'), cax=plt.axes([0, -20 / (y_max - y_min), 1, 10 / (y_max - y_min)]), orientation='horizontal', fraction=1)
    plt.xlabel('KQI')

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=SCALE_INCH_PT * dots_per_unit)
