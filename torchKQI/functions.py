import torch
import ctypes
import itertools
import numpy as np
import math
from collections import defaultdict
from .function_base import FuncBase as FB
from typing import Tuple, Dict


def unsign_to_sign(val):
    if torch.rand(1).element_size() == 4:
        return ctypes.c_int32(val).value
    else:
        return ctypes.c_int64(val).value


class AccumulateGrad(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=0, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        return tuple()

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (W, ), (vO, ) = volume_inputs, volume_outputs
        kqi_vO = FB.temporary_KQI(vO, W)
        return (kqi_vO, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=0, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (vO, ) = outputs
        adj = {int(o): tuple() for o in torch.flatten(vO)}
        return adj


class TBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out,) = grad_fn(volume_outputs[0]), volume_outputs
        input = 1 + out.T
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (vI, ), (vO,) = volume_inputs, volume_outputs
        kqi_out = FB.temporary_KQI(vO, vI.T)
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (vI, ), (vO,) = inputs, outputs
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(vI), torch.flatten(vO.T))}
        return adj


class MvBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=2, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (mat, vec), (out,) = grad_fn(volume_outputs[0]), volume_outputs
        size = (out.shape[0], mat.shape[1] if mat is not None else vec.shape[0])
        if mat is not None and vec is not None:
            mat = 1 + out.unsqueeze(1).expand(size) / size[1] / 2
            vec = (1 + out.unsqueeze(1).expand(size) / size[1] / 2).sum(0)
        elif mat is not None:
            mat = 1 + out.unsqueeze(1).expand(size) / size[1]
        else:
            vec = (1 + out.unsqueeze(1).expand(size) / size[1]).sum(0)
        return (mat, vec)

    @classmethod
    @FB.cell_KQI_Checking(args_in=2, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (mat, vec), (out,) = volume_inputs, volume_outputs
        size = (out.shape[0], mat.shape[1] if mat is not None else vec.shape[0])
        if mat is not None and vec is not None:
            kqi_out = FB.temporary_KQI(out.unsqueeze(1).expand(size) / size[1] / 2, mat).sum(1)
            kqi_out += FB.temporary_KQI(out.unsqueeze(1).expand(size) / size[1] / 2, vec.unsqueeze(0).expand(size)).sum(1)
        elif mat is not None:
            kqi_out = FB.temporary_KQI(out.unsqueeze(1).expand(size) / size[1], mat).sum(1)
        else:
            kqi_out = FB.temporary_KQI(out.unsqueeze(1).expand(size) / size[1], vec.unsqueeze(0).expand(size)).sum(1)
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=2, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (mat, vec), (out,) = inputs, outputs
        if mat is not None and vec is not None:
            adj = {int(o): tuple(int(k) for k in row) + tuple(int(k) for k in vec) for row, o in zip(mat, out)}
        elif mat is not None:
            adj = {int(o): tuple(int(k) for k in row) for row, o in zip(mat, out)}
        else:
            adj = {int(o): tuple(int(k) for k in vec) for o in out}
        return adj


class MmBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=2, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (mat1, mat2), (out,) = grad_fn(volume_outputs[0]), volume_outputs
        size = (out.shape[0], mat1.shape[1] if mat1 is not None else mat2.shape[0], out.shape[1])
        if mat1 is not None and mat2 is not None:
            mat1 = (1 + out.unsqueeze(1).expand(size) / (size[1] * 2)).sum(2)
            mat2 = (1 + out.unsqueeze(1).expand(size) / (size[1] * 2)).sum(0)
        elif mat1 is not None:
            mat1 = (1 + out.unsqueeze(1).expand(size) / (size[1])).sum(2)
        else:
            mat2 = (1 + out.unsqueeze(1).expand(size) / (size[1])).sum(0)
        return (mat1, mat2)

    @classmethod
    @FB.cell_KQI_Checking(args_in=2, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (mat1, mat2), (out,) = volume_inputs, volume_outputs
        size = (out.shape[0], mat1.shape[1] if mat1 is not None else mat2.shape[0], out.shape[1])
        if mat1 is not None and mat2 is not None:
            kqi_out = FB.temporary_KQI(out.unsqueeze(1).expand(size) / (size[1] * 2), mat1.unsqueeze(2).expand(size)).sum(1)
            kqi_out += FB.temporary_KQI(out.unsqueeze(1).expand(size) / (size[1] * 2), mat2.unsqueeze(0).expand(size)).sum(1)
        elif mat1 is not None:
            kqi_out = FB.temporary_KQI(out.unsqueeze(1).expand(size) / (size[1]), mat1.unsqueeze(2).expand(size)).sum(1)
        else:
            kqi_out = FB.temporary_KQI(out.unsqueeze(1).expand(size) / (size[1]), mat2.unsqueeze(0).expand(size)).sum(1)
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=2, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (mat1, mat2), (out,) = inputs, outputs
        m, n = out.shape
        if mat1 is not None and mat2 is not None:
            adj = {int(out[r, c]): tuple(int(k) for k in mat1[r, :]) + tuple(int(k) for k in mat2[:, c]) for r in range(m) for c in range(n)}
        elif mat1 is not None:
            adj = {int(out[r, c]): tuple(int(k) for k in mat1[r, :]) for r in range(m) for c in range(n)}
        else:
            adj = {int(out[r, c]): tuple(int(k) for k in mat2[:, c]) for r in range(m) for c in range(n)}
        return adj


class OnetoOneMapping(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        input = 1 + out
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        kqi_out = FB.temporary_KQI(out, input)
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input), torch.flatten(out))}
        return adj


class CopySlices(OnetoOneMapping):
    pass


class TanhBackward0(OnetoOneMapping):
    pass


class SigmoidBackward0(OnetoOneMapping):
    pass


class GeluBackward0(OnetoOneMapping):
    pass


class HardshrinkBackward0(OnetoOneMapping):
    pass


class LogSigmoidBackward0(OnetoOneMapping):
    pass


class SoftplusBackward0(OnetoOneMapping):
    pass


class SoftshrinkBackward0(OnetoOneMapping):
    pass


class NegBackward0(OnetoOneMapping):
    pass


class HardsigmoidBackward0(OnetoOneMapping):
    pass


class AbsBackward0(OnetoOneMapping):
    pass


class CloneBackward0(OnetoOneMapping):
    pass


class TwotoOneMapping(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=2, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (left, right), (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        if left is not None and right is not None:
            left = 1 + out / 2
            right = 1 + out / 2
        elif left is not None:
            left = 1 + out
        else:
            right = 1 + out
        return (left, right)

    @classmethod
    @FB.cell_KQI_Checking(args_in=2, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (left, right), (out, ) = volume_inputs, volume_outputs
        if left is not None and right is not None:
            kqi_out = FB.temporary_KQI(out / 2, left) + FB.temporary_KQI(out / 2, right)
        elif left is not None:
            kqi_out = FB.temporary_KQI(out, left)
        else:
            kqi_out = FB.temporary_KQI(out, right)
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=2, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (left, right), (out, ) = inputs, outputs
        if left is not None and right is not None:
            adj = {int(o): (int(le), int(ri)) for le, ri, o in zip(torch.flatten(left), torch.flatten(right), torch.flatten(out))}
        elif left is not None:
            adj = {int(o): (int(le), ) for le, o in zip(torch.flatten(left), torch.flatten(out))}
        else:
            adj = {int(o): (int(ri), ) for ri, o in zip(torch.flatten(right), torch.flatten(out))}
        return adj


class AddBackward0(TwotoOneMapping):
    pass


class SubBackward0(TwotoOneMapping):
    pass


class MulBackward0(TwotoOneMapping):
    pass


class DivBackward0(TwotoOneMapping):
    pass


class SliceBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        dim, start, end, step = grad_fn.__getattribute__('_saved_dim'), grad_fn.__getattribute__('_saved_start'), grad_fn.__getattribute__('_saved_end'), grad_fn.__getattribute__('_saved_step')
        input = torch.zeros_like(input)
        input[tuple(slice(start, end, step) if i == dim else slice(None) for i in range(input.dim()))] = 1 + out
        return (input,)

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        dim, start, end, step = grad_fn.__getattribute__('_saved_dim'), grad_fn.__getattribute__('_saved_start'), grad_fn.__getattribute__('_saved_end'), grad_fn.__getattribute__('_saved_step')
        kqi_out = FB.temporary_KQI(out, input[tuple(slice(start, end, step) if i == dim else slice(None) for i in range(input.dim()))])
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        dim, start, end, step = grad_fn.__getattribute__('_saved_dim'), grad_fn.__getattribute__('_saved_start'), grad_fn.__getattribute__('_saved_end'), grad_fn.__getattribute__('_saved_step')
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input[tuple(slice(start, end, step) if i == dim else slice(None) for i in range(input.dim()))]), torch.flatten(out))}
        return adj


class SelectBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        dim, index = grad_fn.__getattribute__('_saved_dim'), unsign_to_sign(grad_fn.__getattribute__('_saved_index'))
        input = torch.zeros_like(input)
        input[tuple(index if i == dim else slice(None) for i in range(input.dim()))] = 1 + out
        return (input,)

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        dim, index = grad_fn.__getattribute__('_saved_dim'), unsign_to_sign(grad_fn.__getattribute__('_saved_index'))
        kqi_out = FB.temporary_KQI(out, input.select(dim, index))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        dim, index = grad_fn.__getattribute__('_saved_dim'), unsign_to_sign(grad_fn.__getattribute__('_saved_index'))
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input.select(dim, index)), torch.flatten(out))}
        return adj


class SqueezeBackward1(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        dim = grad_fn.__getattribute__('_saved_dim')
        input = torch.zeros_like(input)
        input = 1 + torch.unsqueeze(out, dim)
        return (input,)

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        dim = grad_fn.__getattribute__('_saved_dim')
        kqi_out = FB.temporary_KQI(out, torch.squeeze(input, dim))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        dim = grad_fn.__getattribute__('_saved_dim')
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(torch.squeeze(input, dim)), torch.flatten(out))}
        return adj


class UnsqueezeBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        dim = grad_fn.__getattribute__('_saved_dim')
        input = torch.zeros_like(input)
        input = 1 + torch.squeeze(out, dim)
        return (input,)

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        dim = grad_fn.__getattribute__('_saved_dim')
        kqi_out = FB.temporary_KQI(out, torch.unsqueeze(input, dim))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        dim = grad_fn.__getattribute__('_saved_dim')
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(torch.unsqueeze(input, dim)), torch.flatten(out))}
        return adj


class StackBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=None, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        inputs, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        dim = grad_fn.__getattribute__('_saved_dim')
        inputs = tuple(torch.zeros_like(input) + 1 + out.select(dim, index) for index, input in enumerate(inputs))
        return inputs

    @classmethod
    @FB.cell_KQI_Checking(args_in=None, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        inputs, (out, ) = volume_inputs, volume_outputs
        dim = grad_fn.__getattribute__('_saved_dim')
        kqi_out = FB.temporary_KQI(out, torch.stack(inputs, dim))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=None, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        inputs, (out, ) = inputs, outputs
        dim = grad_fn.__getattribute__('_saved_dim')
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(torch.stack(inputs, dim)), torch.flatten(out))}
        return adj


class UnbindBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=None)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, outputs = grad_fn(*volume_outputs), volume_outputs
        dim = grad_fn.__getattribute__('_saved_dim')
        input = torch.zeros_like(input) + 1 + torch.stack(outputs, dim)
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=None)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), outputs = volume_inputs, volume_outputs
        dim = grad_fn.__getattribute__('_saved_dim')
        kqi_outs = tuple(FB.temporary_KQI(o, i) for i, o in zip(torch.unbind(input, dim), outputs))
        return kqi_outs

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=None)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), outputs = inputs, outputs
        dim = grad_fn.__getattribute__('_saved_dim')
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input), torch.flatten(torch.stack(outputs, dim)))}
        return adj


class UnsafeSplitBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=None)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, outputs = grad_fn(*volume_outputs), volume_outputs
        dim = grad_fn.__getattribute__('_saved_dim')
        input = torch.zeros_like(input) + 1 + torch.cat(outputs, dim)
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=None)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), outputs = volume_inputs, volume_outputs
        dim, split_size = grad_fn.__getattribute__('_saved_dim'), grad_fn.__getattribute__('_saved_split_size')
        kqi_outs = tuple(FB.temporary_KQI(o, i) for i, o in zip(torch.split(input, split_size, dim), outputs))
        return kqi_outs

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=None)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), outputs = inputs, outputs
        dim = grad_fn.__getattribute__('_saved_dim')
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input), torch.flatten(torch.cat(outputs, dim)))}
        return adj


class ViewBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        input = 1 + out.contiguous().view_as(input)
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        kqi_out = FB.temporary_KQI(out, input.view_as(out))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input.view_as(out)), torch.flatten(out))}
        return adj


class UnsafeViewBackward0(ViewBackward0):
    pass


class ReshapeAliasBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        input = 1 + out.reshape_as(input)
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        kqi_out = FB.temporary_KQI(out, input.reshape_as(out))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input.reshape_as(out)), torch.flatten(out))}
        return adj


class AsStridedBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        size, stride, storage_offset = grad_fn.__getattribute__('_saved_size'), grad_fn.__getattribute__('_saved_stride'), grad_fn.__getattribute__('_saved_storage_offset')
        torch.zero_(input)
        torch.as_strided(input, size, stride, storage_offset).add_(1 + out)
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        size, stride, storage_offset = grad_fn.__getattribute__('_saved_size'), grad_fn.__getattribute__('_saved_stride'), grad_fn.__getattribute__('_saved_storage_offset')
        kqi_out = FB.temporary_KQI(out, torch.as_strided(input, size, stride, storage_offset))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        size, stride, storage_offset = grad_fn.__getattribute__('_saved_size'), grad_fn.__getattribute__('_saved_stride'), grad_fn.__getattribute__('_saved_storage_offset')
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(torch.as_strided(input, size, stride, storage_offset)), torch.flatten(out))}
        return adj


class ConvolutionBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=3, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, weight, bias, ), (output, ) = grad_fn(volume_outputs[0]), volume_outputs

        dilation = grad_fn.__getattribute__('_saved_dilation')
        stride = grad_fn.__getattribute__('_saved_stride')
        padding = grad_fn.__getattribute__('_saved_padding')
        kernel_size = grad_fn.__getattribute__('_saved_weight').shape[2:]
        groups = grad_fn.__getattribute__('_saved_groups')
        transposed = grad_fn.__getattribute__('_saved_transposed')
        saved_input = grad_fn.__getattribute__('_saved_input')

        ndim = output.dim() - 2
        in_channels, out_channels = saved_input.shape[1], output.shape[1]
        n_input, n_output = int(in_channels / groups), int(out_channels / groups)
        channel_input_slice = [slice(n_input * i, n_input * i + n_input) for i in range(groups)]
        channel_output_slice = [slice(n_output * i, n_output * i + n_output) for i in range(groups)]
        indexing = lambda *args: [slice(i, H * s + i, s) for i, H, s in zip(args, output.shape[2:], stride)]

        if transposed:
            raise NotImplementedError('ConvolutionBackward0 with transposed parameters is not yet implemented.')
        else:
            degree = cls.degree(input, weight, bias, saved_input, output.shape[2:], kernel_size, dilation, stride, padding, False)

            if input is not None:
                volume_padding = torch.zeros((input.shape[0], input.shape[1], *[input[0].shape[i] + 2 * padding[i - 1] for i in range(1, ndim + 1)]))
                for cin, cout in zip(channel_input_slice, channel_output_slice):
                    for offset in itertools.product(*[range(0, kernel_size[i] * dilation[i], dilation[i]) for i in range(ndim)]):
                        volume_padding[(slice(None), cin) + tuple(indexing(*offset))] += n_output + (output[0][cout] / degree / n_input).sum(dim=0)

                input = volume_padding[(slice(None), slice(None)) + tuple(slice(padding[i], None if padding[i] == 0 else -padding[i]) for i in range(ndim))].clone()

            if weight is not None:
                weight = torch.zeros_like(weight)
                for b in range(out_channels):
                    for offset in itertools.product(*[range(0, kernel_size[i]) for i in range(ndim)]):
                        left = [max(0, math.ceil((padding[d] - offset[d]) / stride[d])) for d in range(ndim)]
                        right = [min(output.shape[2:][d], math.ceil((saved_input[0].shape[d + 1] - offset[d] + padding[d]) / stride[d])) for d in range(ndim)]

                        slices = [slice(left[d], right[d]) for d in range(ndim)]
                        size = np.prod([[right[d] - left[d]] for d in range(ndim)])
                        weight[(b, slice(None)) + tuple(offset)] += (size + (output[0][b][slices] / degree[slices] / n_input).sum())

            if bias is not None:
                bias = torch.zeros_like(bias)
                for c in range(out_channels):
                    bias[c] = output[0][c].sum() + np.prod(output.shape[2:])

        return (input, weight, bias)

    @classmethod
    @FB.cell_KQI_Checking(args_in=3, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, weight, bias, ), (output, ) = volume_inputs, volume_outputs
        dilation = grad_fn.__getattribute__('_saved_dilation')
        stride = grad_fn.__getattribute__('_saved_stride')
        padding = grad_fn.__getattribute__('_saved_padding')
        kernel_size = grad_fn.__getattribute__('_saved_weight').shape[2:]
        groups = grad_fn.__getattribute__('_saved_groups')
        transposed = grad_fn.__getattribute__('_saved_transposed')
        saved_input = grad_fn.__getattribute__('_saved_input')

        ndim = output.dim() - 2
        in_channels, out_channels = saved_input.shape[1], output.shape[1]
        n_input, n_output = int(in_channels / groups), int(out_channels / groups)
        channel_input_slice = [slice(n_input * i, n_input * i + n_input) for i in range(groups)]
        channel_output_slice = [slice(n_output * i, n_output * i + n_output) for i in range(groups)]
        indexing = lambda *args: [slice(i, H * s + i, s) for i, H, s in zip(args, output.shape[2:], stride)]

        kqi_out = torch.zeros_like(output[0])
        if transposed:
            raise NotImplementedError('ConvolutionBackward0 with transposed parameters is not yet implemented.')
        else:
            degree = cls.degree(input, weight, bias, saved_input, output.shape[2:], kernel_size, dilation, stride, padding, False)
            if input is not None:

                volume_padding = torch.zeros((input.shape[1], *[input[0].shape[i] + 2 * padding[i - 1] for i in range(1, ndim + 1)]))
                tmp = torch.zeros_like(volume_padding)
                end = [None if pad == 0 else -pad for pad in padding]
                volume_padding[(slice(None), ) + tuple(slice(padding[k], end[k]) for k in range(ndim))] = input[0]

                for cin, cout in zip(channel_input_slice, channel_output_slice):
                    for co in range(cout.start, cout.stop):
                        for offset in itertools.product(*[range(0, kernel_size[i] * dilation[i], dilation[i]) for i in range(ndim)]):
                            args = [next(m for m in range(j, volume_padding.shape[k + 1], stride[k]) if m >= padding[k]) for k, j in zip(range(ndim), offset)]
                            tmp[(range(cin.start, cin.stop), ) + tuple(indexing(*offset))] = output[0][co] / degree / n_input
                            tmp[(range(cin.start, cin.stop), ) + tuple(slice(arg, end, stride) for arg, end, stride in zip(args, end, stride))] = volume_padding[(range(cin.start, cin.stop),) + tuple(slice(arg, end, stride) for arg, end, stride in zip(args, end, stride))]
                            for i in range(cin.start, cin.stop):
                                kqi_out[co] += FB.temporary_KQI((output[0][co] / degree / n_input), tmp[(i, ) + tuple(indexing(*offset))])

            if weight is not None:
                for b in range(out_channels):
                    for offset in itertools.product(*[range(0, kernel_size[i]) for i in range(ndim)]):
                        left = [max(0, math.ceil((padding[d] - offset[d]) / stride[d])) for d in range(ndim)]
                        right = [min(output.shape[2:][d], math.ceil((saved_input[0].shape[d + 1] - offset[d] + padding[d]) / stride[d])) for d in range(ndim)]
                        slices = [slice(left[d], right[d]) for d in range(ndim)]
                        for i in range(weight.shape[1]):
                            kqi_out[b][slices] += FB.temporary_KQI(output[0][b][slices] / degree[slices] / n_input, torch.ones(output[0][b][slices].shape) * weight[(b, i) + tuple(offset)])

            if bias is not None:
                for c in range(out_channels):
                    kqi_out += FB.temporary_KQI(output[0][c] / degree / n_input, torch.ones(output[0][c].shape) * bias[c])

            return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=3, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, weight, bias, ), (output, ) = inputs, outputs
        dilation = grad_fn.__getattribute__('_saved_dilation')
        stride = grad_fn.__getattribute__('_saved_stride')
        padding = grad_fn.__getattribute__('_saved_padding')
        kernel_size = grad_fn.__getattribute__('_saved_weight').shape[2:]
        groups = grad_fn.__getattribute__('_saved_groups')
        transposed = grad_fn.__getattribute__('_saved_transposed')
        saved_input = grad_fn.__getattribute__('_saved_input')

        in_channels, out_channels = saved_input.shape[1], output.shape[1]
        n_input, n_output = int(in_channels / groups), int(out_channels / groups)
        ndim = output.dim() - 2
        channel_input_slice = [slice(n_input * i, n_input * i + n_input) for i in range(groups)]
        channel_output_slice = [slice(n_output * i, n_output * i + n_output) for i in range(groups)]
        indexing = lambda *args: [slice(max(0, i - pad), H * s + i - pad, s) for i, pad, H, s in zip(args, padding, output.shape[2:], stride)]

        adj = defaultdict(tuple)
        if transposed:
            raise NotImplementedError('ConvolutionBackward0 with transposed parameters is not yet implemented.')
        else:
            if input is not None:
                for cin, cout in zip(channel_input_slice, channel_output_slice):
                    for ci, co in itertools.product(range(cin.start, cin.stop), range(cout.start, cout.stop)):
                        for offset in itertools.product(*[range(0, kernel_size[i] * dilation[i], dilation[i]) for i in range(ndim)]):
                            left = [max(0, math.ceil((padding[d] - offset[d]) / stride[d])) for d in range(ndim)]
                            right = [min(output.shape[2:][d], math.ceil((input[0].shape[d + 1] - offset[d] + padding[d]) / stride[d])) for d in range(ndim)]
                            for i, o in zip(torch.flatten(input[(slice(None), ci) + tuple(indexing(*offset))]), torch.flatten(output[0][co][[slice(left[d], right[d]) for d in range(ndim)]])):
                                adj[int(o)] += (int(i),)

            if weight is not None:
                for b in range(out_channels):
                    for offset in itertools.product(*[range(0, kernel_size[i]) for i in range(ndim)]):
                        left = [max(0, math.ceil((padding[d] - offset[d]) / stride[d])) for d in range(ndim)]
                        right = [min(output.shape[2:][d], math.ceil((saved_input[0].shape[d + 1] - offset[d] + padding[d]) / stride[d])) for d in range(ndim)]

                        for i, o in itertools.product(torch.flatten(weight[(b, slice(None)) + tuple(offset)]), torch.flatten(output[0][b][[slice(left[d], right[d]) for d in range(ndim)]])):
                            adj[int(o)] += (int(i),)

            if bias is not None:
                for c in range(out_channels):
                    for i, o in zip(bias[c], torch.flatten(output[c])):
                        adj[int(o)] += (int(i),)
        return adj

    @classmethod
    def degree(cls, input, weight, bias, saved_input, degree_size, kernel_size, dilation, stride, padding, transposed):
        degree = torch.zeros(degree_size)
        ndim = len(degree_size)

        if transposed:
            degree = torch.nn.functional.pad(degree, padding)
            if input is not None:
                for offset in itertools.product(*[range(0, kernel_size[d] * dilation[d], dilation[d]) for d in range(ndim)]):
                    degree[tuple(slice(offset, input[0].shape[i + 1] * stride + offset, stride) for i, stride in enumerate(stride))] += 1
            if weight is not None:
                for offset in itertools.product(*[range(0, kernel_size[d] * dilation[d], dilation[d]) for d in range(ndim)]):
                    degree[tuple(slice(offset, input[0].shape[i + 1] * stride + offset, stride) for i, stride in enumerate(stride))] += 1
            if bias is not None:
                degree += 1

            degree = degree[(slice(None), ) + tuple(slice(padding[i], None if padding[i] == 0 else -padding[i]) for i in range(ndim))]
        else:
            if input is not None:
                for offset in itertools.product(*[range(0, kernel_size[d] * dilation[d], dilation[d]) for d in range(ndim)]):
                    left = [max(0, math.ceil((padding[d] - offset[d]) / stride[d])) for d in range(ndim)]
                    right = [min(degree_size[d], math.ceil((input[0].shape[d + 1] - offset[d] + padding[d]) / stride[d])) for d in range(ndim)]
                    degree[[slice(left[d], right[d]) for d in range(ndim)]] += 1
            if weight is not None:
                for offset in itertools.product(*[range(0, kernel_size[d] * dilation[d], dilation[d]) for d in range(ndim)]):
                    left = [max(0, math.ceil((padding[d] - offset[d]) / stride[d])) for d in range(ndim)]
                    right = [min(degree_size[d], math.ceil((saved_input[0].shape[d + 1] - offset[d] + padding[d]) / stride[d])) for d in range(ndim)]
                    degree[[slice(left[d], right[d]) for d in range(ndim)]] += 1
            if bias is not None:
                degree += 1

            return degree


class SoftmaxBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        dim = unsign_to_sign(grad_fn.__getattribute__('_saved_dim'))
        input = torch.zeros_like(input)
        input = torch.mean(out, dim, True).expand_as(out) + out.size(dim)
        return (input,)

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        dim = unsign_to_sign(grad_fn.__getattribute__('_saved_dim'))
        kqi_out = sum(FB.temporary_KQI(out.select(dim, id).unsqueeze(dim).expand_as(out) / out.size(dim), input) for id in range(out.size(dim)))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        dim = unsign_to_sign(grad_fn.__getattribute__('_saved_dim'))
        adj = {int(o): tuple(int(i) for i in ii) for id in range(out.size(dim)) for ii, o in zip(torch.flatten(torch.stack(input.unbind(dim), -1), 0, -2), torch.flatten(out.select(dim, id)))}
        return adj


class LogSoftmaxBackward0(SoftmaxBackward0):
    pass


class PreluBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=2, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, weight), (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        if input is not None:
            input = 1 + out / 2
        if weight is not None:
            weight = grad_fn.__getattribute__('_saved_weight')
            weight = torch.zeros_like(weight)
            if weight.shape[2] == 1:
                weight[0][0] = np.prod(input.shape) + (out / 2).sum()
            else:
                weight[0][0] = np.prod(input.shape[:2]) + (out / 2).sum(dim=(0, 1))

        return (input, weight)

    @classmethod
    @FB.cell_KQI_Checking(args_in=2, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, weight), (out, ) = volume_inputs, volume_outputs
        kqi_out = torch.zeros_like(out)
        if input is not None:
            kqi_out += FB.temporary_KQI(out / 2, input)
        if weight is not None:
            kqi_out += FB.temporary_KQI(out / 2, weight.detach().expand_as(out))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=2, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, weight), (out, ) = inputs, outputs
        adj = defaultdict(tuple)
        if input is not None:
            for i, o in zip(torch.flatten(input), torch.flatten(out)):
                adj[int(o)] += (int(i),)
        if weight is not None:
            if weight.shape[2] == 1:
                for i, o in itertools.product(torch.flatten(weight), torch.flatten(out)):
                    adj[int(o)] += (int(i),)
            else:
                for i in torch.flatten(weight):
                    for o in torch.flatten(out[:, :, i]):
                        adj[int(o)] += (int(i),)
        return adj


class GluBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        dim = unsign_to_sign(grad_fn.__getattribute__('_saved_dim'))
        input_half = 1 + out / 2
        input = torch.cat((input_half, input_half), dim=dim)
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        dim = unsign_to_sign(grad_fn.__getattribute__('_saved_dim'))
        input_left, input_right = torch.chunk(input, 2, dim=dim)
        kqi_out = FB.temporary_KQI(out / 2, input_left) + FB.temporary_KQI(out / 2, input_right)
        return (kqi_out,)

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        dim = unsign_to_sign(grad_fn.__getattribute__('_saved_dim'))
        input_left, input_right = torch.chunk(input, 2, dim=dim)
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input_left.reshape_as(out)), torch.flatten(out))}
        for i, o in zip(torch.flatten(input_right.reshape_as(out)), torch.flatten(out)):
            adj[int(o)] += (int(i),)
        return adj


class AddmmBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=3, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, mat1, mat2), (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        size = (out.shape[0], mat1.shape[1] if mat1 is not None else mat2.shape[0], out.shape[1])
        if input is not None:
            input = 1 + out / (size[1] * 2 + 1)
        if mat1 is not None and mat2 is not None:
            mat1 = (1 + out.unsqueeze(1).expand(size) / (size[1] * 2 + 1)).sum(2)
            mat2 = (1 + out.unsqueeze(1).expand(size) / (size[1] * 2 + 1)).sum(0)
        elif mat1 is not None:
            mat1 = (1 + out.unsqueeze(1).expand(size) / (size[1] + 1)).sum(2)
        else:
            mat2 = (1 + out.unsqueeze(1).expand(size) / (size[1] + 1)).sum(0)
        return (input, mat1, mat2)

    @classmethod
    @FB.cell_KQI_Checking(args_in=3, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, mat1, mat2), (out, ) = volume_inputs, volume_outputs
        size = (out.shape[0], mat1.shape[1] if mat1 is not None else mat2.shape[0], out.shape[1])
        kqi_out = torch.zeros_like(out)
        if input is not None:
            kqi_out += FB.temporary_KQI(out / (size[1] * 2 + 1), input)
        if mat1 is not None and mat2 is not None:
            kqi_out += FB.temporary_KQI(out.unsqueeze(1).expand(size) / (size[1] * 2 + 1), mat1.unsqueeze(2).expand(size)).sum(1)
            kqi_out += FB.temporary_KQI(out.unsqueeze(1).expand(size) / (size[1] * 2 + 1), mat2.unsqueeze(0).expand(size)).sum(1)
        elif mat1 is not None:
            kqi_out += FB.temporary_KQI(out.unsqueeze(1).expand(size) / (size[1] + 1), mat1.unsqueeze(2).expand(size)).sum(1)
        else:
            kqi_out += FB.temporary_KQI(out.unsqueeze(1).expand(size) / (size[1] + 1), mat2.unsqueeze(0).expand(size)).sum(1)
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=3, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, mat1, mat2), (out,) = inputs, outputs
        m, n = out.shape
        adj = defaultdict(tuple)
        if input is not None:
            for i, o in zip(torch.flatten(input), torch.flatten(out)):
                adj[int(o)] += (int(i),)
        if mat1 is not None and mat2 is not None:
            for r, c in itertools.product(range(m), range(n)):
                adj[int(out[r, c])] += tuple(int(k) for k in mat1[r, :]) + tuple(int(k) for k in mat2[:, c])
        elif mat1 is not None:
            for r, c in itertools.product(range(m), range(n)):
                adj[int(out[r, c])] += tuple(int(k) for k in mat1[r, :])
        else:
            for r, c in itertools.product(range(m), range(n)):
                adj[int(out[r, c])] += tuple(int(k) for k in mat2[:, c])
        return adj


class TransposeBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out,) = grad_fn(volume_outputs[0]), volume_outputs
        dim0, dim1 = unsign_to_sign(grad_fn.__getattribute__('_saved_dim0')), unsign_to_sign(grad_fn.__getattribute__('_saved_dim1'))
        input = 1 + out.transpose(dim0, dim1)
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (vI, ), (vO,) = volume_inputs, volume_outputs
        dim0, dim1 = unsign_to_sign(grad_fn.__getattribute__('_saved_dim0')), unsign_to_sign(grad_fn.__getattribute__('_saved_dim1'))
        kqi_out = FB.temporary_KQI(vO, vI.transpose(dim0, dim1))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (vI, ), (vO,) = inputs, outputs
        dim0, dim1 = unsign_to_sign(grad_fn.__getattribute__('_saved_dim0')), unsign_to_sign(grad_fn.__getattribute__('_saved_dim1'))
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(vI), torch.flatten(vO.transpose(dim0, dim1)))}
        return adj


class BmmBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=2, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (mat1, mat2), (out,) = grad_fn(volume_outputs[0]), volume_outputs
        size = (out.shape[0], out.shape[1], mat1.shape[2] if mat1 is not None else mat2.shape[1], out.shape[2])
        if mat1 is not None and mat2 is not None:
            mat1 = (1 + out.unsqueeze(2).expand(size) / (size[2] * 2)).sum(3)
            mat2 = (1 + out.unsqueeze(2).expand(size) / (size[2] * 2)).sum(1)
        elif mat1 is not None:
            mat1 = (1 + out.unsqueeze(2).expand(size) / size[2]).sum(3)
        else:
            mat2 = (1 + out.unsqueeze(2).expand(size) / size[2]).sum(1)
        return (mat1, mat2)

    @classmethod
    @FB.cell_KQI_Checking(args_in=2, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (mat1, mat2), (out,) = volume_inputs, volume_outputs
        size = (out.shape[0], out.shape[1], mat1.shape[2] if mat1 is not None else mat2.shape[1], out.shape[2])
        if mat1 is not None and mat2 is not None:
            kqi_out = FB.temporary_KQI(out.unsqueeze(2).expand(size) / (size[2] * 2), mat1.unsqueeze(3).expand(size)).sum(2)
            kqi_out += FB.temporary_KQI(out.unsqueeze(2).expand(size) / (size[2] * 2), mat2.unsqueeze(1).expand(size)).sum(2)
        elif mat1 is not None:
            kqi_out = FB.temporary_KQI(out.unsqueeze(2).expand(size) / size[2], mat1.unsqueeze(3).expand(size)).sum(2)
        else:
            kqi_out = FB.temporary_KQI(out.unsqueeze(2).expand(size) / size[2], mat2.unsqueeze(1).expand(size)).sum(2)
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=2, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (mat1, mat2), (out,) = inputs, outputs
        batch, m, n = out.shape
        if mat1 is not None and mat2 is not None:
            adj = {int(out[b, r, c]): tuple(int(k) for k in mat1[b, r, :]) + tuple(int(k) for k in mat2[b, :, c]) for r in range(m) for c in range(n) for b in range(batch)}
        elif mat1 is not None:
            adj = {int(out[b, r, c]): tuple(int(k) for k in mat1[b, r, :]) for r in range(m) for c in range(n) for b in range(batch)}
        else:
            adj = {int(out[b, r, c]): tuple(int(k) for k in mat2[b, :, c]) for r in range(m) for c in range(n) for b in range(batch)}
        return adj


class SplitBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=None)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, outputs = grad_fn(*volume_outputs), volume_outputs
        dim = grad_fn.__getattribute__('_saved_dim')
        input = torch.zeros_like(input) + 1 + torch.cat(outputs, dim)
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=None)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), outputs = volume_inputs, volume_outputs
        dim = grad_fn.__getattribute__('_saved_dim')
        kqi_outs = tuple(FB.temporary_KQI(o, i) for i, o in zip(torch.split(input, input.shape[dim] // len(outputs), dim), outputs))
        return kqi_outs

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=None)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), outputs = inputs, outputs
        dim = grad_fn.__getattribute__('_saved_dim')
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input), torch.flatten(torch.cat(outputs, dim)))}
        return adj


__functions_mapping = {
    'torch::autograd::AccumulateGrad': AccumulateGrad,
    'struct torch::autograd::AccumulateGrad': AccumulateGrad,
    'torch::autograd::CopySlices': CopySlices,
    'struct torch::autograd::CopySlices': CopySlices,
    'TBackward0': TBackward0,
    'MvBackward0': MvBackward0,
    'MmBackward0': MmBackward0,
    'MulBackward0': MulBackward0,
    'TanhBackward0': TanhBackward0,
    'SigmoidBackward0': SigmoidBackward0,
    'AddBackward0': AddBackward0,
    'SubBackward0': SubBackward0,
    'SliceBackward0': SliceBackward0,
    'SelectBackward0': SelectBackward0,
    'SqueezeBackward1': SqueezeBackward1,
    'UnsqueezeBackward0': UnsqueezeBackward0,
    'StackBackward0': StackBackward0,
    'UnbindBackward0': UnbindBackward0,
    'UnsafeSplitBackward0': UnsafeSplitBackward0,
    'ViewBackward0': ViewBackward0,
    'UnsafeViewBackward0': UnsafeViewBackward0,
    'ReshapeAliasBackward0': ReshapeAliasBackward0,
    'AsStridedBackward0': AsStridedBackward0,
    'ConvolutionBackward0': ConvolutionBackward0,
    'SqueezeBackward4': SqueezeBackward1,
    'SoftmaxBackward0': SoftmaxBackward0,
    'LogSoftmaxBackward0': LogSoftmaxBackward0,
    'GeluBackward0': GeluBackward0,
    'HardshrinkBackward0': HardshrinkBackward0,
    'LogSigmoidBackward0': LogSigmoidBackward0,
    'SoftplusBackward0': SoftplusBackward0,
    'SoftshrinkBackward0': SoftshrinkBackward0,
    'DivBackward0': DivBackward0,
    'NegBackward0': NegBackward0,
    'HardsigmoidBackward0': HardsigmoidBackward0,
    'AbsBackward0': AbsBackward0,
    'PreluKernelBackward0': PreluBackward0,
    'PreluBackward0': PreluBackward0,
    'GluBackward0': GluBackward0,
    'AddmmBackward0': AddmmBackward0,
    'TransposeBackward0': TransposeBackward0,
    'CloneBackward0': CloneBackward0,
    'BmmBackward0': BmmBackward0,
    'SplitBackward0': SplitBackward0,
}


def backward_mapper(grad_fn):
    if grad_fn.name() not in __functions_mapping:
        try:
            grad_fn.__call__()
        except TypeError as err:
            args_out = int(err.args[0].split(' ')[1])
        args_in = len(grad_fn.next_functions)
        attrs = {attr: grad_fn.__getattribute__(attr) for attr in dir(grad_fn) if "_saved_" in attr}
        raise NotImplementedError(f'Class {grad_fn.name()} is not registered.',
                                  f'{grad_fn.name()} should have {args_in} inputs and {args_out} outputs.',
                                  f'{grad_fn.name()} has following attributes to be used: {attrs}')
    return __functions_mapping[grad_fn.name()]
