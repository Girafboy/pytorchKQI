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
    @FB.cell_KQI_Checking(args_in=0, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (vO, ) = volume_outputs
        return (vO * float('nan'), )

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
            mat1 = size[2] + out.unsqueeze(1).sum(2).expand((size[0], size[1])) / (size[1] * 2)
            mat2 = size[0] + out.unsqueeze(1).sum(0).expand((size[1], size[2])) / (size[1] * 2)
        elif mat1 is not None:
            mat1 = size[2] + out.unsqueeze(1).sum(2).expand((size[0], size[1])) / (size[1])
        else:
            mat2 = size[0] + out.unsqueeze(1).sum(0).expand((size[1], size[2])) / (size[1])
        return (mat1, mat2)

    @classmethod
    @FB.cell_KQI_Checking(args_in=2, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (mat1, mat2), (out,) = volume_inputs, volume_outputs
        size = (out.shape[0], mat1.shape[1] if mat1 is not None else mat2.shape[0], out.shape[1])
        kqi_out = torch.zeros_like(out)
        if mat1 is not None and mat2 is not None:
            for i in range(size[1]):
                kqi_out += FB.temporary_KQI(out / (size[1] * 2), mat1[:, i:i + 1].expand_as(out))
                kqi_out += FB.temporary_KQI(out / (size[1] * 2), mat2[i:i + 1, :].expand_as(out))
        elif mat1 is not None:
            for i in range(size[1]):
                kqi_out += FB.temporary_KQI(out / (size[1]), mat1[:, i:i + 1].expand_as(out))
        else:
            for i in range(size[1]):
                kqi_out += FB.temporary_KQI(out / (size[1]), mat2[i:i + 1, :].expand_as(out))
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


class ReluBackward0(OnetoOneMapping):
    pass


class PowBackward0(OnetoOneMapping):
    pass


class SignBackward0(OnetoOneMapping):
    pass


class SqrtBackward0(OnetoOneMapping):
    pass


class AddBackward1(OnetoOneMapping):
    pass


class HardtanhBackward0(OnetoOneMapping):
    pass


class DivBackward1(OnetoOneMapping):
    pass


class HardswishBackward0(OnetoOneMapping):
    pass


class SiluBackward0(OnetoOneMapping):
    pass


class ClampBackward1(OnetoOneMapping):
    pass


class ExpBackward0(OnetoOneMapping):
    pass


class ClampMinBackward0(OnetoOneMapping):
    pass


class RsqrtBackward0(OnetoOneMapping):
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


class TwotoOneMapping_Broadcast(FB):
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
        if left is not None and grad_fn.__getattribute__('_saved_self') is not None and left.shape != grad_fn.__getattribute__('_saved_self').shape:
            left = left.sum_to_size(grad_fn.__getattribute__('_saved_self').shape)
        if right is not None and grad_fn.__getattribute__('_saved_other') is not None and right.shape != grad_fn.__getattribute__('_saved_other').shape:
            right = right.sum_to_size(grad_fn.__getattribute__('_saved_other').shape)
        return (left, right)

    @classmethod
    @FB.cell_KQI_Checking(args_in=2, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (left, right), (out, ) = volume_inputs, volume_outputs
        if left is not None and right is not None:
            kqi_out = FB.temporary_KQI(out / 2, left.broadcast_to(out.shape)) + FB.temporary_KQI(out / 2, right.broadcast_to(out.shape))
        elif left is not None:
            kqi_out = FB.temporary_KQI(out, left.broadcast_to(out.shape))
        else:
            kqi_out = FB.temporary_KQI(out, right.broadcast_to(out.shape))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=2, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (left, right), (out, ) = inputs, outputs
        if left is not None and right is not None:
            adj = {int(o): (int(le), int(ri)) for le, ri, o in zip(torch.flatten(left.broadcast_to(out.shape)), torch.flatten(right.broadcast_to(out.shape)), torch.flatten(out))}
        elif left is not None:
            adj = {int(o): (int(le), ) for le, o in zip(torch.flatten(left.broadcast_to(out.shape)), torch.flatten(out))}
        else:
            adj = {int(o): (int(ri), ) for ri, o in zip(torch.flatten(right.broadcast_to(out.shape)), torch.flatten(out))}
        return adj


class AddBackward0(TwotoOneMapping):
    pass


class SubBackward0(TwotoOneMapping):
    pass


class MulBackward0(TwotoOneMapping_Broadcast):
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
        dim = unsign_to_sign(grad_fn.__getattribute__('_saved_dim'))
        input = torch.zeros_like(input)
        input = 1 + torch.unsqueeze(out, dim)
        return (input,)

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        dim = unsign_to_sign(grad_fn.__getattribute__('_saved_dim'))
        kqi_out = FB.temporary_KQI(out, torch.squeeze(input, dim))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        dim = unsign_to_sign(grad_fn.__getattribute__('_saved_dim'))
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(torch.squeeze(input, dim)), torch.flatten(out))}
        return adj


class UnsqueezeBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        dim = unsign_to_sign(grad_fn.__getattribute__('_saved_dim'))
        input = (1 + out).sum(dim=dim)
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        dim = unsign_to_sign(grad_fn.__getattribute__('_saved_dim'))
        kqi_out = FB.temporary_KQI(out, input.unsqueeze(dim).expand_as(out))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        dim = unsign_to_sign(grad_fn.__getattribute__('_saved_dim'))
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input.unsqueeze(dim).expand_as(out)), torch.flatten(out))}
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


class CatBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=None, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        inputs, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        inputs = inputs if type(inputs) is tuple else (inputs, )
        dim = unsign_to_sign(grad_fn.__getattribute__('_saved_dim'))
        inputs = tuple(1 + input for input in out.split([input.shape[dim] for input in inputs], dim))
        return inputs

    @classmethod
    @FB.cell_KQI_Checking(args_in=None, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        inputs, (out, ) = volume_inputs, volume_outputs
        dim = unsign_to_sign(grad_fn.__getattribute__('_saved_dim'))
        kqi_out = FB.temporary_KQI(out, torch.cat(inputs, dim))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=None, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        inputs, (out, ) = inputs, outputs
        dim = unsign_to_sign(grad_fn.__getattribute__('_saved_dim'))
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(torch.cat(inputs, dim)), torch.flatten(out))}
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


class AsStridedBackward1(AsStridedBackward0):
    pass


class ConvolutionBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=3, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, weight, bias, ), (output, ) = grad_fn(volume_outputs[0]), volume_outputs

        dilation, stride, padding, kernel_size, groups, transposed, saved_input = grad_fn.__getattribute__('_saved_dilation'), grad_fn.__getattribute__('_saved_stride'), grad_fn.__getattribute__('_saved_padding'), grad_fn.__getattribute__('_saved_weight').shape[2:], grad_fn.__getattribute__('_saved_groups'), grad_fn.__getattribute__('_saved_transposed'), grad_fn.__getattribute__('_saved_input')

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
                volume_padding = torch.zeros((input.shape[0], input.shape[1], *[input.shape[2 + i] + 2 * padding[i] for i in range(ndim)]))
                for cin, cout in zip(channel_input_slice, channel_output_slice):
                    for offset in itertools.product(*[range(0, kernel_size[i] * dilation[i], dilation[i]) for i in range(ndim)]):
                        volume_padding[(slice(None), cin) + tuple(indexing(*offset))] += n_output + (output[slice(None), cout] / degree / n_input).sum(dim=1).unsqueeze(1)

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

        dilation, stride, padding, kernel_size, groups, transposed, saved_input = grad_fn.__getattribute__('_saved_dilation'), grad_fn.__getattribute__('_saved_stride'), grad_fn.__getattribute__('_saved_padding'), grad_fn.__getattribute__('_saved_weight').shape[2:], grad_fn.__getattribute__('_saved_groups'), grad_fn.__getattribute__('_saved_transposed'), grad_fn.__getattribute__('_saved_input')

        ndim = output.dim() - 2
        in_channels, out_channels = saved_input.shape[1], output.shape[1]
        n_input, n_output = int(in_channels / groups), int(out_channels / groups)
        channel_input_slice = [slice(n_input * i, n_input * i + n_input) for i in range(groups)]
        channel_output_slice = [slice(n_output * i, n_output * i + n_output) for i in range(groups)]
        indexing = lambda *args: tuple(slice(i, H * s + i, s) for i, H, s in zip(args, output.shape[2:], stride))

        kqi_out = torch.zeros_like(output)
        if transposed:
            raise NotImplementedError('ConvolutionBackward0 with transposed parameters is not yet implemented.')
        else:
            degree = cls.degree(input, weight, bias, saved_input, output.shape[2:], kernel_size, dilation, stride, padding, False)

            if input is not None:
                volume_padding = torch.zeros((input.shape[0], input.shape[1], *[input.shape[2 + i] + 2 * padding[i] for i in range(ndim)]))
                tmp = torch.zeros_like(volume_padding)
                end = [None if pad == 0 else -pad for pad in padding]
                volume_padding[(slice(None), slice(None)) + tuple(slice(padding[k], None if padding[k] == 0 else -padding[k]) for k in range(ndim))] = input

                for cin, cout in zip(channel_input_slice, channel_output_slice):
                    for offset in itertools.product(*[range(0, kernel_size[i] * dilation[i], dilation[i]) for i in range(ndim)]):
                        args = [next(m for m in range(j, volume_padding.shape[k + 2], stride[k]) if m >= padding[k]) for k, j in zip(range(ndim), offset)]
                        for co in range(cout.start, cout.stop):
                            tmp[(slice(None), cin) + indexing(*offset)] = output[slice(None), co] / degree / n_input
                            tmp[(slice(None), cin) + tuple(slice(arg, end, stride) for arg, end, stride in zip(args, end, stride))] = volume_padding[(slice(None), cin) + tuple(slice(arg, end, stride) for arg, end, stride in zip(args, end, stride))]
                            kqi_out[slice(None), co] += FB.temporary_KQI((output[slice(None), slice(co, co + 1)] / degree / n_input).expand((-1, n_input) + (-1,) * ndim), tmp[(slice(None), cin) + indexing(*offset)]).sum(1)

            if weight is not None:
                for offset in itertools.product(*[range(0, kernel_size[i]) for i in range(ndim)]):
                    left = [max(0, math.ceil((padding[d] - offset[d]) / stride[d])) for d in range(ndim)]
                    right = [min(output.shape[2:][d], math.ceil((saved_input[0].shape[d + 1] - offset[d] + padding[d]) / stride[d])) for d in range(ndim)]
                    slices = tuple(slice(left[d], right[d]) for d in range(ndim))
                    for co in range(out_channels):
                        tmp = output[(slice(None), slice(co, co + 1)) + slices].expand((-1, n_input) + (-1,) * ndim)
                        kqi_out[(slice(None), co) + slices] += FB.temporary_KQI(tmp / degree[slices] / n_input, weight[(slice(co, co + 1), slice(None)) + tuple(slice(k, k + 1) for k in offset)].expand_as(tmp)).sum(1)

            if bias is not None:
                for c in range(out_channels):
                    kqi_out += FB.temporary_KQI(output[0][c] / degree / n_input, torch.ones(output[0][c].shape) * bias[c])

            return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=3, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, weight, bias, ), (output, ) = inputs, outputs

        dilation, stride, padding, kernel_size, groups, transposed, saved_input = grad_fn.__getattribute__('_saved_dilation'), grad_fn.__getattribute__('_saved_stride'), grad_fn.__getattribute__('_saved_padding'), grad_fn.__getattribute__('_saved_weight').shape[2:], grad_fn.__getattribute__('_saved_groups'), grad_fn.__getattribute__('_saved_transposed'), grad_fn.__getattribute__('_saved_input')

        in_channels, out_channels = saved_input.shape[1], output.shape[1]
        n_input, n_output = int(in_channels / groups), int(out_channels / groups)
        ndim = output.dim() - 2
        channel_input_slice = [slice(n_input * i, n_input * i + n_input) for i in range(groups)]
        channel_output_slice = [slice(n_output * i, n_output * i + n_output) for i in range(groups)]
        indexing = lambda *args: [slice(next(k - padding[j] for k in range(i, saved_input.shape[j + 2] + 2 * padding[j], stride[j]) if k >= padding[j]), output.shape[j + 2] * stride[j] + i - padding[j], stride[j]) for i, j in zip(args, range(ndim))]

        adj = defaultdict(list)
        if transposed:
            raise NotImplementedError('ConvolutionBackward0 with transposed parameters is not yet implemented.')
        else:
            if input is not None:
                for cin, cout in zip(channel_input_slice, channel_output_slice):
                    for ci, co in itertools.product(range(cin.start, cin.stop), range(cout.start, cout.stop)):
                        for offset in itertools.product(*[range(0, kernel_size[i] * dilation[i], dilation[i]) for i in range(ndim)]):
                            left = [max(0, math.ceil((padding[d] - offset[d]) / stride[d])) for d in range(ndim)]
                            right = [min(output.shape[2:][d], math.ceil((input[0].shape[d + 1] - offset[d] + padding[d]) / stride[d])) for d in range(ndim)]
                            for i, o in zip(torch.flatten(input[(slice(None), ci) + tuple(indexing(*offset))]), torch.flatten(output[(slice(None), co) + tuple(slice(left[d], right[d]) for d in range(ndim))])):
                                adj[int(o)].append(int(i))

            if weight is not None:
                for b in range(out_channels):
                    for offset in itertools.product(*[range(0, kernel_size[i]) for i in range(ndim)]):
                        left = [max(0, math.ceil((padding[d] - offset[d]) / stride[d])) for d in range(ndim)]
                        right = [min(output.shape[2:][d], math.ceil((saved_input[0].shape[d + 1] - offset[d] + padding[d]) / stride[d])) for d in range(ndim)]

                        for i, o in itertools.product(torch.flatten(weight[(b, slice(None)) + tuple(offset)]), torch.flatten(output[(slice(None), b) + tuple(slice(left[d], right[d]) for d in range(ndim))])):
                            adj[int(o)].append(int(i))

            if bias is not None:
                for c in range(out_channels):
                    for i, o in zip(bias[c:c + 1].broadcast_to(output[c:c + 1].shape), torch.flatten(output[c:c + 1])):
                        adj[int(o)].append(int(i))
        return {k: tuple(v) for k, v in adj.items()}

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


class RepeatBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        size = grad_fn.__getattribute__('_saved_repeats')
        num = np.prod(size)
        input = num + out.view(num, *input.shape).sum(dim=0)
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        size = grad_fn.__getattribute__('_saved_repeats')
        kqi_out = FB.temporary_KQI(out, input.repeat(size))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        size = grad_fn.__getattribute__('_saved_repeats')
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input.repeat(size)), torch.flatten(out))}
        return adj


class PreluKernelBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=2, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, weight), (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        if input is not None and weight is not None:
            input = 1 + out / 2
            weight = grad_fn.__getattribute__('_saved_weight')
            assert len(weight.shape) >= 3, "actual weight.shape " + str(weight.shape)
            weight = torch.zeros_like(weight)
            if weight.shape[2] == 1:
                weight[0][0] = np.prod(input.shape) + (out / 2).sum()
            else:
                weight[0][0] = np.prod(input.shape[:2]) + (out / 2).sum(dim=(0, 1))
        elif input is not None:
            input = 1 + out
        else:
            weight = grad_fn.__getattribute__('_saved_weight')
            assert len(weight.shape) >= 3, "actual weight.shape " + str(weight.shape)
            weight = torch.zeros_like(weight)
            if weight.shape[2] == 1:
                weight[0][0] = np.prod(input.shape) + out.sum()
            else:
                weight[0][0] = np.prod(input.shape[:2]) + out.sum(dim=(0, 1))

        return (input, weight)

    @classmethod
    @FB.cell_KQI_Checking(args_in=2, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, weight), (out, ) = volume_inputs, volume_outputs
        kqi_out = torch.zeros_like(out)
        if input is not None and weight is not None:
            kqi_out += FB.temporary_KQI(out / 2, input)
            kqi_out += FB.temporary_KQI(out / 2, weight.detach().expand_as(out))
        elif input is not None:
            kqi_out += FB.temporary_KQI(out, input)
        else:
            kqi_out += FB.temporary_KQI(out, weight.detach().expand_as(out))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=2, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, weight), (out, ) = inputs, outputs
        adj = defaultdict(list)
        if input is not None:
            for i, o in zip(torch.flatten(input), torch.flatten(out)):
                adj[int(o)].append(int(i))
        if weight is not None:
            if weight.shape[2] == 1:
                for i, o in itertools.product(torch.flatten(weight), torch.flatten(out)):
                    adj[int(o)].append(int(i))
            else:
                for i in torch.flatten(weight):
                    for o in torch.flatten(out[:, :, i]):
                        adj[int(o)].append(int(i))
        return {k: tuple(v) for k, v in adj.items()}


class PreluBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=2, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, weight), (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        if input is not None and weight is not None:
            input = 1 + out / 2
            weight = grad_fn.__getattribute__('_saved_weight')
            weight = torch.zeros_like(weight)
            if weight.shape[0] == 1:
                weight[0] = np.prod(input.shape) + (out / 2).sum()
            else:
                weight[0] = np.prod(input.shape[:2]) + (out / 2).sum(dim=(0, 1))
        elif input is not None:
            input = 1 + out
        else:
            weight = grad_fn.__getattribute__('_saved_weight')
            weight = torch.zeros_like(weight)
            if weight.shape[0] == 1:
                weight[0] = np.prod(input.shape) + out.sum()
            else:
                weight[0] = np.prod(input.shape[:2]) + out.sum(dim=(0, 1))

        return (input, weight)

    @classmethod
    @FB.cell_KQI_Checking(args_in=2, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, weight), (out, ) = volume_inputs, volume_outputs
        kqi_out = torch.zeros_like(out)
        if input is not None and weight is not None:
            kqi_out += FB.temporary_KQI(out / 2, input)
            kqi_out += FB.temporary_KQI(out / 2, weight.detach().expand_as(out))
        elif input is not None:
            kqi_out += FB.temporary_KQI(out, input)
        else:
            kqi_out += FB.temporary_KQI(out, weight.detach().expand_as(out))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=2, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, weight), (out, ) = inputs, outputs
        adj = defaultdict(list)
        if input is not None:
            for i, o in zip(torch.flatten(input), torch.flatten(out)):
                adj[int(o)].append(int(i))
        if weight is not None:
            if weight.shape[0] == 1:
                for i, o in itertools.product(torch.flatten(weight), torch.flatten(out)):
                    adj[int(o)].append(int(i))
            else:
                for i in torch.flatten(weight):
                    for o in torch.flatten(out[:, :, i]):
                        adj[int(o)].append(int(i))
        return {k: tuple(v) for k, v in adj.items()}


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
        degree = cls.degree(input, mat1, mat2, size)
        if input is not None:
            input = 1 + out / degree
        if mat1 is not None:
            mat1 = size[2] + out.unsqueeze(1).sum(2).expand((size[0], size[1])) / degree
        if mat2 is not None:
            mat2 = size[0] + out.unsqueeze(1).sum(0).expand((size[1], size[2])) / degree
        return (input, mat1, mat2)

    @classmethod
    @FB.cell_KQI_Checking(args_in=3, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, mat1, mat2), (out, ) = volume_inputs, volume_outputs
        size = (out.shape[0], mat1.shape[1] if mat1 is not None else mat2.shape[0], out.shape[1])
        degree = cls.degree(input, mat1, mat2, size)
        kqi_out = torch.zeros_like(out)
        if input is not None:
            kqi_out += FB.temporary_KQI(out / degree, input)
        if mat1 is not None:
            for i in range(size[1]):
                kqi_out += FB.temporary_KQI(out / degree, mat1[:, i:i + 1].expand_as(out))
        if mat2 is not None:
            for i in range(size[1]):
                kqi_out += FB.temporary_KQI(out / degree, mat2[i:i + 1, :].expand_as(out))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=3, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, mat1, mat2), (out,) = inputs, outputs
        m, n = out.shape
        adj = defaultdict(list)
        if input is not None:
            for i, o in zip(torch.flatten(input), torch.flatten(out)):
                adj[int(o)].append(int(i))
        if mat1 is not None and mat2 is not None:
            for r, c in itertools.product(range(m), range(n)):
                adj[int(out[r, c])].extend(int(k) for k in mat1[r, :])
                adj[int(out[r, c])].extend(int(k) for k in mat2[:, c])
        elif mat1 is not None:
            for r, c in itertools.product(range(m), range(n)):
                adj[int(out[r, c])].extend(int(k) for k in mat1[r, :])
        else:
            for r, c in itertools.product(range(m), range(n)):
                adj[int(out[r, c])].extend(int(k) for k in mat2[:, c])
        return {k: tuple(v) for k, v in adj.items()}

    @classmethod
    def degree(cls, input, mat1, mat2, size):
        degree = 0
        if input is not None:
            degree += 1
        if mat1 is not None:
            degree += size[1]
        if mat2 is not None:
            degree += size[1]
        return degree


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
            mat1 = size[3] + out.unsqueeze(2).sum(3).expand(size[0], size[1], size[2]) / (size[2] * 2)
            mat2 = size[1] + out.unsqueeze(2).sum(1).expand(size[0], size[2], size[3]) / (size[2] * 2)
        elif mat1 is not None:
            mat1 = size[3] + out.unsqueeze(2).sum(3).expand(size[0], size[1], size[2]) / size[2]
        else:
            mat2 = size[1] + out.unsqueeze(2).sum(1).expand(size[0], size[2], size[3]) / size[2]
        return (mat1, mat2)

    @classmethod
    @FB.cell_KQI_Checking(args_in=2, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (mat1, mat2), (out,) = volume_inputs, volume_outputs
        size = (out.shape[0], out.shape[1], mat1.shape[2] if mat1 is not None else mat2.shape[1], out.shape[2])
        kqi_out = torch.zeros_like(out)
        if mat1 is not None and mat2 is not None:
            for i in range(size[2]):
                kqi_out += FB.temporary_KQI(out / (size[2] * 2), mat1[:, :, i:i + 1].expand_as(out))
                kqi_out += FB.temporary_KQI(out / (size[2] * 2), mat2[:, i:i + 1, :].expand_as(out))
        elif mat1 is not None:
            for i in range(size[2]):
                kqi_out += FB.temporary_KQI(out / size[2], mat1[:, :, i:i + 1].expand_as(out))
        else:
            for i in range(size[2]):
                kqi_out += FB.temporary_KQI(out / size[2], mat2[:, i:i + 1, :].expand_as(out))
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
        dim = unsign_to_sign(grad_fn.__getattribute__('_saved_dim'))
        input = torch.zeros_like(input) + 1 + torch.cat(outputs, dim)
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=None)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), outputs = volume_inputs, volume_outputs
        dim = unsign_to_sign(grad_fn.__getattribute__('_saved_dim'))
        kqi_outs = tuple(FB.temporary_KQI(o, i) for i, o in zip(torch.split(input, input.shape[dim] // len(outputs), dim), outputs))
        return kqi_outs

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=None)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), outputs = inputs, outputs
        dim = unsign_to_sign(grad_fn.__getattribute__('_saved_dim'))
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input), torch.flatten(torch.cat(outputs, dim)))}
        return adj


class NativeLayerNormBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=3, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, weight, bias), (out,) = grad_fn(volume_outputs[0]), volume_outputs
        size = grad_fn.__getattribute__('_saved_normalized_shape')
        degree = cls.degree(input, weight, bias, size)
        index = len(out.shape) - len(size)
        out_slice = out.reshape(-1, *size)
        if input is not None:
            if index != 0:
                input_slice = torch.zeros_like(out_slice)
                for i in range(input_slice.shape[0]):
                    input_slice[i] += np.prod(size) + (out_slice[i] / degree).sum()
                input = input_slice.reshape_as(out)
            else:
                input = torch.ones_like(out) * (np.prod(size) + (out / degree).sum())

        if weight is not None:
            weight = torch.zeros(size)
            if index != 0:
                for i in range(out_slice.shape[0]):
                    weight += 1 + out_slice[i] / degree
            else:
                weight += 1 + out / degree

        if bias is not None:
            bias = torch.zeros(size)
            if index != 0:
                for i in range(out_slice.shape[0]):
                    bias += 1 + out_slice[i] / degree
            else:
                bias += 1 + out / degree
        return (input, weight, bias)

    @classmethod
    @FB.cell_KQI_Checking(args_in=3, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, weight, bias), (out,) = volume_inputs, volume_outputs
        size = grad_fn.__getattribute__('_saved_normalized_shape')
        degree = cls.degree(input, weight, bias, size)
        kqi_out = torch.zeros_like(out)
        index = len(out.shape) - len(size)
        if input is not None:
            if index != 0:
                input_slice, out_slice = input.reshape(-1, *size), out.reshape(-1, *size)
                kqi_slice = torch.zeros_like(kqi_out).reshape(-1, *size)
                for i in range(input_slice.shape[0]):
                    kqi_slice[i] += FB.temporary_KQI(out_slice[i] / degree, input_slice[i]).sum()
                kqi_out += kqi_slice.reshape_as(out)
            else:
                kqi_out += FB.temporary_KQI(out / degree, input.detach()).sum()

        if weight is not None:
            kqi_out += FB.temporary_KQI(out / degree, weight.detach().expand_as(out))

        if bias is not None:
            kqi_out += FB.temporary_KQI(out / degree, bias.detach().expand_as(out))

        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=3, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, weight, bias), (out,) = inputs, outputs
        adj = defaultdict(list)
        size = grad_fn.__getattribute__('_saved_normalized_shape')
        index = len(out.shape) - len(size)
        out_slice = out.reshape(-1, *size)
        if input is not None:
            if index != 0:
                input_slice = input.reshape(-1, *size)
                for c in range(out_slice.shape[0]):
                    for i, o in itertools.product(torch.flatten(input_slice[c]), torch.flatten(out_slice[c])):
                        adj[int(o)].append(int(i))
            else:
                for i, o in itertools.product(torch.flatten(input), torch.flatten(out)):
                    adj[int(o)].append(int(i))
        if weight is not None:
            if index != 0:
                for c in range(out_slice.shape[0]):
                    for i, o in zip(torch.flatten(weight), torch.flatten(out_slice[c])):
                        adj[int(o)].append(int(i))
            else:
                for i, o in zip(torch.flatten(weight), torch.flatten(out)):
                    adj[int(o)].append(int(i))
        if bias is not None:
            if index != 0:
                for c in range(out_slice.shape[0]):
                    for i, o in zip(torch.flatten(bias), torch.flatten(out_slice[c])):
                        adj[int(o)].append(int(i))
            else:
                for i, o in zip(torch.flatten(bias), torch.flatten(out)):
                    adj[int(o)].append(int(i))
        return {k: tuple(v) for k, v in adj.items()}

    @classmethod
    def degree(cls, input, weight, bias, size):
        degree = 0
        if input is not None:
            degree += np.prod(size)
        if weight is not None:
            degree += 1
        if bias is not None:
            degree += 1
        return degree


class NativeGroupNormBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=3, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, weight, bias), (out,) = grad_fn(*volume_outputs, grad_fn._saved_result1, grad_fn._saved_result2), volume_outputs
        channel, group = grad_fn.__getattribute__('_saved_C'), grad_fn.__getattribute__('_saved_group')
        stride = int(channel / group)
        saved_input = grad_fn.__getattribute__('_saved_input')
        num = np.prod(saved_input.shape[1:]) / group
        degree = cls.degree(input, weight, bias, num)
        if input is not None:
            for i in range(0, channel, stride):
                input[:, i:i + stride, :, :] = num + (out[:, i:i + stride, :, :] / degree).sum()
        if weight is not None:
            for i in range(0, channel, stride):
                weight[i] = num + (out[:, i:i + stride, :, :] / degree).sum()
        if bias is not None:
            for i in range(0, channel, stride):
                bias[i] = num + (out[:, i:i + stride, :, :] / degree).sum()
        return (input, weight, bias)

    @classmethod
    @FB.cell_KQI_Checking(args_in=3, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, weight, bias), (out,) = volume_inputs, volume_outputs
        channel, group = grad_fn.__getattribute__('_saved_C'), grad_fn.__getattribute__('_saved_group')
        stride = int(channel / group)
        saved_input = grad_fn.__getattribute__('_saved_input')
        num = np.prod(saved_input.shape[1:]) / group
        degree = cls.degree(input, weight, bias, num)
        kqi_out = torch.zeros_like(out)
        if input is not None:
            for i in range(0, channel, stride):
                kqi_out[:, i:i + stride, :, :] += FB.temporary_KQI(out[:, i:i + stride, :, :] / degree, input[:, i:i + stride, :, :]).sum()
        if weight is not None:
            for i in range(0, channel, stride):
                kqi_out[:, i:i + stride, :, :] += FB.temporary_KQI(out[:, i:i + stride, :, :] / degree, weight[i]).sum()
        if bias is not None:
            for i in range(0, channel, stride):
                kqi_out[:, i:i + stride, :, :] += FB.temporary_KQI(out[:, i:i + stride, :, :] / degree, bias[i]).sum()
        return (kqi_out,)

    @classmethod
    @FB.cell_Graph_Checking(args_in=3, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, weight, bias), (out,) = inputs, outputs
        channel, group = grad_fn.__getattribute__('_saved_C'), grad_fn.__getattribute__('_saved_group')
        stride = int(channel / group)
        adj = defaultdict(list)
        if input is not None:
            for batch_in, batch_out in zip(input, out):
                for c in range(0, channel, stride):
                    for o in torch.flatten(batch_out[c:c + stride, :, :]):
                        adj[int(o)].extend([int(i) for i in torch.flatten(batch_in[c:c + stride, :, :])])
        if weight is not None:
            for batch_in, batch_out in zip(input, weight):
                for c in range(0, channel, stride):
                    for o in torch.flatten(batch_out[c:c + stride]):
                        adj[int(o)].extend([int(i) for i in torch.flatten(batch_in[c:c + stride, :, :])])
        if bias is not None:
            for batch_in, batch_out in zip(input, bias):
                for c in range(0, channel, stride):
                    for o in torch.flatten(batch_out[c:c + stride]):
                        adj[int(o)].extend([int(i) for i in torch.flatten(batch_in[c:c + stride, :, :])])
        return {k: tuple(v) for k, v in adj.items()}

    @classmethod
    def degree(cls, input, weight, bias, num):
        degree = 0
        if input is not None:
            degree += num
        if weight is not None:
            degree += 1
        if bias is not None:
            degree += 1
        return degree


class NativeBatchNormBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=3, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, weight, bias), (out,) = grad_fn(volume_outputs[0]), volume_outputs
        num = np.prod(out.shape) / out.shape[1]
        degree = cls.degree(input, weight, bias, num)
        if input is not None:
            input = torch.zeros_like(input)
            for i in range(input.shape[1]):
                input[:, i, ...] = num + (out[:, i, ...] / degree).sum()
        if weight is not None:
            weight = torch.zeros_like(weight)
            for i in range(out.shape[1]):
                weight[i] = num + (out[:, i, ...] / degree).sum()
        if bias is not None:
            bias = torch.zeros_like(bias)
            for i in range(out.shape[1]):
                bias[i] = num + (out[:, i, ...] / degree).sum()
        return (input, weight, bias)

    @classmethod
    @FB.cell_KQI_Checking(args_in=3, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, weight, bias), (out,) = volume_inputs, volume_outputs
        num = np.prod(out.shape) / out.shape[1]
        degree = cls.degree(input, weight, bias, num)
        kqi_out = torch.zeros_like(out)
        if input is not None:
            for i in range(input.shape[1]):
                kqi_out[:, i, ...] += FB.temporary_KQI(out[:, i, ...] / degree, input.detach()[:, i, ...]).sum()
        if weight is not None:
            for i in range(out.shape[1]):
                kqi_out[:, i, ...] += FB.temporary_KQI(out[:, i, ...] / degree, weight.detach()[i].expand_as(out[:, i, ...]))
        if bias is not None:
            for i in range(out.shape[1]):
                kqi_out[:, i, ...] += FB.temporary_KQI(out[:, i, ...] / degree, bias.detach()[i].expand_as(out[:, i, ...]))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=3, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, weight, bias), (out,) = inputs, outputs
        adj = defaultdict(list)
        if input is not None:
            for c in range(out.shape[1]):
                for i, o in itertools.product(torch.flatten(input[:, c, ...]), torch.flatten(out[:, c, ...])):
                    adj[int(o)].append(int(i))
        if weight is not None:
            for c in range(out.shape[1]):
                for i, o in itertools.product(torch.flatten(weight[c]), torch.flatten(out[:, c, ...])):
                    adj[int(o)].append(int(i))
        if bias is not None:
            for c in range(out.shape[1]):
                for i, o in itertools.product(torch.flatten(bias[c]), torch.flatten(out[:, c, ...])):
                    adj[int(o)].append(int(i))
        return {k: tuple(v) for k, v in adj.items()}

    @classmethod
    def degree(cls, input, weight, bias, num):
        degree = 0
        if input is not None:
            degree += num
        if weight is not None:
            degree += 1
        if bias is not None:
            degree += 1
        return degree


class SumBackward1(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out,) = grad_fn(volume_outputs[0]), volume_outputs
        dim, keepdim = tuple(unsign_to_sign(i) for i in grad_fn.__getattribute__('_saved_dim')), grad_fn.__getattribute__('_saved_keepdim')
        if not keepdim:
            out = out[tuple(None if d in dim else slice(None) for d in range(input.dim()))]
        num = np.prod([input.shape[i] for i in dim])
        input = 1 + out.expand_as(input) / num
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        dim, keepdim = tuple(unsign_to_sign(i) for i in grad_fn.__getattribute__('_saved_dim')), grad_fn.__getattribute__('_saved_keepdim')
        if not keepdim:
            out = out[tuple(None if d in dim else slice(None) for d in range(input.dim()))]
        num = np.prod([input.shape[i] for i in dim])
        kqi_out = FB.temporary_KQI(out.expand_as(input) / num, input).sum(dim, keepdim)
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out,) = inputs, outputs
        dim = tuple(unsign_to_sign(i) for i in grad_fn.__getattribute__('_saved_dim'))
        dim = tuple(sorted(d if d >= 0 else input.dim() + d for d in dim))
        adj = {int(o.item()): tuple(int(i.item()) for i in torch.flatten(i_list)) for i_list, o in zip(torch.flatten(input.permute(tuple(d for d in range(input.dim()) if d not in dim) + dim), end_dim=input.dim() - len(dim) - 1), torch.flatten(out))}
        return adj


class NormBackward1(SumBackward1):
    pass


class MeanBackward1(SumBackward1):
    pass


class ExpandBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out,) = grad_fn(volume_outputs[0]), volume_outputs
        num = np.prod(out.shape) / np.prod(input.shape)
        input = num + out.sum_to_size(input.shape)
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        kqi_out = FB.temporary_KQI(out, input.expand_as(out))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out,) = inputs, outputs
        adj = defaultdict(list)
        for i, o in zip(torch.flatten(input), torch.flatten(out.sum_to_size(input.shape))):
            adj[int(o)].append(int(i))
        return {k: tuple(v) for k, v in adj.items()}


class MeanBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out,) = grad_fn(volume_outputs[0]), volume_outputs
        num = np.prod(input.shape)
        input = 1 + out.expand_as(input) / num
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        num = np.prod(input.shape)
        kqi_out = FB.temporary_KQI(out.expand_as(input) / num, input).sum()
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out,) = inputs, outputs
        adj = {int(o): (int(i), ) for i, o in itertools.product(torch.flatten(input), torch.flatten(out))}
        return adj


class AvgPoolBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        kernel_size = grad_fn.__getattribute__('_saved_kernel_size')
        padding = grad_fn.__getattribute__('_saved_padding')
        stride = grad_fn.__getattribute__('_saved_stride')
        degree = cls.degree(input, out, kernel_size, stride, padding)
        ndim = len(kernel_size)
        index = input.dim() - ndim

        indexing = lambda *args: [slice(None)] * index + [slice(i, H * s + i, s) for i, H, s in zip(args, out.shape[index:], stride)]

        add = [max(0, (s - 1) * stride[k] + kernel_size[k] - input.shape[k + index] - 2 * padding[k]) for s, k in zip(out.shape[index:], range(ndim))]

        end = [None if padding[i] + add[i] == 0 else -padding[i] - add[i] for i in range(ndim)]

        input_padding = torch.zeros(input.shape[:index] + tuple(input.shape[i] + 2 * padding[i - index] + add[i - index] for i in range(index, ndim + index)))

        for offset in itertools.product(*[range(0, kernel_size[i]) for i in range(ndim)]):
            input_padding[indexing(*offset)] += 1 + out / degree
        input = input_padding[[slice(None)] * index + [slice(padding[i], end[i]) for i in range(ndim)]].clone()
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        kernel_size = grad_fn.__getattribute__('_saved_kernel_size')
        padding = grad_fn.__getattribute__('_saved_padding')
        stride = grad_fn.__getattribute__('_saved_stride')
        degree = cls.degree(input, out, kernel_size, stride, padding)
        ndim = len(kernel_size)
        index = input.dim() - ndim

        indexing = lambda *args: [slice(None)] * index + [slice(i, H * s + i, s) for i, H, s in zip(args, out.shape[index:], stride)]
        add = [max(0, (s - 1) * stride[k] + kernel_size[k] - input.shape[k + index] - 2 * padding[k]) for s, k in zip(out.shape[index:], range(ndim))]

        end = [None if padding[i] + add[i] == 0 else -padding[i] - add[i] for i in range(ndim)]

        input_padding = torch.zeros(input.shape[:index] + tuple(input.shape[i] + 2 * padding[i - index] + add[i - index] for i in range(index, ndim + index)))
        input_padding[[slice(None)] * index + [slice(padding[i], end[i]) for i in range(ndim)]] = input
        kqi_out = torch.zeros_like(out)
        for offset in itertools.product(*[range(0, kernel_size[i]) for i in range(ndim)]):
            tmp = input_padding.clone()
            args = [next(m for m in range(j, input_padding.shape[k + index], stride[k]) if m >= padding[k]) for k, j in zip(range(ndim), offset)]
            tmp[indexing(*offset)] = out / degree
            tmp[(slice(None),) * index + tuple(slice(arg, e, s) for arg, e, s in zip(args, end, stride))] = input_padding[(slice(None),) * index + tuple(slice(arg, e, s) for arg, e, s in zip(args, end, stride))]
            kqi_out += FB.temporary_KQI(out / degree, tmp[indexing(*offset)])
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        kernel_size = grad_fn.__getattribute__('_saved_kernel_size')
        padding = grad_fn.__getattribute__('_saved_padding')
        stride = grad_fn.__getattribute__('_saved_stride')
        ndim = len(kernel_size)
        index = input.dim() - ndim

        indexing = lambda *args: [slice(next(k - padding[j] for k in range(i, input.shape[j + index] + 2 * padding[j], stride[j]) if k >= padding[j]), out.shape[j + index] * stride[j] + i - padding[j], stride[j]) for i, j in zip(args, range(ndim))]
        adj = defaultdict(list)
        for offset in itertools.product(*[range(0, kernel_size[d]) for d in range(ndim)]):
            left = [max(0, math.ceil((padding[d] - offset[d]) / stride[d])) for d in range(ndim)]
            right = [min(out.shape[d + index], math.ceil((input.shape[d + index] - offset[d] + padding[d]) / stride[d])) for d in range(ndim)]
            for i, o in zip(torch.flatten(input[(slice(None),) * index + tuple(indexing(*offset))]), torch.flatten(out[(slice(None),) * index + tuple(slice(left[d], right[d]) for d in range(ndim))])):
                adj[int(o)].append(int(i))
        return {k: tuple(v) for k, v in adj.items()}

    @classmethod
    def degree(cls, input, out, kernel_size, stride, padding):
        degree = torch.zeros_like(out)
        ndim = len(kernel_size)
        index = input.dim() - ndim
        for offset in itertools.product(*[range(0, kernel_size[d]) for d in range(ndim)]):
            left = [max(0, math.ceil((padding[d] - offset[d]) / stride[d])) for d in range(ndim)]
            right = [min(out.shape[d + index], math.ceil((input.shape[d + index] - offset[d] + padding[d]) / stride[d])) for d in range(ndim)]
            degree[(slice(None), ) * index + tuple(slice(left[d], right[d]) for d in range(ndim))] += 1
        return degree


class AvgPool2DBackward0(AvgPoolBackward0):
    pass


class AvgPool3DBackward0(AvgPoolBackward0):
    pass


class AdaptivePoolBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        ndim = input.dim() - 1
        padding = [0] * ndim
        stride = [None] * ndim
        kernel_size = [None] * ndim
        for i in range(ndim):
            stride[i] = math.floor(input.shape[i + 1] / out.shape[i + 1])
            kernel_size[i] = input.shape[i + 1] - (out.shape[i + 1] - 1) * stride[i]
        degree = cls.degree(input, out, kernel_size, stride, padding)

        indexing = lambda *args: [slice(None)] + [slice(i, H * s + i, s) for i, H, s in zip(args, out.shape[1:], stride)]

        add = [max(0, (s - 1) * stride[k] + kernel_size[k] - input.shape[k + 1] - 2 * padding[k]) for s, k in zip(out.shape[1:], range(ndim))]

        end = [None if padding[i] + add[i] == 0 else -padding[i] - add[i] for i in range(ndim)]

        input_padding = torch.zeros((input.shape[0], *[input.shape[i] + 2 * padding[i - 1] + add[i - 1] for i in range(1, ndim + 1)]))

        for offset in itertools.product(*[range(0, kernel_size[i]) for i in range(ndim)]):
            input_padding[indexing(*offset)] += 1 + out / degree
        input = input_padding[[slice(None)] + [slice(padding[i], end[i]) for i in range(ndim)]].clone()
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        ndim = input.dim() - 1
        padding = [0] * ndim
        stride = [None] * ndim
        kernel_size = [None] * ndim
        for i in range(ndim):
            stride[i] = math.floor(input.shape[i + 1] / out.shape[i + 1])
            kernel_size[i] = input.shape[i + 1] - (out.shape[i + 1] - 1) * stride[i]
        degree = cls.degree(input, out, kernel_size, stride, padding)

        indexing = lambda *args: [slice(None)] + [slice(i, H * s + i, s) for i, H, s in zip(args, out.shape[1:], stride)]
        add = [max(0, (s - 1) * stride[k] + kernel_size[k] - input.shape[k + 1] - 2 * padding[k]) for s, k in zip(out.shape[1:], range(ndim))]

        end = [None if padding[i] + add[i] == 0 else -padding[i] - add[i] for i in range(ndim)]

        input_padding = torch.zeros((input.shape[0], *[input.shape[i] + 2 * padding[i - 1] + add[i - 1] for i in range(1, ndim + 1)]))
        input_padding[[slice(None)] + [slice(padding[i], end[i]) for i in range(ndim)]] = input
        kqi_out = torch.zeros_like(out)
        for offset in itertools.product(*[range(0, kernel_size[i]) for i in range(ndim)]):
            tmp = input_padding.clone()
            args = [next(m for m in range(j, input_padding.shape[k + 1], stride[k]) if m >= padding[k]) for k, j in zip(range(ndim), offset)]
            tmp[indexing(*offset)] = out / degree
            tmp[(slice(None),) + tuple(slice(arg, e, s) for arg, e, s in zip(args, end, stride))] = input_padding[(slice(None),) + tuple(slice(arg, e, s) for arg, e, s in zip(args, end, stride))]
            kqi_out += FB.temporary_KQI(out / degree, tmp[indexing(*offset)])
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        ndim = input.dim() - 1
        padding = [0] * ndim
        stride = [None] * ndim
        kernel_size = [None] * ndim
        for i in range(ndim):
            stride[i] = math.floor(input.shape[i + 1] / out.shape[i + 1])
            kernel_size[i] = input.shape[i + 1] - (out.shape[i + 1] - 1) * stride[i]
        indexing = lambda *args: [slice(next(k - padding[j] for k in range(i, input.shape[j + 1] + 2 * padding[j], stride[j]) if k >= padding[j]), out.shape[j + 1] * stride[j] + i - padding[j], stride[j]) for i, j in zip(args, range(ndim))]
        adj = defaultdict(list)
        for offset in itertools.product(*[range(0, kernel_size[d]) for d in range(ndim)]):
            left = [max(0, math.ceil((padding[d] - offset[d]) / stride[d])) for d in range(ndim)]
            right = [min(out.shape[d + 1], math.ceil((input.shape[d + 1] - offset[d] + padding[d]) / stride[d])) for d in range(ndim)]
            for i, o in zip(torch.flatten(input[(slice(None),) + tuple(indexing(*offset))]), torch.flatten(out[(slice(None),) + tuple(slice(left[d], right[d]) for d in range(ndim))])):
                adj[int(o)].append(int(i))
        return {k: tuple(v) for k, v in adj.items()}

    @classmethod
    def degree(cls, input, out, kernel_size, stride, padding):
        degree = torch.zeros_like(out)
        ndim = input.dim() - 1
        for offset in itertools.product(*[range(0, kernel_size[d]) for d in range(ndim)]):
            left = [max(0, math.ceil((padding[d] - offset[d]) / stride[d])) for d in range(ndim)]
            right = [min(out.shape[d + 1], math.ceil((input.shape[d + 1] - offset[d] + padding[d]) / stride[d])) for d in range(ndim)]
            degree[(slice(None), ) + tuple(slice(left[d], right[d]) for d in range(ndim))] += 1
        return degree


class AdaptiveAvgPool2DBackward0(AdaptivePoolBackward0):
    pass


class AdaptiveAvgPool3DBackward0(AdaptivePoolBackward0):
    pass


class MaxPoolBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        kernel_size = grad_fn.__getattribute__('_saved_kernel_size')
        padding = grad_fn.__getattribute__('_saved_padding')
        stride = grad_fn.__getattribute__('_saved_stride')
        dilation = grad_fn.__getattribute__('_saved_dilation')
        degree = cls.degree(input, out, kernel_size, stride, padding, dilation)
        ndim = len(kernel_size)
        index = input.dim() - ndim

        indexing = lambda *args: [slice(None)] * index + [slice(i, H * s + i, s) for i, H, s in zip(args, out.shape[index:], stride)]

        add = [max(0, (s - 1) * stride[k] + kernel_size[k] * dilation[k] - input.shape[k + index] - 2 * padding[k]) for s, k in zip(out.shape[index:], range(ndim))]

        end = [None if padding[i] + add[i] == 0 else -padding[i] - add[i] for i in range(ndim)]

        input_padding = torch.zeros(input.shape[:index] + tuple(input.shape[i] + 2 * padding[i - index] + add[i - index] for i in range(index, ndim + index)))

        for offset in itertools.product(*[range(0, kernel_size[i] * dilation[i], dilation[i]) for i in range(ndim)]):
            input_padding[indexing(*offset)] += 1 + out / degree
        input = input_padding[[slice(None)] * index + [slice(padding[i], end[i]) for i in range(ndim)]].clone()
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        kernel_size = grad_fn.__getattribute__('_saved_kernel_size')
        padding = grad_fn.__getattribute__('_saved_padding')
        stride = grad_fn.__getattribute__('_saved_stride')
        dilation = grad_fn.__getattribute__('_saved_dilation')
        degree = cls.degree(input, out, kernel_size, stride, padding, dilation)
        ndim = len(kernel_size)
        index = input.dim() - ndim

        indexing = lambda *args: [slice(None)] * index + [slice(i, H * s + i, s) for i, H, s in zip(args, out.shape[index:], stride)]
        add = [max(0, (s - 1) * stride[k] + kernel_size[k] * dilation[k] - input.shape[k + index] - 2 * padding[k]) for s, k in zip(out.shape[index:], range(ndim))]

        end = [None if padding[i] + add[i] == 0 else -padding[i] - add[i] for i in range(ndim)]

        input_padding = torch.zeros(input.shape[:index] + tuple(input.shape[i] + 2 * padding[i - index] + add[i - index] for i in range(index, ndim + index)))
        input_padding[[slice(None)] * index + [slice(padding[i], end[i]) for i in range(ndim)]] = input
        kqi_out = torch.zeros_like(out)
        for offset in itertools.product(*[range(0, kernel_size[i] * dilation[i], dilation[i]) for i in range(ndim)]):
            tmp = input_padding.clone()
            args = [next(m for m in range(j, input_padding.shape[k + index], stride[k]) if m >= padding[k]) for k, j in zip(range(ndim), offset)]
            tmp[indexing(*offset)] = out / degree
            tmp[(slice(None),) * index + tuple(slice(arg, e, s) for arg, e, s in zip(args, end, stride))] = input_padding[(slice(None),) * index + tuple(slice(arg, e, s) for arg, e, s in zip(args, end, stride))]
            kqi_out += FB.temporary_KQI(out / degree, tmp[indexing(*offset)])
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        kernel_size = grad_fn.__getattribute__('_saved_kernel_size')
        padding = grad_fn.__getattribute__('_saved_padding')
        stride = grad_fn.__getattribute__('_saved_stride')
        dilation = grad_fn.__getattribute__('_saved_dilation')
        ndim = len(kernel_size)
        index = input.dim() - ndim
        indexing = lambda *args: [slice(next(k - padding[j] for k in range(i, input.shape[j + index] + 2 * padding[j], stride[j]) if k >= padding[j]), out.shape[j + index] * stride[j] + i - padding[j], stride[j]) for i, j in zip(args, range(ndim))]
        adj = defaultdict(list)
        for offset in itertools.product(*[range(0, kernel_size[d] * dilation[d], dilation[d]) for d in range(ndim)]):
            left = [max(0, math.ceil((padding[d] - offset[d]) / stride[d])) for d in range(ndim)]
            right = [min(out.shape[d + index], math.ceil((input.shape[d + index] - offset[d] + padding[d]) / stride[d])) for d in range(ndim)]
            for i, o in zip(torch.flatten(input[(slice(None),) * index + tuple(indexing(*offset))]), torch.flatten(out[(slice(None),) * index + tuple(slice(left[d], right[d]) for d in range(ndim))])):
                adj[int(o)].append(int(i))
        return {k: tuple(v) for k, v in adj.items()}

    @classmethod
    def degree(cls, input, out, kernel_size, stride, padding, dilation):
        degree = torch.zeros_like(out)
        ndim = len(kernel_size)
        index = input.dim() - ndim
        for offset in itertools.product(*[range(0, kernel_size[d] * dilation[d], dilation[d]) for d in range(ndim)]):
            left = [max(0, math.ceil((padding[d] - offset[d]) / stride[d])) for d in range(ndim)]
            right = [min(out.shape[d + index], math.ceil((input.shape[d + index] - offset[d] + padding[d]) / stride[d])) for d in range(ndim)]
            degree[(slice(None), ) * index + tuple(slice(left[d], right[d]) for d in range(ndim))] += 1
        return degree


class MaxPool2DWithIndicesBackward0(MaxPoolBackward0):
    pass


class MaxPool3DWithIndicesBackward0(MaxPoolBackward0):
    pass


class AdaptiveMaxPool2DBackward0(AdaptivePoolBackward0):
    pass


class AdaptiveMaxPool3DBackward0(AdaptivePoolBackward0):
    pass


class EmbeddingBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        saved_input = grad_fn.__getattribute__('_saved_indices')
        dim = out.shape[2]
        C, H = saved_input.shape[:2]
        input = torch.zeros(saved_input.shape)
        for i, j in itertools.product(range(C), range(H)):
            input[i, j] += dim + out[i, j, :].sum()
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        kqi_out = FB.temporary_KQI(out, input.unsqueeze(-1).expand_as(out))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input.unsqueeze(-1).expand_as(out)), torch.flatten(out))}
        return adj


class ConstantPadNdBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        pad = grad_fn.__getattribute__('_saved_pad')
        input = torch.masked_select(out, torch.nn.functional.pad(torch.zeros_like(input), pad, value=-1) == 0).reshape_as(input) + 1
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        pad = grad_fn.__getattribute__('_saved_pad')
        kqi_out = FB.temporary_KQI(out, torch.nn.functional.pad(input, pad, value=float('nan')))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        pad = grad_fn.__getattribute__('_saved_pad')
        adj = {int(o): (int(i), ) if i == i else tuple() for i, o in zip(torch.flatten(torch.nn.functional.pad(input, pad, value=float('nan'))), torch.flatten(out))}
        return adj


class PixelShuffleBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out,) = grad_fn(volume_outputs[0]), volume_outputs
        factor = grad_fn.__getattribute__('_saved_upscale_factor')
        input = 1 + cls.pixel_shuffle_inv(out, factor)
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        factor = grad_fn.__getattribute__('_saved_upscale_factor')
        kqi_out = FB.temporary_KQI(out, cls.pixel_shuffle(input, factor))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        factor = grad_fn.__getattribute__('_saved_upscale_factor')
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input), torch.flatten(cls.pixel_shuffle_inv(out, factor)))}
        return adj

    @classmethod
    def pixel_shuffle(cls, tensor, scale_factor):
        num, ch, height, width = tensor.shape

        new_ch = ch // (scale_factor * scale_factor)
        new_height = height * scale_factor
        new_width = width * scale_factor

        tensor = tensor.view(num, new_ch, scale_factor, scale_factor, height, width)
        tensor = tensor.permute(0, 1, 4, 2, 5, 3).contiguous()
        tensor = tensor.view(num, new_ch, new_height, new_width)
        return tensor

    @classmethod
    def pixel_shuffle_inv(cls, tensor, scale_factor):

        num, ch, height, width = tensor.shape
        new_ch = ch * (scale_factor * scale_factor)
        new_height = height // scale_factor
        new_width = width // scale_factor

        tensor = tensor.view(num, ch, new_height, scale_factor, new_width, scale_factor)
        tensor = tensor.permute(0, 1, 3, 5, 2, 4).contiguous()
        tensor = tensor.view(num, new_ch, new_height, new_width)

        return tensor


class PixelUnshuffleBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out,) = grad_fn(volume_outputs[0]), volume_outputs
        factor = grad_fn.__getattribute__('_saved_downscale_factor')
        input = 1 + cls.pixel_shuffle(out, factor)
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        factor = grad_fn.__getattribute__('_saved_downscale_factor')
        kqi_out = FB.temporary_KQI(out, cls.pixel_shuffle_inv(input, factor))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        factor = grad_fn.__getattribute__('_saved_downscale_factor')
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input), torch.flatten(cls.pixel_shuffle(out, factor)))}
        return adj

    @classmethod
    def pixel_shuffle(cls, tensor, scale_factor):
        num, ch, height, width = tensor.shape

        new_ch = ch // (scale_factor * scale_factor)
        new_height = height * scale_factor
        new_width = width * scale_factor

        tensor = tensor.view(num, new_ch, scale_factor, scale_factor, height, width)
        tensor = tensor.permute(0, 1, 4, 2, 5, 3).contiguous()
        tensor = tensor.view(num, new_ch, new_height, new_width)
        return tensor

    @classmethod
    def pixel_shuffle_inv(cls, tensor, scale_factor):

        num, ch, height, width = tensor.shape
        new_ch = ch * (scale_factor * scale_factor)
        new_height = height // scale_factor
        new_width = width // scale_factor

        tensor = tensor.view(num, ch, new_height, scale_factor, new_width, scale_factor)
        tensor = tensor.permute(0, 1, 3, 5, 2, 4).contiguous()
        tensor = tensor.view(num, new_ch, new_height, new_width)

        return tensor


class Im2ColBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        kernel_size = grad_fn.__getattribute__('_saved_kernel_size')
        padding = grad_fn.__getattribute__('_saved_padding')
        stride = grad_fn.__getattribute__('_saved_stride')
        dilation = grad_fn.__getattribute__('_saved_dilation')

        input_padding = torch.zeros(input.shape[:2] + tuple(input.shape[i] + 2 * padding[i - 2] for i in range(2, 4)))
        end = [None if padding[i] == 0 else -padding[i] for i in range(2)]

        _, channels, Hin, Win = input.shape
        dilated_h, dilated_w = [(k - 1) * d + 1 for k, d in zip(kernel_size, dilation)]
        Hout, Wout = [(i + 2 * pad - d) // s + 1 for i, pad, d, s in zip((Hin, Win), padding, (dilated_h, dilated_w), stride)]

        indexing = lambda c, i, j: [slice(None), c, slice(i, Hout * stride[0] + i, stride[0]), slice(j, Wout * stride[1] + j, stride[1])]
        index = 0
        for c, i, j in itertools.product(range(channels), range(0, kernel_size[0] * dilation[0], dilation[0]), range(0, kernel_size[1] * dilation[1], dilation[1])):
            input_padding[indexing(c, i, j)] += 1 + out[:, index, :].reshape(-1, Hout, Wout)
            index += 1

        input = input_padding[[slice(None)] * 2 + [slice(padding[i], end[i]) for i in range(2)]].clone()
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        kernel_size = grad_fn.__getattribute__('_saved_kernel_size')
        padding = grad_fn.__getattribute__('_saved_padding')
        stride = grad_fn.__getattribute__('_saved_stride')
        dilation = grad_fn.__getattribute__('_saved_dilation')

        _, channels, Hin, Win = input.shape
        dilated_h, dilated_w = [(k - 1) * d + 1 for k, d in zip(kernel_size, dilation)]
        Hout, Wout = [(i + 2 * pad - d) // s + 1 for i, pad, d, s in zip((Hin, Win), padding, (dilated_h, dilated_w), stride)]

        kqi_out = torch.zeros_like(out)
        input_padding = torch.nn.functional.pad(input, (padding[0], padding[0], padding[1], padding[1]), value=float('nan'))

        indexing = lambda c, i, j: [slice(None), c, slice(i, Hout * stride[0] + i, stride[0]), slice(j, Wout * stride[1] + j, stride[1])]
        index = 0
        for c, i, j in itertools.product(range(channels), range(0, kernel_size[0] * dilation[0], dilation[0]), range(0, kernel_size[1] * dilation[1], dilation[1])):
            kqi_out[:, index, :] += FB.temporary_KQI(out[:, index, :], input_padding[indexing(c, i, j)].reshape(-1, Hout * Wout))
            index += 1

        return (kqi_out,)

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        kernel_size = grad_fn.__getattribute__('_saved_kernel_size')
        padding = grad_fn.__getattribute__('_saved_padding')
        stride = grad_fn.__getattribute__('_saved_stride')
        dilation = grad_fn.__getattribute__('_saved_dilation')

        _, channels, Hin, Win = input.shape
        dilated_h, dilated_w = [(k - 1) * d + 1 for k, d in zip(kernel_size, dilation)]
        Hout, Wout = [(i + 2 * pad - d) // s + 1 for i, pad, d, s in zip((Hin, Win), padding, (dilated_h, dilated_w), stride)]

        input_padding = torch.nn.functional.pad(input, (padding[0], padding[0], padding[1], padding[1]), value=float('nan'))
        indexing = lambda c, i, j: [slice(None), c, slice(i, Hout * stride[0] + i, stride[0]), slice(j, Wout * stride[1] + j, stride[1])]
        index = 0
        adj = defaultdict(list)
        for c, i, j in itertools.product(range(channels), range(0, kernel_size[0] * dilation[0], dilation[0]), range(0, kernel_size[1] * dilation[1], dilation[1])):
            for i, o in zip(torch.flatten(input_padding[indexing(c, i, j)]), torch.flatten(out[:, index, :])):
                adj[int(o)].append(int(i)) if i == i else tuple()
            index += 1
        return {k: tuple(v) for k, v in adj.items()}


class Col2ImBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        kernel_size = grad_fn.__getattribute__('_saved_kernel_size')
        padding = grad_fn.__getattribute__('_saved_padding')
        stride = grad_fn.__getattribute__('_saved_stride')
        dilation = grad_fn.__getattribute__('_saved_dilation')

        _, channels, Hout, Wout = out.shape
        dilated_h, dilated_w = [(k - 1) * d + 1 for k, d in zip(kernel_size, dilation)]
        Hin, Win = [(i + 2 * pad - d) // s + 1 for i, pad, d, s in zip((Hout, Wout), padding, (dilated_h, dilated_w), stride)]
        out_padding = torch.nn.functional.pad(out, (padding[0], padding[0], padding[1], padding[1]), value=0)
        degree = cls.degree(Hin, Win, out, kernel_size, stride, padding, dilation)

        indexing = lambda c, i, j: [slice(None), c, slice(i, Hin * stride[0] + i, stride[0]), slice(j, Win * stride[1] + j, stride[1])]
        index = 0
        for c, i, j in itertools.product(range(channels), range(0, kernel_size[0] * dilation[0], dilation[0]), range(0, kernel_size[1] * dilation[1], dilation[1])):
            input[:, index, :] += 1 + (out_padding / degree)[indexing(c, i, j)].reshape(-1, Hin * Win)
            index += 1
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        kernel_size = grad_fn.__getattribute__('_saved_kernel_size')
        padding = grad_fn.__getattribute__('_saved_padding')
        stride = grad_fn.__getattribute__('_saved_stride')
        dilation = grad_fn.__getattribute__('_saved_dilation')

        _, channels, Hout, Wout = out.shape
        dilated_h, dilated_w = [(k - 1) * d + 1 for k, d in zip(kernel_size, dilation)]
        Hin, Win = [(i + 2 * pad - d) // s + 1 for i, pad, d, s in zip((Hout, Wout), padding, (dilated_h, dilated_w), stride)]
        out_padding = torch.nn.functional.pad(out, (padding[0], padding[0], padding[1], padding[1]), value=0)
        degree = cls.degree(Hin, Win, out, kernel_size, stride, padding, dilation)

        indexing = lambda c, i, j: [slice(None), c, slice(i, Hin * stride[0] + i, stride[0]), slice(j, Win * stride[1] + j, stride[1])]
        kqi_pad = torch.zeros_like(out_padding)
        index = 0
        for c, i, j in itertools.product(range(channels), range(0, kernel_size[0] * dilation[0], dilation[0]), range(0, kernel_size[1] * dilation[1], dilation[1])):
            kqi_pad[indexing(c, i, j)] += FB.temporary_KQI((out_padding / degree)[indexing(c, i, j)], input[:, index, :].reshape(-1, Hin, Win))
        end = [None if padding[i] == 0 else -padding[i] for i in range(2)]
        kqi_out = kqi_pad[[slice(None)] * 2 + [slice(padding[i], end[i]) for i in range(2)]].clone()

        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        kernel_size = grad_fn.__getattribute__('_saved_kernel_size')
        padding = grad_fn.__getattribute__('_saved_padding')
        stride = grad_fn.__getattribute__('_saved_stride')
        dilation = grad_fn.__getattribute__('_saved_dilation')

        _, channels, Hout, Wout = out.shape
        dilated_h, dilated_w = [(k - 1) * d + 1 for k, d in zip(kernel_size, dilation)]
        Hin, Win = [(i + 2 * pad - d) // s + 1 for i, pad, d, s in zip((Hout, Wout), padding, (dilated_h, dilated_w), stride)]

        out_padding = torch.nn.functional.pad(out, (padding[0], padding[0], padding[1], padding[1]), value=float('nan'))
        indexing = lambda c, i, j: [slice(None), c, slice(i, Hin * stride[0] + i, stride[0]), slice(j, Win * stride[1] + j, stride[1])]

        index = 0
        adj = defaultdict(list)
        for c, i, j in itertools.product(range(channels), range(0, kernel_size[0] * dilation[0], dilation[0]), range(0, kernel_size[1] * dilation[1], dilation[1])):
            for i, o in zip(torch.flatten(input[:, index, :]), torch.flatten(out_padding[indexing(c, i, j)])):
                if o == o:
                    adj[int(o)].append(int(i))
            index += 1
        return {k: tuple(v) for k, v in adj.items()}

    @classmethod
    def degree(cls, Hin, Win, out, kernel_size, stride, padding, dilation):
        indexing = lambda c, i, j: [slice(None), c, slice(i, Hin * stride[0] + i, stride[0]), slice(j, Win * stride[1] + j, stride[1])]
        out_padding = torch.nn.functional.pad(out, (padding[0], padding[0], padding[1], padding[1]), value=0)
        degree_pad = torch.zeros_like(out_padding)
        for c, i, j in itertools.product(range(out.shape[1]), range(0, kernel_size[0] * dilation[0], dilation[0]), range(0, kernel_size[1] * dilation[1], dilation[1])):
            degree_pad[indexing(c, i, j)] += 1
        end = [None if padding[i] == 0 else -padding[i] for i in range(2)]
        degree = degree_pad[[slice(None)] * 2 + [slice(padding[i], end[i]) for i in range(2)]].clone()
        return degree


class L1LossBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=2, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, target), (out,) = grad_fn(volume_outputs[0]), volume_outputs
        if input is not None:
            input = torch.zeros_like(input)
            input += 1 + out / np.prod(input.shape)
        if target is not None:
            target = torch.zeros_like(target)
            target += 1 + out / np.prod(target.shape)
        return (input, target)

    @classmethod
    @FB.cell_KQI_Checking(args_in=2, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, target), (out, ) = volume_inputs, volume_outputs
        kqi_out = torch.zeros_like(out)
        if input is not None:
            kqi_out += FB.temporary_KQI(out.expand_as(input) / np.prod(input.shape), input).sum()
        if target is not None:
            kqi_out += FB.temporary_KQI(out.expand_as(target) / np.prod(target.shape), target).sum()
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=2, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, target), (out, ) = inputs, outputs
        adj = defaultdict(list)
        if input is not None:
            for i, o in itertools.product(torch.flatten(input), torch.flatten(out)):
                adj[int(o)].append(int(i))
        if target is not None:
            for i, o in itertools.product(torch.flatten(target), torch.flatten(out)):
                adj[int(o)].append(int(i))
        return {k: tuple(v) for k, v in adj.items()}


class IndexCopyBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=2, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, source), (out,) = grad_fn(volume_outputs[0]), volume_outputs
        index = grad_fn.__getattribute__('_saved_index')
        index = index.unsqueeze(0) if index.dim() == 0 else index
        dim = grad_fn.__getattribute__('_saved_dim')
        if input is not None:
            input = torch.zeros_like(input)
            for i in range(out.size(dim)):
                if i not in index:
                    selected_input = input.select(dim, i)
                    selected_input += 1 + out.select(dim, i)
                    input.select(dim, i).copy_(selected_input)
        if source is not None:
            source = torch.zeros_like(source)
            for num, i in enumerate(index):
                selected_source = source.select(dim, num)
                selected_source += 1 + out.select(dim, i)
                source.select(dim, num).copy_(selected_source)
        return (input, source)

    @classmethod
    @FB.cell_KQI_Checking(args_in=2, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, source), (out, ) = volume_inputs, volume_outputs
        kqi_out = torch.zeros_like(out)
        index = grad_fn.__getattribute__('_saved_index')
        index = index.unsqueeze(0) if index.dim() == 0 else index
        dim = grad_fn.__getattribute__('_saved_dim')
        if input is not None:
            for i in range(out.size(dim)):
                if i not in index:
                    selected_kqi = FB.temporary_KQI(out.select(dim, i), input.select(dim, i))
                    kqi_out.select(dim, i).copy_(selected_kqi)
        if source is not None:
            for num, i in enumerate(index):
                selected_kqi = FB.temporary_KQI(out.select(dim, i), source.select(dim, num))
                kqi_out.select(dim, i).copy_(selected_kqi)
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=2, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, source), (out, ) = inputs, outputs
        index = grad_fn.__getattribute__('_saved_index')
        index = index.unsqueeze(0) if index.dim() == 0 else index
        dim = grad_fn.__getattribute__('_saved_dim')
        adj = defaultdict(list)
        for ind in range(out.size(dim)):
            if ind not in index:
                if input is not None:
                    for i, o in zip(torch.flatten(input.select(dim, ind)), torch.flatten(out.select(dim, ind))):
                        adj[int(o)].append(int(i))
                else:
                    for o in torch.flatten(out.select(dim, ind)):
                        adj[int(o)].append(())
            else:
                num = list(index).index(ind)
                if source is not None:
                    for i, o in zip(torch.flatten(source.select(dim, num)), torch.flatten(out.select(dim, ind))):
                        adj[int(o)].append(int(i))
                else:
                    for o in torch.flatten(out.select(dim, ind)):
                        adj[int(o)].append(())
        return {k: tuple(v) for k, v in adj.items()}


class PermuteBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        dims = tuple(unsign_to_sign(i) for i in grad_fn.__getattribute__('_saved_dims'))
        dims_r = tuple(i for _, i in sorted(tuple((d, i) for i, d in enumerate(dims))))
        input = 1 + out.permute(dims_r)
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        dims = tuple(unsign_to_sign(i) for i in grad_fn.__getattribute__('_saved_dims'))
        kqi_out = FB.temporary_KQI(out, input.permute(dims))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        dims = tuple(unsign_to_sign(i) for i in grad_fn.__getattribute__('_saved_dims'))
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input.permute(dims)), torch.flatten(out))}
        return adj


class GatherBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        index = grad_fn.__getattribute__('_saved_index')
        dim = grad_fn.__getattribute__('_saved_dim')
        input = torch.zeros_like(input).scatter_(dim, index, out + 1)
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        index = grad_fn.__getattribute__('_saved_index')
        dim = grad_fn.__getattribute__('_saved_dim')
        kqi_out = FB.temporary_KQI(out, torch.gather(input, dim, index))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        index = grad_fn.__getattribute__('_saved_index')
        dim = grad_fn.__getattribute__('_saved_dim')
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(torch.gather(input, dim, index)), torch.flatten(out))}
        return adj


class SqueezeBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        for dim in range(input.dim()):
            if input.size(dim) == 1:
                out = torch.unsqueeze(out, dim)
        input = 1 + out
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        kqi_out = FB.temporary_KQI(out, input.squeeze())
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input.squeeze()), torch.flatten(out))}
        return adj


class IndexSelectBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        index = grad_fn.__getattribute__('_saved_index')
        index = index.unsqueeze(0) if index.dim() == 0 else index
        dim = grad_fn.__getattribute__('_saved_dim')
        select_index = 1 + out
        for num, i in enumerate(index):
            input.select(dim, i).copy_(select_index[num])
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        index = grad_fn.__getattribute__('_saved_index')
        index = index.unsqueeze(0) if index.dim() == 0 else index
        dim = grad_fn.__getattribute__('_saved_dim')
        kqi_out = torch.zeros_like(out)
        for num, i in enumerate(index):
            kqi_out[num] += FB.temporary_KQI(out[num], input.select(dim, i))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        index = grad_fn.__getattribute__('_saved_index')
        index = index.unsqueeze(0) if index.dim() == 0 else index
        dim = grad_fn.__getattribute__('_saved_dim')
        adj = defaultdict(list)
        for num, ind in enumerate(index):
            for i, o in zip(torch.flatten(input.select(dim, ind)), torch.flatten(out[num])):
                adj[int(o)].append(int(i))
        return {k: tuple(v) for k, v in adj.items()}


class RollBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        dims = grad_fn.__getattribute__('_saved_dims')
        shifts = tuple(unsign_to_sign(s) for s in grad_fn.__getattribute__('_saved_shifts'))
        shifts_inv = tuple(-s for s in shifts)
        input = 1 + out.roll(dims=dims, shifts=shifts_inv)
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        dims = grad_fn.__getattribute__('_saved_dims')
        shifts = tuple(unsign_to_sign(s) for s in grad_fn.__getattribute__('_saved_shifts'))
        kqi_out = FB.temporary_KQI(out, input.roll(dims=dims, shifts=shifts))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        dims = grad_fn.__getattribute__('_saved_dims')
        shifts = tuple(unsign_to_sign(s) for s in grad_fn.__getattribute__('_saved_shifts'))
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input.roll(dims=dims, shifts=shifts)), torch.flatten(out))}
        return adj


class IndexBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        indices = grad_fn.__getattribute__('_saved_indices')[0]
        if indices.dtype == torch.bool:
            indices = torch.nonzero(indices, as_tuple=True)[0]
        input = torch.zeros_like(input)
        for i in range(len(indices)):
            input[indices[i], :] = 1 + out[i, :]
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        indices = grad_fn.__getattribute__('_saved_indices')[0]
        if indices.dtype == torch.bool:
            indices = torch.nonzero(indices, as_tuple=True)[0]
        kqi_out = torch.zeros_like(out)
        for i in range(len(indices)):
            kqi_out[i, :] = FB.temporary_KQI(out[i, :], input[indices[i], :])
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        indices = grad_fn.__getattribute__('_saved_indices')[0]
        if indices.dtype == torch.bool:
            indices = torch.nonzero(indices, as_tuple=True)[0]
        adj = defaultdict(list)
        for k in range(len(indices)):
            for i, o in zip(torch.flatten(input[indices[k], :]), torch.flatten(out[k, :])):
                adj[int(o)].append(int(i))
        return {k: tuple(v) for k, v in adj.items()}


class Upsample2DBackward1(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=1, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        input, (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        offset = np.prod(out.shape[2:]) + (out / np.prod(input.shape[2:])).sum(axis=tuple(range(2, out.ndim)))
        input = torch.ones_like(input) * offset.view(*input.shape[:2], *([1] * (input.dim() - 2))).expand_as(input)
        return (input, )

    @classmethod
    @FB.cell_KQI_Checking(args_in=1, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, ), (out, ) = volume_inputs, volume_outputs
        input_sum = input.sum_to_size(*input.shape[:2], *([1] * (input.dim() - 2)))
        kqi_out = FB.temporary_KQI(out, input_sum.expand_as(out))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), (out, ) = inputs, outputs
        batch, channel = input.shape[:2]
        adj = defaultdict(list)
        for b, c in itertools.product(range(batch), range(channel)):
            for i, o in itertools.product(torch.flatten(input[b, c]), torch.flatten(out[b, c])):
                adj[int(o)].append(int(i))
        return {k: tuple(v) for k, v in adj.items()}


class UpsampleBilinear2DBackward1(Upsample2DBackward1):
    pass


class UpsampleNearest2DBackward1(Upsample2DBackward1):
    pass


class SplitWithSizesBackward0(FB):
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
        sizes = grad_fn.__getattribute__('_saved_split_sizes')
        kqi_outs = tuple(FB.temporary_KQI(o, i) for i, o in zip(input.split_with_sizes(sizes, dim), outputs))
        return kqi_outs

    @classmethod
    @FB.cell_Graph_Checking(args_in=1, args_out=None)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, ), outputs = inputs, outputs
        dim = grad_fn.__getattribute__('_saved_dim')
        adj = {int(o): (int(i), ) for i, o in zip(torch.flatten(input), torch.flatten(torch.cat(outputs, dim)))}
        return adj


class IndexPutBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=2, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, value), (out, ) = grad_fn(volume_outputs[0]), volume_outputs
        indices = grad_fn.__getattribute__('_saved_indices')[0]
        if len(indices) == 0:
            input = 1 + out
        else:
            if input is not None:
                all_indices = torch.arange(input.shape[0])
                remain_indices = [i for i in all_indices if i not in indices]
                for i in remain_indices:
                    input[i] = 1 + out[i]
            if value is not None:
                for num, i in enumerate(indices):
                    value[num] = 1 + out[i]
        return (input, value)

    @classmethod
    @FB.cell_KQI_Checking(args_in=2, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, value), (out, ) = volume_inputs, volume_outputs
        indices = indices = grad_fn.__getattribute__('_saved_indices')[0]
        if len(indices) == 0:
            kqi_out = FB.temporary_KQI(out, input)
        else:
            kqi_out = torch.zeros_like(out)
            if input is not None:
                all_indices = torch.arange(input.shape[0])
                remain_indices = [i for i in all_indices if i not in indices]
                for i in remain_indices:
                    kqi_out[i] += FB.temporary_KQI(out[i], input[i])
            if value is not None:
                for num, i in enumerate(indices):
                    kqi_out[i] += FB.temporary_KQI(out[i], value[num])
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=2, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, value), (out, ) = inputs, outputs
        indices = indices = grad_fn.__getattribute__('_saved_indices')[0]
        adj = defaultdict(list)
        if len(indices) == 0:
            for i, o in zip(torch.flatten(input), torch.flatten(out)):
                adj[int(o)].append(int(i))
        else:
            if input is not None:
                all_indices = torch.arange(input.shape[0])
                remain_indices = [i for i in all_indices if i not in indices]
                for k in remain_indices:
                    for i, o in zip(torch.flatten(input[k]), torch.flatten(out[k])):
                        adj[int(o)].append(int(i))
            if value is not None:
                for num, k in enumerate(indices):
                    for i, o in zip(torch.flatten(value[num]), torch.flatten(out[k])):
                        adj[int(o)].append(int(i))
        return {k: tuple(v) for k, v in adj.items()}


class WhereBackward0(FB):
    @classmethod
    @FB.cell_Volume_Checking(args_in=2, args_out=1)
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, other), (out,) = grad_fn(volume_outputs[0]), volume_outputs
        if input is not None:
            input = 1 + out.sum_to_size(input.shape)
        if other is not None:
            other = 1 + out.sum_to_size(other.shape)
        return (input, other)

    @classmethod
    @FB.cell_KQI_Checking(args_in=2, args_out=1)
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        (input, other), (out, ) = volume_inputs, volume_outputs
        kqi_out = torch.zeros_like(out)
        if input is not None:
            kqi_out += FB.temporary_KQI(out, input.broadcast_to(out.shape))
        if other is not None:
            kqi_out += FB.temporary_KQI(out, other.broadcast_to(out.shape))
        return (kqi_out, )

    @classmethod
    @FB.cell_Graph_Checking(args_in=2, args_out=1)
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        (input, other), (out, ) = inputs, outputs
        adj = defaultdict(list)
        if input is not None:
            for i, o in zip(torch.flatten(input.broadcast_to(out.shape)), torch.flatten(out)):
                adj[int(o)].append(int(i))
        if other is not None:
            for i, o in zip(torch.flatten(other.broadcast_to(out.shape)), torch.flatten(out)):
                adj[int(o)].append(int(i))
        return {k: tuple(v) for k, v in adj.items()}


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
    'AddBackward1': AddBackward1,
    'SubBackward0': SubBackward0,
    'SliceBackward0': SliceBackward0,
    'SelectBackward0': SelectBackward0,
    'SqueezeBackward1': SqueezeBackward1,
    'UnsqueezeBackward0': UnsqueezeBackward0,
    'StackBackward0': StackBackward0,
    'CatBackward0': CatBackward0,
    'UnbindBackward0': UnbindBackward0,
    'UnsafeSplitBackward0': UnsafeSplitBackward0,
    'ViewBackward0': ViewBackward0,
    'UnsafeViewBackward0': UnsafeViewBackward0,
    'ReshapeAliasBackward0': ReshapeAliasBackward0,
    'AsStridedBackward0': AsStridedBackward0,
    'AsStridedBackward1': AsStridedBackward1,
    'ConvolutionBackward0': ConvolutionBackward0,
    'SqueezeBackward4': SqueezeBackward1,
    'SqueezeBackward3': SqueezeBackward1,
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
    'SqrtBackward0': SqrtBackward0,
    'PreluKernelBackward0': PreluKernelBackward0,
    'PreluBackward0': PreluBackward0,
    'GluBackward0': GluBackward0,
    'AddmmBackward0': AddmmBackward0,
    'TransposeBackward0': TransposeBackward0,
    'CloneBackward0': CloneBackward0,
    'BmmBackward0': BmmBackward0,
    'SplitBackward0': SplitBackward0,
    'NativeLayerNormBackward0': NativeLayerNormBackward0,
    'NativeGroupNormBackward0': NativeGroupNormBackward0,
    'ReluBackward0': ReluBackward0,
    'PowBackward0': PowBackward0,
    'SignBackward0': SignBackward0,
    'RepeatBackward0': RepeatBackward0,
    'NativeBatchNormBackward0': NativeBatchNormBackward0,
    'NormBackward1': NormBackward1,
    'SumBackward1': SumBackward1,
    'MeanBackward1': MeanBackward1,
    'ExpandBackward0': ExpandBackward0,
    'MeanBackward0': MeanBackward0,
    'AvgPool2DBackward0': AvgPool2DBackward0,
    'AvgPool3DBackward0': AvgPool3DBackward0,
    'AdaptiveAvgPool2DBackward0': AdaptiveAvgPool2DBackward0,
    'AdaptiveAvgPool3DBackward0': AdaptiveAvgPool3DBackward0,
    'MaxPool2DWithIndicesBackward0': MaxPool2DWithIndicesBackward0,
    'MaxPool3DWithIndicesBackward0': MaxPool3DWithIndicesBackward0,
    'AdaptiveMaxPool2DBackward0': AdaptiveMaxPool2DBackward0,
    'AdaptiveMaxPool3DBackward0': AdaptiveMaxPool3DBackward0,
    'EmbeddingBackward0': EmbeddingBackward0,
    'PixelShuffleBackward0': PixelShuffleBackward0,
    'PixelUnshuffleBackward0': PixelUnshuffleBackward0,
    'ConstantPadNdBackward0': ConstantPadNdBackward0,
    'L1LossBackward0': L1LossBackward0,
    'HardtanhBackward0': HardtanhBackward0,
    'DivBackward0': DivBackward0,
    'HardswishBackward0': HardswishBackward0,
    'SiluBackward0': SiluBackward0,
    'PermuteBackward0': PermuteBackward0,
    'IndexCopyBackward0': IndexCopyBackward0,
    'GatherBackward0': GatherBackward0,
    'SqueezeBackward0': SqueezeBackward0,
    'IndexSelectBackward0': IndexSelectBackward0,
    'RollBackward0': RollBackward0,
    'IndexBackward0': IndexBackward0,
    'Im2ColBackward0': Im2ColBackward0,
    'Col2ImBackward0': Col2ImBackward0,
    'UpsampleBilinear2DBackward1': UpsampleBilinear2DBackward1,
    'UpsampleNearest2DBackward1': UpsampleNearest2DBackward1,
    'ClampBackward1': ClampBackward1,
    'ClampMinBackward0': ClampMinBackward0,
    'ExpBackward0': ExpBackward0,
    'SplitWithSizesBackward0': SplitWithSizesBackward0,
    'IndexPutBackward0': IndexPutBackward0,
    'RsqrtBackward0': RsqrtBackward0,
    'WhereBackward0': WhereBackward0,
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
