import torch

from .function_base import FuncBase as FB
from typing import Tuple, Dict


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


class AddBackward0(OnetoOneMapping):
    pass


class SubBackward0(OnetoOneMapping):
    pass


__functions_mapping = {
    'torch::autograd::AccumulateGrad': AccumulateGrad,
    'TBackward0': TBackward0,
    'MvBackward0': MvBackward0,
    'MmBackward0': MmBackward0,
    'AddBackward0': AddBackward0,
    'SubBackward0': SubBackward0
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
