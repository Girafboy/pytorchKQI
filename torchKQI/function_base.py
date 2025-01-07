import psutil
import torch
import logging
import tqdm
import ctypes
from typing import Tuple, Dict
from functools import wraps

logging.basicConfig(level=logging.DEBUG, filename='debug.log', filemode='w', format="%(asctime)s - %(levelname)s - %(message)s")
bar = tqdm.tqdm()
device = torch.device('cpu')


def to_device(tensor, device=device):
    if isinstance(tensor, tuple):
        return tuple(t.to(device) if t is not None else t for t in tensor)
    else:
        return tensor.to(device) if tensor is not None else tensor


def grad_fn_attr_info(grad_fn):
    return {attr: grad_fn.__getattribute__(attr).shape 
            if isinstance(grad_fn.__getattribute__(attr), torch.Tensor) 
            else grad_fn.__getattribute__(attr) 
            for attr in dir(grad_fn) if "_saved_" in attr}


class GradFn:
    def __init__(self, grad_fn):
        self.grad_fn = grad_fn
        self.ctype = torch.rand(1).element_size()

    def __call__(self, *args):
        args = to_device(args, device=torch.device('cpu'))
        try:
            res = self.grad_fn(*args)
        except Exception as err:
            logging.debug(f'ERROR!!! {self.grad_fn}->{self.grad_fn.next_functions} {tuple(a.shape for a in args)} {grad_fn_attr_info(self.grad_fn)}')
            raise(err)
        return to_device(res)
    
    def __getattribute__(self, __name):
        def unsign_to_sign(attr):            
            if self.ctype == 4:
                return ctypes.c_int32(attr).value
            else:
                return ctypes.c_int64(attr).value
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            attr = self.grad_fn.__getattribute__(__name)
            if __name == '_saved_keepdim':
                return bool(attr)
            if __name == '_saved_end':
                return attr
            else:
                if isinstance(attr, int):
                    attr = unsign_to_sign(attr)
                if isinstance(attr, tuple):
                    attr = tuple(unsign_to_sign(a) if isinstance(a, int) else a for a in attr)
                return attr


class FuncBase:
    @staticmethod
    def cell_Volume_Checking(args_in: int, args_out: int):
        def cell_Volume_Checking_decorator(func):
            @wraps(func)
            def wrapped_function(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
                if args_out is not None:
                    assert len(volume_outputs) == args_out, f"{cls.__name__}.cell_Volume must have exactly {args_out} volume_outputs. {grad_fn_attr_info(grad_fn)}"
                
                volume_inputs = to_device(func(cls, GradFn(grad_fn), to_device(volume_outputs)), device=torch.device('cpu'))
                bar.update()

                if args_in is not None:
                    assert len(volume_inputs) == args_in, f"{cls.__name__}.cell_Volume must have exactly {args_in} volume_inputs. {grad_fn_attr_info(grad_fn)}"
                ensure_tuple = lambda result: result if isinstance(result, tuple) else (result,)
                for volume_in, grad_fn_return in zip(volume_inputs, ensure_tuple(GradFn(grad_fn)(*volume_outputs))):
                    if volume_in is not None or grad_fn_return is not None:
                        assert volume_in.shape == grad_fn_return.shape, f"{cls.__name__}.cell_Volume must return the same size of volume_in {volume_in.shape} and grad_fn_return {grad_fn_return.shape}. {grad_fn_attr_info(grad_fn)}"
                logging.debug(f'[{psutil.Process().memory_info().rss/1024**3:.2f} GB] {cls.__name__}.cell_Volume: volume_inputs=[{", ".join([f"{k.sum()} {k.shape}" if k is not None else "None" for k in volume_inputs])}], volume_outputs=[{", ".join([f"{k.sum()} {k.shape}" if k is not None else "None" for k in volume_outputs])}]')
                return volume_inputs
            return wrapped_function
        return cell_Volume_Checking_decorator

    @classmethod
    def cell_Volume(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        raise NotImplementedError(f'Class {cls.__name__} is missing the required cell_Volume function')

    @staticmethod
    def cell_KQI_Checking(args_in: int, args_out: int):
        def cell_KQI_Checking_decorator(func):
            @wraps(func)
            def wrapped_function(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
                if args_out is not None:
                    assert len(volume_outputs) == args_out, f"{cls.__name__}.cell_KQI must have exactly {args_out} volume_outputs. {grad_fn_attr_info(grad_fn)}"
                if args_in is not None:
                    assert len(volume_inputs) == args_in, f"{cls.__name__}.cell_KQI must have exactly {args_in} volume_inputs. {grad_fn_attr_info(grad_fn)}"

                kqis = to_device(func(cls, GradFn(grad_fn), to_device(volume_inputs), to_device(volume_outputs)), device=torch.device('cpu'))
                bar.update()

                assert len(kqis) == len(volume_outputs), f"{cls.__name__}.cell_KQI must return {len(volume_outputs)} kqis, but now return {len(kqis)} kqis. {grad_fn_attr_info(grad_fn)}"
                for kqi, volume_out in zip(kqis, volume_outputs):
                    assert kqi.shape == volume_out.shape, f"{cls.__name__}.cell_KQI must return the same size of volume_output {volume_out.shape} and kqi {kqi.shape}. {grad_fn_attr_info(grad_fn)}"
                logging.debug(f'[{psutil.Process().memory_info().rss/1024**3:.2f} GB] {cls.__name__}.cell_KQI: kqi=[{", ".join([f"{k.sum()}/W {k.shape}" if k is not None else "None" for k in kqis])}]')
                return kqis
            return wrapped_function
        return cell_KQI_Checking_decorator

    @classmethod
    def cell_KQI(cls, grad_fn, volume_inputs: Tuple[torch.Tensor], volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        raise NotImplementedError(f'Class {cls.__name__} is missing the required cell_KQI function')

    @staticmethod
    def cell_Graph_Checking(args_in: int, args_out: int):
        def cell_Graph_Checking_decorator(func):
            @wraps(func)
            def wrapped_function(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
                if args_out is not None:
                    assert len(outputs) == args_out, f"{cls.__name__}.cell_Graph must have exactly {args_out} outputs. {grad_fn_attr_info(grad_fn)}"
                if args_in is not None:
                    assert len(inputs) == args_in, f"{cls.__name__}.cell_Graph must have exactly {args_in} inputs. {grad_fn_attr_info(grad_fn)}"

                adj = func(cls, GradFn(grad_fn), to_device(inputs), to_device(outputs))
                bar.update()

                logging.debug(f'[{psutil.Process().memory_info().rss/1024**3:.2f} GB] {cls.__name__}.cell_Graph: nodes={len(adj)}, edges={sum(map(len, adj.values()))}')
                return adj
            return wrapped_function
        return cell_Graph_Checking_decorator

    @classmethod
    def cell_Graph(cls, grad_fn, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> Dict[int, Tuple[int]]:
        raise NotImplementedError(f'Class {cls.__name__} is missing the required cell_Graph function')

    @staticmethod
    def temporary_KQI(volume: torch.Tensor, volume_backward: torch.Tensor) -> torch.Tensor:
        '''
        This function provides a way to compute the temporary KQI (without divide by W).
        Remember to divide by W before returning the final KQI value.
        The volume parameter and the volume_backward parameter should be the same shape, unless the volume_backward is a scalar.
        '''
        if volume_backward.dim() != 0 and volume.shape != volume_backward.shape:
            raise ValueError(f'Shape of volume {volume.shape} is incompatible with volume_backward {volume_backward.shape}')

        if volume.eq(0).any():
            volume = volume.clone()
            if volume.dim() == 0:
                volume = volume_backward
            else:
                if volume_backward.dim() == 0:
                    volume[torch.where(volume == 0)] = volume_backward
                else:
                    volume[torch.where(volume == 0)] = volume_backward[torch.where(volume == 0)]
        return - volume * torch.log2(volume / volume_backward)
