import psutil
import torch
import logging
import tqdm
import ctypes
from typing import Tuple, Dict
from functools import wraps
import multiprocessing
import itertools

logging.basicConfig(level=logging.DEBUG, filename='debug.log', filemode='w', format="%(asctime)s - %(levelname)s - %(message)s")


class Context:
    bar = tqdm.tqdm()
    device = [torch.device('cpu')]
    grad_fn_info = {}
    pool = None

    @staticmethod
    def to_device(tensor, device):
        if isinstance(tensor, tuple):
            return tuple(t.to(device) if t is not None else t for t in tensor)
        else:
            return tensor.to(device) if tensor is not None else tensor

    @staticmethod
    def hook_factory(grad_fn):
        def hook(grad_inputs, grad_outputs):
            Context.grad_fn_info[grad_fn] = {'input': tuple((input.shape, input.dtype) if input is not None else input for input in grad_inputs),
                                             'output': tuple((output.shape, output.dtype) if output is not None else output for output in grad_outputs)}
        return hook

    @staticmethod
    def grad_fn_attr_info(grad_fn):
        return {'Inputs': Context.grad_fn_info[grad_fn]['input'],
                'Outputs': Context.grad_fn_info[grad_fn]['output'],
                'Attributes': {attr: grad_fn.__getattribute__(attr).shape if isinstance(grad_fn.__getattribute__(attr), torch.Tensor)
                               else tuple(a.shape if isinstance(a, torch.Tensor) else a for a in grad_fn.__getattribute__(attr)) if isinstance(grad_fn.__getattribute__(attr), tuple)
                               else grad_fn.__getattribute__(attr)
                               for attr in dir(grad_fn) if '_saved' in attr and '_raw' not in attr}
                }

    @staticmethod
    def init_pool():
        multiprocessing.set_start_method('spawn', True)
        Context.pool = multiprocessing.Pool(len(Context.device))

    @staticmethod
    def parallel_map(func, iterable):
        return map(lambda res: res.to(Context.device[0]), Context.pool.imap(func, map(lambda args, d: tuple(arg.to(d) if isinstance(arg, torch.Tensor) else arg for arg in args), iterable, itertools.cycle(Context.device))))


class GradFn:
    def __init__(self, grad_fn):
        self.grad_fn = grad_fn
        self.ctype = torch.rand(1).element_size()

    def __call__(self):
        return tuple(torch.zeros(size=input[0], dtype=input[1], device=Context.device[0]) if input is not None else input for input in Context.grad_fn_info[self.grad_fn]['input'])

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
                    assert len(volume_outputs) == args_out, f"{cls.__name__}.cell_Volume must have exactly {args_out} volume_outputs. {Context.grad_fn_attr_info(grad_fn)}"

                try:
                    volume_inputs = Context.to_device(func(cls, GradFn(grad_fn), Context.to_device(volume_outputs, device=Context.device[0])), device=torch.device('cpu'))
                except Exception as err:
                    logging.debug(f'ERROR!!! {cls.__name__}({id(grad_fn)}<-{",".join(map(lambda k: str(id(k[0])), grad_fn.next_functions))}).cell_Volume\n \
                              \t\t\t\tvolume_outputs=[{", ".join([f"{k.sum()} {k.shape}" if k is not None else "None" for k in volume_outputs])}]\n \
                              \t\t\t\tgrad_fn={Context.grad_fn_attr_info(grad_fn)}')
                    raise err
                Context.bar.update()

                if args_in is not None:
                    assert len(volume_inputs) == args_in, f"{cls.__name__}.cell_Volume must have exactly {args_in} volume_inputs. {Context.grad_fn_attr_info(grad_fn)}"
                for volume_in, true_shape in zip(volume_inputs, Context.grad_fn_info[grad_fn]['input']):
                    if volume_in is not None or true_shape is not None:
                        assert volume_in.shape == true_shape[0], f"{cls.__name__}.cell_Volume must return the same size of volume_in {volume_in.shape} as true_shape {true_shape[0]}. {Context.grad_fn_attr_info(grad_fn)}"
                logging.debug(f'{psutil.Process().memory_info().rss/1024**3:.2f} GB - {cls.__name__}({id(grad_fn)}<-{",".join(map(lambda k: str(id(k[0])), grad_fn.next_functions))}).cell_Volume\n \
                              \t\t\t\tvolume_inputs=[{", ".join([f"{k.sum()} {k.shape}" if k is not None else "None" for k in volume_inputs])}]\n \
                              \t\t\t\tvolume_outputs=[{", ".join([f"{k.sum()} {k.shape}" if k is not None else "None" for k in volume_outputs])}]\n \
                              \t\t\t\tgrad_fn={Context.grad_fn_attr_info(grad_fn)}')
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
                    assert len(volume_outputs) == args_out, f"{cls.__name__}.cell_KQI must have exactly {args_out} volume_outputs. {Context.grad_fn_attr_info(grad_fn)}"
                if args_in is not None:
                    assert len(volume_inputs) == args_in, f"{cls.__name__}.cell_KQI must have exactly {args_in} volume_inputs. {Context.grad_fn_attr_info(grad_fn)}"

                try:
                    kqis = Context.to_device(func(cls, GradFn(grad_fn), Context.to_device(volume_inputs, device=Context.device[0]), Context.to_device(volume_outputs, device=Context.device[0])), device=torch.device('cpu'))
                except Exception as err:
                    logging.debug(f'ERROR!!! {cls.__name__}({id(grad_fn)}<-{",".join(map(lambda k: str(id(k[0])), grad_fn.next_functions))}).cell_KQI\n \
                              \t\t\t\tvolume_inputs=[{", ".join([f"{k.sum()} {k.shape}" if k is not None else "None" for k in volume_inputs])}]\n \
                              \t\t\t\tvolume_outputs=[{", ".join([f"{k.sum()} {k.shape}" if k is not None else "None" for k in volume_outputs])}]\n \
                              \t\t\t\tgrad_fn={Context.grad_fn_attr_info(grad_fn)}')
                    raise err
                Context.bar.update()

                assert len(kqis) == len(volume_outputs), f"{cls.__name__}.cell_KQI must return {len(volume_outputs)} kqis, but now return {len(kqis)} kqis. {Context.grad_fn_attr_info(grad_fn)}"
                for kqi, volume_out in zip(kqis, volume_outputs):
                    assert kqi.shape == volume_out.shape, f"{cls.__name__}.cell_KQI must return the same size of volume_output {volume_out.shape} and kqi {kqi.shape}. {Context.grad_fn_attr_info(grad_fn)}"
                logging.debug(f'{psutil.Process().memory_info().rss/1024**3:.2f} GB - {cls.__name__}({id(grad_fn)}<-{",".join(map(lambda k: str(id(k[0])), grad_fn.next_functions))}).cell_KQI\n \
                              \t\t\t\tkqi=[{", ".join([f"{k.sum()}/W {k.shape}" if k is not None else "None" for k in kqis])}]\n \
                              \t\t\t\tgrad_fn={Context.grad_fn_attr_info(grad_fn)}')
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
                    assert len(outputs) == args_out, f"{cls.__name__}.cell_Graph must have exactly {args_out} outputs. {Context.grad_fn_attr_info(grad_fn)}"
                if args_in is not None:
                    assert len(inputs) == args_in, f"{cls.__name__}.cell_Graph must have exactly {args_in} inputs. {Context.grad_fn_attr_info(grad_fn)}"

                try:
                    adj = func(cls, GradFn(grad_fn), Context.to_device(inputs, device=Context.device[0]), Context.to_device(outputs, device=Context.device[0]))
                except Exception as err:
                    logging.debug(f'ERROR!!! {cls.__name__}({id(grad_fn)}<-{",".join(map(lambda k: str(id(k[0])), grad_fn.next_functions))}).cell_Graph\n \
                              \t\t\t\tinputs=[{", ".join([f"{k.sum()} {k.shape}" if k is not None else "None" for k in inputs])}]\n \
                              \t\t\t\toutputs=[{", ".join([f"{k.sum()} {k.shape}" if k is not None else "None" for k in outputs])}]\n \
                              \t\t\t\tgrad_fn={Context.grad_fn_attr_info(grad_fn)}')
                    raise err
                Context.bar.update()

                logging.debug(f'{psutil.Process().memory_info().rss/1024**3:.2f} GB - {cls.__name__}({id(grad_fn)}<-{",".join(map(lambda k: str(id(k[0])), grad_fn.next_functions))}).cell_Graph\n \
                              \t\t\t\tnodes={len(adj)}, edges={sum(map(len, adj.values()))}\n \
                              \t\t\t\tgrad_fn={Context.grad_fn_attr_info(grad_fn)}')
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

        volume = torch.where(volume == 0, volume_backward, volume)
        return - volume * torch.log2(volume / volume_backward)
