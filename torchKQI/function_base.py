import torch
import logging

from typing import Tuple, Dict
from functools import wraps

logging.basicConfig(level=logging.DEBUG)


class FuncBase:
    @staticmethod
    def cell_Volume_Checking(args_in: int, args_out: int):
        def cell_Volume_Checking_decorator(func):
            @wraps(func)
            def wrapped_function(cls, grad_fn, volume_outputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
                if args_out is not None:
                    assert len(volume_outputs) == args_out, f"{cls.__name__}.cell_Volume must have exactly {args_out} volume_outputs."

                volume_inputs = func(cls, grad_fn, volume_outputs)

                if args_in is not None:
                    assert len(volume_inputs) == args_in, f"{cls.__name__}.cell_Volume must have exactly {args_in} volume_inputs."
                logging.debug(f'{cls.__name__}.cell_Volume: volume_inputs=[{", ".join([f"{k.sum()} {k.shape}" if k is not None else "None" for k in volume_inputs])}], volume_outputs=[{", ".join([f"{k.sum()} {k.shape}" if k is not None else "None" for k in volume_outputs])}]')
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
                    assert len(volume_outputs) == args_out, f"{cls.__name__}.cell_KQI must have exactly {args_out} volume_outputs."
                if args_in is not None:
                    assert len(volume_inputs) == args_in, f"{cls.__name__}.cell_KQI must have exactly {args_in} volume_inputs."

                kqis = func(cls, grad_fn, volume_inputs, volume_outputs)

                assert len(kqis) == len(volume_outputs), f"{cls.__name__}.cell_KQI must return {len(volume_outputs)} kqis, but now return {len(kqis)} kqis."
                logging.debug(f'{cls.__name__}.cell_KQI: kqi=[{", ".join([f"{k.sum()}/W {k.shape}" if k is not None else "None" for k in kqis])}]')
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
                    assert len(outputs) == args_out, f"{cls.__name__}.cell_Graph must have exactly {args_out} outputs."
                if args_in is not None:
                    assert len(inputs) == args_in, f"{cls.__name__}.cell_Graph must have exactly {args_in} inputs."

                adj = func(cls, grad_fn, inputs, outputs)

                logging.debug(f'{cls.__name__}.cell_Graph: nodes={len(adj)}, edges={sum(map(len, adj.values()))}')
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
        volume = volume.clone()
        if volume_backward.dim() == 0:
            volume[torch.where(volume == 0)] = volume_backward
        elif volume.shape == volume_backward.shape:
            volume[torch.where(volume == 0)] = volume_backward[torch.where(volume == 0)]
        else:
            raise ValueError(f'Shape of volume {volume.shape} is incompatible with volume_backward {volume_backward.shape}')
        return - volume * torch.log2(volume / volume_backward)
