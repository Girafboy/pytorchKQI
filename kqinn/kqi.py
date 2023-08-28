import torch
import numpy as np 
import logging


class KQI:
    W = 0
    kqi = 0

    def KQI(self, x: torch.Tensor) -> float:
        KQI.W = np.prod(x.shape)
        KQI.kqi = 0

        x = self.KQIforward(x)
        volumes = self.KQIbackward(torch.zeros_like(x))
        KQI.kqi += self.KQI_formula(volumes, torch.tensor(KQI.W))
        
        logging.debug(f'Root: KQI={KQI.kqi}, node={np.product(volumes.shape)}, volume={volumes.sum()}')
        logging.debug(f'Total volume = {KQI.W}')
        return KQI.kqi


    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required KQIforward function')
    

    def KQIbackward(self, volumes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required KQIbackward function')
    

    def KQI_formula(self, volumes: torch.Tensor, volumes_forward: torch.Tensor) -> float:
        if volumes.shape != volumes_forward.shape and volumes_forward.dim():
            raise ValueError(f'Shape of volumes {volumes.shape} is incompatible with volumes_forward {volumes_forward.shape}')
        if volumes.dim():
            pos = torch.where(volumes != 0)
            if volumes_forward.dim():
                return (- volumes[pos] / KQI.W * np.log2(volumes[pos] / volumes_forward[pos])).sum()
            else:
                return (- volumes[pos] / KQI.W * np.log2(volumes[pos] / volumes_forward)).sum()
        else:
            return - volumes / KQI.W * np.log2(volumes / volumes_forward) if volumes else 0