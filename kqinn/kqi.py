import torch
import numpy as np 


class KQI:
    W = 0

    def KQI(self, x: torch.Tensor) -> float:
        KQI.W = 0
        x = self.KQIforward(x)

        volumes, kqi = self.KQIbackward(torch.zeros_like(x), 0)
        return kqi + sum(map(lambda V: -V / KQI.W * np.log2(V / KQI.W), volumes))


    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"KQIforward\" function")
    

    def KQIbackward(self, volumes: torch.Tensor, kqi: float) -> (torch.Tensor, float):
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"KQIbackward\" function")