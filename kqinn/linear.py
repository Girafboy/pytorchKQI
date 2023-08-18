import torch
import numpy as np

from .kqi import KQI


class Linear(torch.nn.Linear, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        KQI.W = KQI.W + self.in_features * self.out_features 
        return self.forward(x)


    def KQIbackward(self, volumes: torch.Tensor, kqi: float) -> (torch.Tensor, float):
        V_ = self.out_features + (volumes / self.in_features).sum()
        
        return torch.ones(self.in_features) * V_, kqi + sum(map(lambda V: -V / KQI.W * np.log2(V / self.in_features / V_) if V else 0, volumes))
