import numpy as np 
import torch.nn as nn


class ReLU(nn.ReLU):
    def non_callable_method(self):
        raise TypeError("This method is not callable.")

