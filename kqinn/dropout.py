import numpy as np 
import torch.nn as nn


class Dropout(nn.Dropout):
        def non_callable_method(self):
            raise TypeError("This method is not callable.")