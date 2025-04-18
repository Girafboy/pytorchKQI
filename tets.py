import torchKQI
import torch

class SimpleConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Conv2d (in_channels=2,                                                      	out_channels=1, kernel_size=2, bias=True)
        self.layer2 = torch.nn.Flatten()
        self.layer3 = torch.nn.Linear(4,1,bias=False)
       
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

x = torch.randn(1, 2, 3, 3) # 1 sample, 2 channels, 2x2 image
model = SimpleConv()
torchKQI.VisualKQI(model, x)