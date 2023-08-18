import torch
import torch.nn as nn
import kqinn as kqinn





class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv_layers = kqinn.Sequential(
            kqinn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            kqinn.ReLU(inplace=True),
            kqinn.MaxPool2d(kernel_size=3, stride=2),
            kqinn.Conv2d(64, 192, kernel_size=5, padding=2),
            kqinn.ReLU(inplace=True),
            kqinn.MaxPool2d(kernel_size=3, stride=2),

            kqinn.Conv2d(192, 384, kernel_size=3, padding=1),
            kqinn.ReLU(inplace=True),
            kqinn.Conv2d(384, 256, kernel_size=3, padding=1),
            kqinn.ReLU(inplace=True),
            kqinn.Conv2d(256, 256, kernel_size=3, padding=1),
            kqinn.ReLU(inplace=True),
            kqinn.MaxPool2d(kernel_size=3, stride=2),

        )
        self.fc_layers = kqinn.Sequential(
            kqinn.Linear(256 * 6 * 6, 4096),
            kqinn.ReLU(inplace=True),
            kqinn.Dropout(),
            kqinn.Linear(4096, 4096),
            kqinn.ReLU(inplace=True),
            kqinn.Dropout(),
            kqinn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
    def graph_size(self, x):
        # 自上而下
        x, W = self.conv_layers.graph_size(x, W=0)
        x = x.flatten()
        x, W = self.fc_layers.graph_size(x, W)

        return x, W
    
    def Kqi(self, alpha_volumes, W):
        # 自下而上
        alpha_volumes, kqi = self.fc_layers.Kqi(alpha_volumes, W)
        alpha_volumes, kqi = self.conv_layers.Kqi(alpha_volumes, W)

        return alpha_volumes, kqi


if __name__ == '__main__':
    # AlexNet
    alexnet = AlexNet()
    x = torch.randn(3,224,224)
    kqi = kqinn.caculate_Kqi(alexnet, x)
    print("AlexNet: ", kqi)
