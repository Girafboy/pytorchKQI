import torch
import torch.nn as nn
import kqinn as kqinn


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers1 = kqinn.Sequential(
            kqinn.Linear(in_features = 784, out_features = 512, bias=False),
            kqinn.Linear(in_features = 512, out_features = 512, bias=False),
        )
        self.layers2 = kqinn.Sequential(
            kqinn.Linear(in_features = 512, out_features = 10, bias=False),
        )
    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)

        return x

    def graph_size(self, x):
        # 自上而下
        x, W = self.layers1.graph_size(x, W=0)
        x, W = self.layers2.graph_size(x, W)
        
        return x, W
    
    def Kqi(self, alpha_volumes, W):
        # 自下而上
        alpha_volumes, kqi = self.layers2.Kqi(alpha_volumes, W)
        alpha_volumes, kqi = self.layers1.Kqi(alpha_volumes, W)

        return alpha_volumes, kqi


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers1 = kqinn.Sequential(
            # 1x28x28
            kqinn.Conv2d(in_channels=1,out_channels=2,kernel_size=3,stride=1,padding=1,bias=False),
            kqinn.ReLU(inplace=True),
            kqinn.MaxPool2d(kernel_size=2, stride=2),

            kqinn.Conv2d(in_channels=2,out_channels=3,kernel_size=3,stride=1,padding=1,bias=False),
            kqinn.ReLU(inplace=True),
            kqinn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.layers2 = kqinn.Sequential(
            # 2x14x14
            kqinn.Linear(in_features = 3*7*7, out_features = 100, bias=False),
            kqinn.Linear(in_features = 100, out_features = 10, bias=False),
        )

    def forward(self, x):
        x = self.layers1(x)
        x = x.flatten()
        x = self.layers2(x)

        return x
    
    def graph_size(self, x):
        # 自上而下
        x, W = self.layers1.graph_size(x, W=0)
        x = x.flatten()
        x, W = self.layers2.graph_size(x, W)

        return x, W
    
    def Kqi(self, alpha_volumes, W):
        # 自下而上
        alpha_volumes, kqi = self.layers2.Kqi(alpha_volumes, W)
        alpha_volumes, kqi = self.layers1.Kqi(alpha_volumes, W)

        return alpha_volumes, kqi


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

    # MLP
    mlp = MLP()
    x = torch.randn(1*28*28)
    kqi = kqinn.caculate_Kqi(mlp, x)
    print("MLP: ", kqi)

    # CNN
    cnn = CNN()
    x = torch.randn(1,28,28)
    k = kqinn.caculate_Kqi(cnn, x)
    print("CNN: ", k)

    # AlexNet
    alexnet = AlexNet()
    x = torch.randn(3,224,224)
    kqi = kqinn.caculate_Kqi(alexnet, x)
    print("AlexNet: ", kqi)
