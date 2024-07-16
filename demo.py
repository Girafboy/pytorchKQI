import torch
import torchKQI
from torchvision.models import alexnet, densenet121, googlenet, mobilenet_v2, resnet152, shufflenet_v2_x2_0, vit_h_14


def task_ImageClassification():
    x = torch.randn(1, 3, 224, 224)

    print(f'AlexNet: KQI = {torchKQI.KQI(alexnet(), x)}')
    print(f'DenseNet121: KQI = {torchKQI.KQI(densenet121(), x)}')
    print(f'GoogleNet: KQI = {torchKQI.KQI(googlenet(), x)}')
    print(f'MobileNet_v2: KQI = {torchKQI.KQI(mobilenet_v2(), x)}')
    print(f'ResNet152: KQI = {torchKQI.KQI(resnet152(), x)}')
    print(f'ShuffleNet_v2_x2_0: KQI = {torchKQI.KQI(shufflenet_v2_x2_0(), x)}')
    print(f'vit_h_14: KQI = {torchKQI.KQI(vit_h_14(), x)}')


if __name__ == '__main__':
    task_ImageClassification()
