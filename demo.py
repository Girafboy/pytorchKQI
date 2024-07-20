import torch
import torchKQI
import pandas as pd
from torchvision.models import *
import os

def task_ImageClassification():
    x = torch.randn(1, 3, 224, 224)
    
    model_names = [
        'alexnet',
        'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
        'densenet121', 'densenet161', 'densenet169', 'densenet201',
        "efficientnet_b0","efficientnet_b1","efficientnet_b2","efficientnet_b3","efficientnet_b4","efficientnet_b5","efficientnet_b6","efficientnet_b7","efficientnet_v2_s","efficientnet_v2_m","efficientnet_v2_l",
        'googlenet',
        'inception_v3',
        "maxvit_t",
        "mnasnet0_5","mnasnet0_75","mnasnet1_0","mnasnet1_3",
        'mobilenet_v2',
        "mobilenet_v3_large","mobilenet_v3_small",
        "regnet_y_400mf","regnet_y_800mf","regnet_y_1_6gf","regnet_y_3_2gf","regnet_y_8gf","regnet_y_16gf","regnet_y_32gf","regnet_y_128gf","regnet_x_400mf","regnet_x_800mf","regnet_x_1_6gf","regnet_x_3_2gf","regnet_x_8gf","regnet_x_16gf","regnet_x_32gf",
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        "resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d",
        "wide_resnet50_2", "wide_resnet101_2",
        'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
        "squeezenet1_0", "squeezenet1_1",
        "swin_t","swin_s","swin_b","swin_v2_t","swin_v2_s","swin_v2_b",
        "vgg11","vgg11_bn","vgg13","vgg13_bn","vgg16","vgg16_bn","vgg19","vgg19_bn",
        'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14'
    ]

    results_file = 'model_results.csv'
    errors_file = 'model_errors.csv'

    if not os.path.exists(results_file):
        pd.DataFrame(columns=['Model Name', 'KQI']).to_csv(results_file, index=False)
    if not os.path.exists(errors_file):
        pd.DataFrame(columns=['Model Name', 'Error']).to_csv(errors_file, index=False)

    for model_name in model_names:
        if model_name in pd.read_csv(results_file)['Model Name'].values:
            continue
        try:
            model_fn = globals()[model_name]
            model = model_fn().eval()
            kqi = torchKQI.KQI(model, x).item()
            result = pd.DataFrame([[model_name, kqi]], columns=['Model Name', 'KQI'])
            result.to_csv(results_file, mode='a', header=False, index=False)
        except Exception as e:
            error = pd.DataFrame([[model_name, str(e)]], columns=['Model Name', 'Error'])
            error.to_csv(errors_file, mode='a', header=False, index=False)


if __name__ == '__main__':
    task_ImageClassification()
