import torch
import torchvision
import torchKQI
import pandas as pd
import os


def task_ImageClassification():
    x = torch.randn(1, 3, 224, 224)

    model_fns = [
        torchvision.models.alexnet,
        torchvision.models.convnext_tiny, torchvision.models.convnext_small, torchvision.models.convnext_base, torchvision.models.convnext_large,
        torchvision.models.densenet121, torchvision.models.densenet161, torchvision.models.densenet169, torchvision.models.densenet201,
        torchvision.models.efficientnet_b0, torchvision.models.efficientnet_b1, torchvision.models.efficientnet_b2, torchvision.models.efficientnet_b3, torchvision.models.efficientnet_b4, torchvision.models.efficientnet_b5, torchvision.models.efficientnet_b6, torchvision.models.efficientnet_b7, torchvision.models.efficientnet_v2_s, torchvision.models.efficientnet_v2_m, torchvision.models.efficientnet_v2_l,
        torchvision.models.googlenet,
        torchvision.models.inception_v3,
        # torchvision.models.maxvit_t,
        torchvision.models.mnasnet0_5, torchvision.models.mnasnet0_75, torchvision.models.mnasnet1_0, torchvision.models.mnasnet1_3,
        torchvision.models.mobilenet_v2,
        torchvision.models.mobilenet_v3_large, torchvision.models.mobilenet_v3_small,
        torchvision.models.regnet_y_400mf, torchvision.models.regnet_y_800mf, torchvision.models.regnet_y_1_6gf, torchvision.models.regnet_y_3_2gf, torchvision.models.regnet_y_8gf, torchvision.models.regnet_y_16gf, torchvision.models.regnet_y_32gf, torchvision.models.regnet_y_128gf, torchvision.models.regnet_x_400mf, torchvision.models.regnet_x_800mf, torchvision.models.regnet_x_1_6gf, torchvision.models.regnet_x_3_2gf, torchvision.models.regnet_x_8gf, torchvision.models.regnet_x_16gf, torchvision.models.regnet_x_32gf,
        torchvision.models.resnet18, torchvision.models.resnet34, torchvision.models.resnet50, torchvision.models.resnet101, torchvision.models.resnet152,
        torchvision.models.resnext50_32x4d, torchvision.models.resnext101_32x8d, torchvision.models.resnext101_64x4d,
        torchvision.models.wide_resnet50_2, torchvision.models.wide_resnet101_2,
        torchvision.models.shufflenet_v2_x0_5, torchvision.models.shufflenet_v2_x1_0, torchvision.models.shufflenet_v2_x1_5, torchvision.models.shufflenet_v2_x2_0,
        torchvision.models.squeezenet1_0, torchvision.models.squeezenet1_1,
        torchvision.models.swin_t, torchvision.models.swin_s, torchvision.models.swin_b,  # torchvision.models.swin_v2_t, torchvision.models.swin_v2_s, torchvision.models.swin_v2_b,
        torchvision.models.vgg11, torchvision.models.vgg11_bn, torchvision.models.vgg13, torchvision.models.vgg13_bn, torchvision.models.vgg16, torchvision.models.vgg16_bn, torchvision.models.vgg19, torchvision.models.vgg19_bn,
        torchvision.models.vit_b_16, torchvision.models.vit_b_32, torchvision.models.vit_l_16, torchvision.models.vit_l_32, torchvision.models.vit_h_14
    ]

    results_file = 'model_results.csv'
    errors_file = 'model_errors.csv'

    if not os.path.exists(results_file):
        pd.DataFrame(columns=['Model Name', 'KQI']).to_csv(results_file, index=False)
    if not os.path.exists(errors_file):
        pd.DataFrame(columns=['Model Name', 'Error']).to_csv(errors_file, index=False)

    for model_fn in model_fns:
        if model_fn.__name__ in pd.read_csv(results_file)['Model Name'].values:
            continue
        try:
            model = model_fn().eval()
            kqi = torchKQI.KQI(model, x).item()
            result = pd.DataFrame([[model_fn.__name__, kqi]], columns=['Model Name', 'KQI'])
            result.to_csv(results_file, mode='a', header=False, index=False)
        except Exception as e:
            error = pd.DataFrame([[model_fn.__name__, str(e)]], columns=['Model Name', 'Error'])
            error.to_csv(errors_file, mode='a', header=False, index=False)


def task_SemanticSegmentation():
    x = torch.randn(1, 3, 224, 224)

    model_fns = [
        torchvision.models.segmentation.deeplabv3_mobilenet_v3_large, torchvision.models.segmentation.deeplabv3_resnet50, torchvision.models.segmentation.deeplabv3_resnet101,
        torchvision.models.segmentation.deeplabv3_resnet50,
        torchvision.models.segmentation.deeplabv3_resnet101,
    ]

    results_file = 'model_results.csv'
    errors_file = 'model_errors.csv'

    if not os.path.exists(results_file):
        pd.DataFrame(columns=['Model Name', 'KQI']).to_csv(results_file, index=False)
    if not os.path.exists(errors_file):
        pd.DataFrame(columns=['Model Name', 'Error']).to_csv(errors_file, index=False)

    for model_fn in model_fns:
        if model_fn.__name__ in pd.read_csv(results_file)['Model Name'].values:
            continue
        try:
            model = model_fn().eval()
            kqi = torchKQI.KQI(model, x, lambda model, x: model(x)['out']).item()
            result = pd.DataFrame([[model_fn.__name__, kqi]], columns=['Model Name', 'KQI'])
            result.to_csv(results_file, mode='a', header=False, index=False)
        except Exception as e:
            error = pd.DataFrame([[model_fn.__name__, str(e)]], columns=['Model Name', 'Error'])
            error.to_csv(errors_file, mode='a', header=False, index=False)


def task_ObjectDetection():
    x = torch.randn(1, 3, 224, 224)

    model_fns = [
        # torchvision.models.detection.fasterrcnn_resnet50_fpn, torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn, torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn, torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
        torchvision.models.detection.fcos_resnet50_fpn,
        torchvision.models.detection.retinanet_resnet50_fpn, torchvision.models.detection.retinanet_resnet50_fpn_v2,
        torchvision.models.detection.ssd300_vgg16,
        torchvision.models.detection.ssdlite320_mobilenet_v3_large,
    ]

    results_file = 'model_results.csv'
    errors_file = 'model_errors.csv'

    if not os.path.exists(results_file):
        pd.DataFrame(columns=['Model Name', 'KQI']).to_csv(results_file, index=False)
    if not os.path.exists(errors_file):
        pd.DataFrame(columns=['Model Name', 'Error']).to_csv(errors_file, index=False)

    for model_fn in model_fns:
        if model_fn.__name__ in pd.read_csv(results_file)['Model Name'].values:
            continue
        try:
            model = model_fn().eval()
            kqi = torchKQI.KQI(model, x, lambda model, x: model(x)[0]['boxes']).item()
            result = pd.DataFrame([[model_fn.__name__, kqi]], columns=['Model Name', 'KQI'])
            result.to_csv(results_file, mode='a', header=False, index=False)
        except Exception as e:
            error = pd.DataFrame([[model_fn.__name__, str(e)]], columns=['Model Name', 'Error'])
            error.to_csv(errors_file, mode='a', header=False, index=False)


def task_VideoClassification():
    x = torch.randn(1, 3, 3, 224, 224)

    model_fns = [
        # torchvision.models.video.mvit_v1_b, torchvision.models.video.mvit_v2_s,
        torchvision.models.video.r3d_18, torchvision.models.video.mc3_18, torchvision.models.video.r2plus1d_18,
        # torchvision.models.video.s3d,
        # torchvision.models.video.swin3d_t, torchvision.models.video.swin3d_s, torchvision.models.video.swin3d_b

    ]

    results_file = 'model_results.csv'
    errors_file = 'model_errors.csv'

    if not os.path.exists(results_file):
        pd.DataFrame(columns=['Model Name', 'KQI']).to_csv(results_file, index=False)
    if not os.path.exists(errors_file):
        pd.DataFrame(columns=['Model Name', 'Error']).to_csv(errors_file, index=False)

    for model_fn in model_fns:
        if model_fn.__name__ in pd.read_csv(results_file)['Model Name'].values:
            continue
        try:
            model = model_fn().eval()
            kqi = torchKQI.KQI(model, x, lambda model, x: model(x)['out']).item()
            result = pd.DataFrame([[model_fn.__name__, kqi]], columns=['Model Name', 'KQI'])
            result.to_csv(results_file, mode='a', header=False, index=False)
        except Exception as e:
            error = pd.DataFrame([[model_fn.__name__, str(e)]], columns=['Model Name', 'Error'])
            error.to_csv(errors_file, mode='a', header=False, index=False)


if __name__ == '__main__':
    task_ImageClassification()
    task_SemanticSegmentation()
    task_ObjectDetection()
    task_VideoClassification()
