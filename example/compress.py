import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchKQI
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
import struct
from tqdm import tqdm
import multiprocessing


if torch.cuda.is_available():
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda:0')
else:
    cpu_num = multiprocessing.cpu_count()
    torch.set_num_threads(cpu_num)


def reduce_precision(tensor, significand_bits, exponent_bits):
    """
    Reduce the precision of a PyTorch tensor by specifying exponent width and significand precision.

    Args:
        tensor (torch.Tensor): The input tensor of dtype float32.
        exponent_bits (int): The number of bits for the exponent (>= 2).
        significand_bits (int): The number of bits for the significand (>= 0).

    Returns:
        torch.Tensor: A new tensor with reduced precision.
    """
    # Ensure valid input for exponent and significand bits
    if exponent_bits < 2:
        raise ValueError("exponent_bits must be >= 2 to represent a valid range.")
    if significand_bits < 0:
        raise ValueError("significand_bits must be >= 0.")

    # Constants for IEEE 754 floating-point representation
    EXPONENT_BIAS = (1 << (exponent_bits - 1)) - 1  # Bias for the given exponent width
    
    # Ensure the tensor is in float32
    tensor = tensor.float()

    # Handle special cases (NaN, Inf, 0)
    is_nan = tensor.isnan()
    is_inf = tensor.isinf()
    zero_mask = tensor == 0

    # Handle nonzero values separately
    nonzero_tensor = tensor.clone()
    nonzero_tensor[zero_mask] = 1.0  # Temporarily replace zero with a non-zero value

    # Decompose tensor into components
    sign = torch.sign(tensor)  # Sign of the value
    abs_tensor = torch.abs(nonzero_tensor)  # Absolute value
    exponent = torch.floor(torch.log2(abs_tensor))  # Compute base-2 exponent
    significand = abs_tensor / (2 ** exponent)  # Normalize value to [1.0, 2.0)

    # Quantize exponent
    exponent = torch.clamp(exponent + EXPONENT_BIAS, 0, (1 << exponent_bits) - 1)  # Clamp to valid range
    quantized_exponent = exponent - EXPONENT_BIAS  # De-bias back for reconstruction

    # Quantize significand
    step = 2 ** (-significand_bits)  # Quantization step size for the significand
    quantized_significand = torch.floor(significand / step) * step

    # Reconstruct the quantized tensor
    quantized_tensor = sign * quantized_significand * (2 ** quantized_exponent)

    # Handle zeros
    quantized_tensor[zero_mask] = 0.0

    # Handle NaN and Inf (propagate as is)
    quantized_tensor[is_nan] = float('nan')
    quantized_tensor[is_inf] = tensor[is_inf]  # Preserve the sign of infinity

    return quantized_tensor


def calculate_accuracy(model, dataloader):
    correct_top1 = 0
    correct_top5 = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            # Top-1 Accuracy
            _, predicted_classes = torch.max(outputs, 1)
            correct_top1 += (predicted_classes == labels).sum().item()

            # Top-5 Accuracy
            _, top5_classes = torch.topk(outputs, 5, dim=1, largest=True, sorted=True)
            top5_classes = top5_classes.cpu().numpy()
            labels = labels.cpu().numpy()
            correct_top5 += sum([label in top5_classes[i] for i, label in enumerate(labels)])

            total_images += labels.size

    top1_accuracy = correct_top1 / total_images
    top5_accuracy = correct_top5 / total_images

    return top1_accuracy, top5_accuracy


def evaluate_model(model, transform, dataset_path='ILSVRC2012_img_val', batch_size=64):
    model.eval()

    val_dataset = datasets.ImageFolder(dataset_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return calculate_accuracy(model, val_loader)


# Example usage:
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def topkqi_mask(statedict_kqi: dict, percentage: float, top=True):
    kqis = torch.cat(tuple(kqi.flatten() for kqi in statedict_kqi.values()), 0)
    num_masked = int(percentage * len(kqis))
    if top:
        topk = torch.tensor(np.partition(kqis.cpu(), -num_masked)[-num_masked]) if abs(num_masked) > 1e-6 else max(kqis)
        cnt = num_masked - sum(kqis > topk)
    else:
        num_remain = len(kqis) - num_masked
        topk = torch.tensor(np.partition(kqis.cpu(), -num_remain)[-num_remain]) if abs(num_remain) > 1e-6 else max(kqis)
        cnt = num_masked - sum(kqis < topk)
    statedict_mask = {}
    for key, value in statedict_kqi.items():
        res = torch.ones(value.shape, dtype=bool)
        if top:
            res[value > topk] = False
        else:
            res[value < topk] = False
        if cnt > 0:
            idx = torch.where(value.flatten() == topk)[0]
            if idx.shape[0] > cnt:
                idx = torch.from_numpy(np.random.choice(idx.cpu().numpy(), int(cnt), replace=False))
            res = res.flatten()
            res[idx] = False
            res = res.reshape(value.shape)
            cnt -= idx.shape[0]
        statedict_mask[key] = res
    return statedict_mask


def mask_model(model_builder, model_weight, per, top, significand_bits=3, exponent_bits=4):
    model = model_builder()
    x = torch.randn(1, 3, 224, 224)
    model_params = {var: name for name, var in model.named_parameters()}
    statedict_kqi = {}
    for grad_fn, kqis in torchKQI.KQI_generator(model, x):
        if 'AccumulateGrad' in grad_fn.name():
            assert len(kqis) == 1
            statedict_kqi[model_params[grad_fn.variable]] = kqis[0][0]
    statedict_mask = topkqi_mask(statedict_kqi, per, top)
    model = model_builder(weights=model_weight)

    state_dict = {}
    compressed_bits = 0
    original_bits = 0
    for key, value in model.state_dict().items():
        if key in statedict_mask:
            decompressed_tensor = reduce_precision(value, significand_bits, exponent_bits)
            state_dict[key] = torch.where(statedict_mask[key], value, decompressed_tensor.reshape(value.shape))
            num_true = torch.broadcast_to(statedict_mask[key], value.shape).sum()
            num_false = value.numel() - num_true
            compressed_bits += num_true * 32 + num_false * (significand_bits + exponent_bits + 1)
            original_bits += value.numel() * 32
        else:
            state_dict[key] = value

    kqi_mask_dict = {key: value * (~statedict_mask[key]) for key, value in statedict_kqi.items()}
    kqi_mask = sum(torch.sum(v).item() for v in kqi_mask_dict.values())
    model.load_state_dict(state_dict=state_dict)
    top1_accuracy, top5_accuracy = evaluate_model(model, transform)

    return model.__class__.__name__, top1_accuracy, top5_accuracy, kqi_mask, original_bits / (10 ** 6), compressed_bits / (10 ** 6)


def output(model_builder, model_weight):
    if not os.path.exists('top8bit.csv'):
        pd.DataFrame(columns=['Name', 'Mask Percentage', 'Top-1 Accuracy', 'Top-5 Accuracy', 'Mask KQI', 'original bits(Mb)', 'compressed bits(Mb)']).to_csv('top8bit.csv', index=False)
    for per in np.arange(0, 1.1, 0.1):
        name, top1_accuracy, top5_accuracy, kqi_mask, original_bits, compressed_bits = mask_model(model_builder, model_weight, per, True, significand_bits=3, exponent_bits=4)
        result = (name, per * 100, top1_accuracy, top5_accuracy, kqi_mask, original_bits, compressed_bits)
        df = pd.DataFrame([result], columns=["Name", "Mask Percentage", "Top-1 Accuracy", "Top-5 Accuracy", "Mask KQI", "original bits(Mb)", "compressed bits(Mb)"])
        df["Mask Percentage"] = df["Mask Percentage"].map(lambda x: f"{x:.1f}%")
        df.to_csv('top8bit.csv', mode='a', header=False, index=False)

    if not os.path.exists('bottom8bit.csv'):
        pd.DataFrame(columns=['Name', 'Mask Percentage', 'Top-1 Accuracy', 'Top-5 Accuracy', 'Mask KQI', 'original bits(Mb)', 'compressed bits(Mb)']).to_csv('bottom8bit.csv', index=False)
    for per in np.arange(0, 1.1, 0.1):
        name, top1_accuracy, top5_accuracy, kqi_mask, original_bits, compressed_bits = mask_model(model_builder, model_weight, per, False, significand_bits=3, exponent_bits=4)
        result = (name, per * 100, top1_accuracy, top5_accuracy, kqi_mask, original_bits, compressed_bits)
        df = pd.DataFrame([result], columns=["Name", "Mask Percentage", "Top-1 Accuracy", "Top-5 Accuracy", "Mask KQI", "original bits(Mb)", "compressed bits(Mb)"])
        df["Mask Percentage"] = df["Mask Percentage"].map(lambda x: f"{x:.1f}%")
        df.to_csv('bottom8bit.csv', mode='a', header=False, index=False)

    if not os.path.exists('top16bit.csv'):
        pd.DataFrame(columns=['Name', 'Mask Percentage', 'Top-1 Accuracy', 'Top-5 Accuracy', 'Mask KQI', 'original bits(Mb)', 'compressed bits(Mb)']).to_csv('top16bit.csv', index=False)
    for per in np.arange(0, 1.1, 0.1):
        name, top1_accuracy, top5_accuracy, kqi_mask, original_bits, compressed_bits = mask_model(model_builder, model_weight, per, True, significand_bits=10, exponent_bits=5)
        result = (name, per * 100, top1_accuracy, top5_accuracy, kqi_mask, original_bits, compressed_bits)
        df = pd.DataFrame([result], columns=["Name", "Mask Percentage", "Top-1 Accuracy", "Top-5 Accuracy", "Mask KQI", "original bits(Mb)", "compressed bits(Mb)"])
        df["Mask Percentage"] = df["Mask Percentage"].map(lambda x: f"{x:.1f}%")
        df.to_csv('top16bit.csv', mode='a', header=False, index=False)

    if not os.path.exists('bottom16bit.csv'):
        pd.DataFrame(columns=['Name', 'Mask Percentage', 'Top-1 Accuracy', 'Top-5 Accuracy', 'Mask KQI', 'original bits(Mb)', 'compressed bits(Mb)']).to_csv('bottom16bit.csv', index=False)
    for per in np.arange(0, 1.1, 0.1):
        name, top1_accuracy, top5_accuracy, kqi_mask, original_bits, compressed_bits = mask_model(model_builder, model_weight, per, False, significand_bits=10, exponent_bits=5)
        result = (name, per * 100, top1_accuracy, top5_accuracy, kqi_mask, original_bits, compressed_bits)
        df = pd.DataFrame([result], columns=["Name", "Mask Percentage", "Top-1 Accuracy", "Top-5 Accuracy", "Mask KQI", "original bits(Mb)", "compressed bits(Mb)"])
        df["Mask Percentage"] = df["Mask Percentage"].map(lambda x: f"{x:.1f}%")
        df.to_csv('bottom16bit.csv', mode='a', header=False, index=False)

if __name__ == '__main__':
    for model_builder, model_weight in [
        (models.alexnet, models.AlexNet_Weights.IMAGENET1K_V1),
        (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
        (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        (models.googlenet, models.GoogLeNet_Weights.IMAGENET1K_V1),
        (models.inception_v3, models.Inception_V3_Weights.IMAGENET1K_V1),
        (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1),
        (models.regnet_x_16gf, models.RegNet_X_16GF_Weights.IMAGENET1K_V1),
        (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
        (models.shufflenet_v2_x0_5, models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1),
        (models.squeezenet1_0, models.SqueezeNet1_0_Weights.IMAGENET1K_V1),
        (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
        (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1)]:
        output(model_builder, model_weight)