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


def compress_float(num, mantissa_bits=6, exponent_bits=0):
    """
    Compress a float32 number by reducing the mantissa and exponent bits,
    with support for the sign bit.

    Args:
        num (float): The float number to compress.
        mantissa_bits (int): Number of bits to retain in the mantissa.
        exponent_bits (int): Number of bits to retain in the exponent.

    Returns:
        (int, int, int): Compressed sign, mantissa, and exponent.
    """
    # Convert the float number to raw binary representation
    raw = int.from_bytes(struct.pack('>f', num), 'big')

    # Extract the sign bit
    sign_bit = (raw >> 31) & 0x1

    # Extract original exponent and mantissa
    original_mantissa = (raw & 0x007FFFFF) | 0x00800000  # Include the implicit leading 1
    original_exponent = (raw >> 23) & 0xFF

    if exponent_bits == 0:
        # Fix the exponent to 127 (bias for IEEE 754)
        compressed_exponent = 127
    else:
        # Bias correction for the exponent (from 8-bit bias to reduced bias)
        original_exponent_unbiased = original_exponent - 127
        reduced_exponent_bias = (2 ** (exponent_bits - 1)) - 1
        compressed_exponent = max(-reduced_exponent_bias,
                                  min(reduced_exponent_bias, original_exponent_unbiased))
        compressed_exponent += reduced_exponent_bias

    # Handle subnormal case
    if exponent_bits == 0 or compressed_exponent <= 0:
        compressed_exponent = 0
        compressed_mantissa = 0
    else:
        # Reduce mantissa precision
        compressed_mantissa = original_mantissa >> (23 - mantissa_bits)

    return sign_bit, compressed_mantissa, compressed_exponent


def decompress_float(sign_bit, compressed_mantissa, compressed_exponent, mantissa_bits=6, exponent_bits=0):
    """
    Decompress a compressed float32 number.

    Args:
        sign_bit (int): Compressed sign bit.
        compressed_mantissa (int): Compressed mantissa.
        compressed_exponent (int): Compressed exponent.
        mantissa_bits (int): Number of bits retained in the mantissa.
        exponent_bits (int): Number of bits retained in the exponent.

    Returns:
        float: Decompressed float number.
    """
    if exponent_bits == 0:
        # Fix the exponent to 127 (bias for IEEE 754)
        original_exponent = 127
    else:
        # Reconstruct the exponent with bias correction
        reduced_exponent_bias = (2 ** (exponent_bits - 1)) - 1
        original_exponent = compressed_exponent - reduced_exponent_bias + 127

    # Expand mantissa back to 23 bits
    expanded_mantissa = compressed_mantissa << (23 - mantissa_bits)
    expanded_mantissa &= 0x007FFFFF  # Ensure it fits within 23 bits

    # Combine the sign, exponent, and mantissa to reconstruct the float
    raw = (sign_bit << 31) | (original_exponent << 23) | expanded_mantissa
    return struct.unpack('>f', raw.to_bytes(4, 'big'))[0]


def compress_tensor(tensor, mantissa_bits=3, exponent_bits=4):
    """
    Compress all values in a tensor.

    Args:
        tensor (torch.Tensor): Tensor of float32 values to compress.
        mantissa_bits (int): Number of bits to retain in the mantissa.
        exponent_bits (int): Number of bits to retain in the exponent.

    Returns:
        torch.Tensor: Tensor containing the compressed representation of the input tensor.
    """
    if tensor.dtype == torch.float32:

        sign_bits = []
        mantissas = []
        exponents = []

        # Loop through each element in the tensor
        for num in tensor.view(-1):  # Flatten the tensor for iteration
            sign_bit, compressed_mantissa, compressed_exponent = compress_float(num.item(), mantissa_bits, exponent_bits)
            sign_bits.append(sign_bit)
            mantissas.append(compressed_mantissa)
            exponents.append(compressed_exponent)

        # Convert the lists into tensors
        sign_bits_tensor = torch.tensor(sign_bits, dtype=torch.int32)
        mantissas_tensor = torch.tensor(mantissas, dtype=torch.int32)
        exponents_tensor = torch.tensor(exponents, dtype=torch.int32)

        return sign_bits_tensor, mantissas_tensor, exponents_tensor


def decompress_tensor(sign_bits_tensor, mantissas_tensor, exponents_tensor, mantissa_bits=6, exponent_bits=1):
    """
    Decompress a tensor of compressed values.

    Args:
        sign_bits_tensor (torch.Tensor): Tensor containing sign bits.
        mantissas_tensor (torch.Tensor): Tensor containing mantissas.
        exponents_tensor (torch.Tensor): Tensor containing exponents.
        mantissa_bits (int): Number of bits retained in the mantissa.
        exponent_bits (int): Number of bits retained in the exponent.

    Returns:
        torch.Tensor: Tensor containing decompressed float32 values.
    """
    decompressed_values = []

    for sign_bit, mantissa, exponent in zip(sign_bits_tensor.view(-1), mantissas_tensor.view(-1), exponents_tensor.view(-1)):
        decompressed_value = decompress_float(sign_bit.item(), mantissa.item(), exponent.item(), mantissa_bits, exponent_bits)
        decompressed_values.append(decompressed_value)

    return torch.tensor(decompressed_values, dtype=torch.float32)


def compress(tensor, mantissa_bits, exponent_bits):
    sign_bits_tensor, mantissas_tensor, exponents_tensor = compress_tensor(tensor, mantissa_bits, exponent_bits)
    decompressed_tensor = decompress_tensor(sign_bits_tensor, mantissas_tensor, exponents_tensor, mantissa_bits, exponent_bits)
    return decompressed_tensor.reshape_as(tensor)


def calculate_accuracy(model, dataloader):
    correct_top1 = 0
    correct_top5 = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            outputs = model(images)
            # Top-1 Accuracy
            _, predicted_classes = torch.max(outputs, 1)
            correct_top1 += (predicted_classes == labels).sum().item()

            # Top-5 Accuracy
            _, top5_classes = torch.topk(outputs, 5, dim=1, largest=True, sorted=True)
            top5_classes = top5_classes.numpy()
            labels = labels.numpy()
            correct_top5 += sum([label in top5_classes[i] for i, label in enumerate(labels)])

            total_images += labels.size

    top1_accuracy = correct_top1 / total_images
    top5_accuracy = correct_top5 / total_images

    return top1_accuracy, top5_accuracy


def evaluate_model(model, transform, dataset_path='ILSVRC2012_img_val', batch_size=64, num_workers=4):
    model.eval()

    val_dataset = datasets.ImageFolder(dataset_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return calculate_accuracy(model, val_loader)


# Example usage:
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def topkqi_mask(statedict_kqi: dict, percentage: float):
    kqis = torch.cat(tuple(kqi.flatten() for kqi in statedict_kqi.values()), 0)
    num_masked = int(percentage * len(kqis))
    topk = np.partition(kqis, -num_masked)[-num_masked] if abs(num_masked) > 1e-6 else max(kqis)
    cnt = num_masked - sum(kqis > topk)
    statedict_mask = {}
    for key, value in statedict_kqi.items():
        res = np.ones(value.shape, dtype=bool)
        res[value > topk] = False
        if cnt > 0:
            idx = np.where(value.flatten() == topk)[0]
            if idx.shape[0] > cnt:
                idx = np.random.choice(idx, int(cnt), replace=False)
            res = res.flatten()
            res[idx] = False
            res = res.reshape(value.shape)
            cnt -= idx.shape[0]
        statedict_mask[key] = res
    return statedict_mask


def bottomkqi_mask(statedict_kqi: dict, percentage: float):
    kqis = torch.cat(tuple(kqi.flatten() for kqi in statedict_kqi.values()), 0)
    num_masked = int(percentage * len(kqis))
    num_remain = len(kqis) - num_masked
    topk = np.partition(kqis, -num_remain)[-num_remain] if abs(num_remain) > 1e-6 else max(kqis)
    cnt = num_masked - sum(kqis < topk)
    statedict_mask = {}
    for key, value in statedict_kqi.items():
        res = np.ones(value.shape, dtype=bool)
        res[value < topk] = False
        if cnt > 0:
            idx = np.where(value.flatten() == topk)[0]
            if idx.shape[0] > cnt:
                idx = np.random.choice(idx, int(cnt), replace=False)
            res = res.flatten()
            res[idx] = False
            res = res.reshape(value.shape)
            cnt -= idx.shape[0]
        statedict_mask[key] = res
    return statedict_mask


def mask_alexnet(per, top, mantissa_bits=3, exponent_bits=4):
    model = models.alexnet()
    x = torch.randn(1, 3, 224, 224)
    model_params = {var: name for name, var in model.named_parameters()}
    statedict_kqi = {}
    for grad_fn, kqis in torchKQI.KQI_generator(model, x):
        if 'AccumulateGrad' in grad_fn.name():
            assert len(kqis) == 1
            statedict_kqi[model_params[grad_fn.variable]] = kqis[0][0]
    if top:
        statedict_mask = topkqi_mask(statedict_kqi, per)
    else:
        statedict_mask = bottomkqi_mask(statedict_kqi, per)
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    state_dict = {}
    compressed_bits = 0
    original_bits = 0
    for key, value in model.state_dict().items():
        if key in statedict_mask:
            decompressed_tensor = compress(value, mantissa_bits, exponent_bits)
            state_dict[key] = torch.where(torch.from_numpy(statedict_mask[key]), value, decompressed_tensor.reshape(value.shape))
            num_true = np.broadcast_to(statedict_mask[key], value.shape).sum()
            num_false = value.numel() - num_true
            compressed_bits += num_true * 32 + num_false * (mantissa_bits + exponent_bits + 1)
            original_bits += value.numel() * 32
        else:
            state_dict[key] = value

    kqi_mask_dict = {key: value * (1 - statedict_mask[key]) for key, value in statedict_kqi.items()}
    kqi_mask = sum(torch.sum(v).item() for v in kqi_mask_dict.values())
    model.load_state_dict(state_dict=state_dict)
    top1_accuracy, top5_accuracy = evaluate_model(model, transform)

    return model.__class__.__name__, top1_accuracy, top5_accuracy, kqi_mask, original_bits / (10 ** 6), compressed_bits / (10 ** 6)


def mask_densenet(per, top, mantissa_bits=3, exponent_bits=4):
    model = models.densenet121()
    x = torch.randn(1, 3, 224, 224)
    model_params = {var: name for name, var in model.named_parameters()}
    statedict_kqi = {}
    for grad_fn, kqis in torchKQI.KQI_generator(model, x):
        if 'AccumulateGrad' in grad_fn.name():
            assert len(kqis) == 1
            statedict_kqi[model_params[grad_fn.variable]] = kqis[0][0]
    if top:
        statedict_mask = topkqi_mask(statedict_kqi, per)
    else:
        statedict_mask = bottomkqi_mask(statedict_kqi, per)
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    state_dict = {}
    compressed_bits = 0
    original_bits = 0
    for key, value in model.state_dict().items():
        if key in statedict_mask:
            decompressed_tensor = compress(value, mantissa_bits, exponent_bits)
            state_dict[key] = torch.where(torch.from_numpy(statedict_mask[key]), value, decompressed_tensor.reshape(value.shape))
            num_true = np.broadcast_to(statedict_mask[key], value.shape).sum()
            num_false = value.numel() - num_true
            compressed_bits += num_true * 32 + num_false * (mantissa_bits + exponent_bits + 1)
            original_bits += value.numel() * 32
        else:
            state_dict[key] = value

    kqi_mask_dict = {key: value * (1 - statedict_mask[key]) for key, value in statedict_kqi.items()}
    kqi_mask = sum(torch.sum(v).item() for v in kqi_mask_dict.values())
    model.load_state_dict(state_dict=state_dict)
    top1_accuracy, top5_accuracy = evaluate_model(model, transform)

    return model.__class__.__name__, top1_accuracy, top5_accuracy, kqi_mask, original_bits / (10 ** 6), compressed_bits / (10 ** 6)


def mask_efficientnet(per, top, mantissa_bits=3, exponent_bits=4):
    model = models.efficientnet_b0()
    x = torch.randn(1, 3, 224, 224)
    model_params = {var: name for name, var in model.named_parameters()}
    statedict_kqi = {}
    for grad_fn, kqis in torchKQI.KQI_generator(model, x):
        if 'AccumulateGrad' in grad_fn.name():
            assert len(kqis) == 1
            statedict_kqi[model_params[grad_fn.variable]] = kqis[0][0]
    if top:
        statedict_mask = topkqi_mask(statedict_kqi, per)
    else:
        statedict_mask = bottomkqi_mask(statedict_kqi, per)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    state_dict = {}
    compressed_bits = 0
    original_bits = 0
    for key, value in model.state_dict().items():
        if key in statedict_mask:
            decompressed_tensor = compress(value, mantissa_bits, exponent_bits)
            state_dict[key] = torch.where(torch.from_numpy(statedict_mask[key]), value, decompressed_tensor.reshape(value.shape))
            num_true = np.broadcast_to(statedict_mask[key], value.shape).sum()
            num_false = value.numel() - num_true
            compressed_bits += num_true * 32 + num_false * (mantissa_bits + exponent_bits + 1)
            original_bits += value.numel() * 32
        else:
            state_dict[key] = value

    kqi_mask_dict = {key: value * (1 - statedict_mask[key]) for key, value in statedict_kqi.items()}
    kqi_mask = sum(torch.sum(v).item() for v in kqi_mask_dict.values())
    model.load_state_dict(state_dict=state_dict)
    top1_accuracy, top5_accuracy = evaluate_model(model, transform)

    return model.__class__.__name__, top1_accuracy, top5_accuracy, kqi_mask, original_bits / (10 ** 6), compressed_bits / (10 ** 6)


def mask_googlenet(per, top, mantissa_bits=3, exponent_bits=4):
    model = models.googlenet()
    x = torch.randn(1, 3, 224, 224)
    model_params = {var: name for name, var in model.named_parameters()}
    statedict_kqi = {}
    for grad_fn, kqis in torchKQI.KQI_generator(model, x):
        if 'AccumulateGrad' in grad_fn.name():
            assert len(kqis) == 1
            statedict_kqi[model_params[grad_fn.variable]] = kqis[0][0]
    if top:
        statedict_mask = topkqi_mask(statedict_kqi, per)
    else:
        statedict_mask = bottomkqi_mask(statedict_kqi, per)
    model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
    state_dict = {}
    compressed_bits = 0
    original_bits = 0
    for key, value in model.state_dict().items():
        if key in statedict_mask:
            decompressed_tensor = compress(value, mantissa_bits, exponent_bits)
            state_dict[key] = torch.where(torch.from_numpy(statedict_mask[key]), value, decompressed_tensor.reshape(value.shape))
            num_true = np.broadcast_to(statedict_mask[key], value.shape).sum()
            num_false = value.numel() - num_true
            compressed_bits += num_true * 32 + num_false * (mantissa_bits + exponent_bits + 1)
            original_bits += value.numel() * 32
        else:
            state_dict[key] = value

    kqi_mask_dict = {key: value * (1 - statedict_mask[key]) for key, value in statedict_kqi.items()}
    kqi_mask = sum(torch.sum(v).item() for v in kqi_mask_dict.values())
    model.load_state_dict(state_dict=state_dict)
    top1_accuracy, top5_accuracy = evaluate_model(model, transform)

    return model.__class__.__name__, top1_accuracy, top5_accuracy, kqi_mask, original_bits / (10 ** 6), compressed_bits / (10 ** 6)


def mask_inception(per, top, mantissa_bits=3, exponent_bits=4):
    model = models.inception_v3()
    x = torch.randn(1, 3, 224, 224)
    model_params = {var: name for name, var in model.named_parameters()}
    statedict_kqi = {}
    for grad_fn, kqis in torchKQI.KQI_generator(model, x):
        if 'AccumulateGrad' in grad_fn.name():
            assert len(kqis) == 1
            statedict_kqi[model_params[grad_fn.variable]] = kqis[0][0]
    if top:
        statedict_mask = topkqi_mask(statedict_kqi, per)
    else:
        statedict_mask = bottomkqi_mask(statedict_kqi, per)
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    state_dict = {}
    compressed_bits = 0
    original_bits = 0
    for key, value in model.state_dict().items():
        if key in statedict_mask:
            decompressed_tensor = compress(value, mantissa_bits, exponent_bits)
            state_dict[key] = torch.where(torch.from_numpy(statedict_mask[key]), value, decompressed_tensor.reshape(value.shape))
            num_true = np.broadcast_to(statedict_mask[key], value.shape).sum()
            num_false = value.numel() - num_true
            compressed_bits += num_true * 32 + num_false * (mantissa_bits + exponent_bits + 1)
            original_bits += value.numel() * 32
        else:
            state_dict[key] = value

    kqi_mask_dict = {key: value * (1 - statedict_mask[key]) for key, value in statedict_kqi.items()}
    kqi_mask = sum(torch.sum(v).item() for v in kqi_mask_dict.values())
    model.load_state_dict(state_dict=state_dict)
    top1_accuracy, top5_accuracy = evaluate_model(model, transform)

    return model.__class__.__name__, top1_accuracy, top5_accuracy, kqi_mask, original_bits / (10 ** 6), compressed_bits / (10 ** 6)


def mask_mobilenet(per, top, mantissa_bits=3, exponent_bits=4):
    model = models.mobilenet_v2()
    x = torch.randn(1, 3, 224, 224)
    model_params = {var: name for name, var in model.named_parameters()}
    statedict_kqi = {}
    for grad_fn, kqis in torchKQI.KQI_generator(model, x):
        if 'AccumulateGrad' in grad_fn.name():
            assert len(kqis) == 1
            statedict_kqi[model_params[grad_fn.variable]] = kqis[0][0]
    if top:
        statedict_mask = topkqi_mask(statedict_kqi, per)
    else:
        statedict_mask = bottomkqi_mask(statedict_kqi, per)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    state_dict = {}
    compressed_bits = 0
    original_bits = 0
    for key, value in model.state_dict().items():
        if key in statedict_mask:
            decompressed_tensor = compress(value, mantissa_bits, exponent_bits)
            state_dict[key] = torch.where(torch.from_numpy(statedict_mask[key]), value, decompressed_tensor.reshape(value.shape))
            num_true = np.broadcast_to(statedict_mask[key], value.shape).sum()
            num_false = value.numel() - num_true
            compressed_bits += num_true * 32 + num_false * (mantissa_bits + exponent_bits + 1)
            original_bits += value.numel() * 32
        else:
            state_dict[key] = value

    kqi_mask_dict = {key: value * (1 - statedict_mask[key]) for key, value in statedict_kqi.items()}
    kqi_mask = sum(torch.sum(v).item() for v in kqi_mask_dict.values())
    model.load_state_dict(state_dict=state_dict)
    top1_accuracy, top5_accuracy = evaluate_model(model, transform)

    return model.__class__.__name__, top1_accuracy, top5_accuracy, kqi_mask, original_bits / (10 ** 6), compressed_bits / (10 ** 6)


def mask_regnet(per, top, mantissa_bits=3, exponent_bits=4):
    model = models.regnet_x_16gf()
    x = torch.randn(1, 3, 224, 224)
    model_params = {var: name for name, var in model.named_parameters()}
    statedict_kqi = {}
    for grad_fn, kqis in torchKQI.KQI_generator(model, x):
        if 'AccumulateGrad' in grad_fn.name():
            assert len(kqis) == 1
            statedict_kqi[model_params[grad_fn.variable]] = kqis[0][0]
    if top:
        statedict_mask = topkqi_mask(statedict_kqi, per)
    else:
        statedict_mask = bottomkqi_mask(statedict_kqi, per)
    model = models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.IMAGENET1K_V1)
    state_dict = {}
    compressed_bits = 0
    original_bits = 0
    for key, value in model.state_dict().items():
        if key in statedict_mask:
            decompressed_tensor = compress(value, mantissa_bits, exponent_bits)
            state_dict[key] = torch.where(torch.from_numpy(statedict_mask[key]), value, decompressed_tensor.reshape(value.shape))
            num_true = np.broadcast_to(statedict_mask[key], value.shape).sum()
            num_false = value.numel() - num_true
            compressed_bits += num_true * 32 + num_false * (mantissa_bits + exponent_bits + 1)
            original_bits += value.numel() * 32
        else:
            state_dict[key] = value

    kqi_mask_dict = {key: value * (1 - statedict_mask[key]) for key, value in statedict_kqi.items()}
    kqi_mask = sum(torch.sum(v).item() for v in kqi_mask_dict.values())
    model.load_state_dict(state_dict=state_dict)
    top1_accuracy, top5_accuracy = evaluate_model(model, transform)

    return model.__class__.__name__, top1_accuracy, top5_accuracy, kqi_mask, original_bits / (10 ** 6), compressed_bits / (10 ** 6)


def mask_resnet(per, top, mantissa_bits=3, exponent_bits=4):
    model = models.resnet50()
    x = torch.randn(1, 3, 224, 224)
    model_params = {var: name for name, var in model.named_parameters()}
    statedict_kqi = {}
    for grad_fn, kqis in torchKQI.KQI_generator(model, x):
        if 'AccumulateGrad' in grad_fn.name():
            assert len(kqis) == 1
            statedict_kqi[model_params[grad_fn.variable]] = kqis[0][0]
    if top:
        statedict_mask = topkqi_mask(statedict_kqi, per)
    else:
        statedict_mask = bottomkqi_mask(statedict_kqi, per)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    state_dict = {}
    compressed_bits = 0
    original_bits = 0
    for key, value in model.state_dict().items():
        if key in statedict_mask:
            decompressed_tensor = compress(value, mantissa_bits, exponent_bits)
            state_dict[key] = torch.where(torch.from_numpy(statedict_mask[key]), value, decompressed_tensor.reshape(value.shape))
            num_true = np.broadcast_to(statedict_mask[key], value.shape).sum()
            num_false = value.numel() - num_true
            compressed_bits += num_true * 32 + num_false * (mantissa_bits + exponent_bits + 1)
            original_bits += value.numel() * 32
        else:
            state_dict[key] = value

    kqi_mask_dict = {key: value * (1 - statedict_mask[key]) for key, value in statedict_kqi.items()}
    kqi_mask = sum(torch.sum(v).item() for v in kqi_mask_dict.values())
    model.load_state_dict(state_dict=state_dict)
    top1_accuracy, top5_accuracy = evaluate_model(model, transform)

    return model.__class__.__name__, top1_accuracy, top5_accuracy, kqi_mask, original_bits / (10 ** 6), compressed_bits / (10 ** 6)


def mask_shufflenet(per, top, mantissa_bits=3, exponent_bits=4):
    model = models.shufflenet_v2_x0_5()
    x = torch.randn(1, 3, 224, 224)
    model_params = {var: name for name, var in model.named_parameters()}
    statedict_kqi = {}
    for grad_fn, kqis in torchKQI.KQI_generator(model, x):
        if 'AccumulateGrad' in grad_fn.name():
            assert len(kqis) == 1
            statedict_kqi[model_params[grad_fn.variable]] = kqis[0][0]
    if top:
        statedict_mask = topkqi_mask(statedict_kqi, per)
    else:
        statedict_mask = bottomkqi_mask(statedict_kqi, per)
    model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
    state_dict = {}
    compressed_bits = 0
    original_bits = 0
    for key, value in model.state_dict().items():
        if key in statedict_mask:
            decompressed_tensor = compress(value, mantissa_bits, exponent_bits)
            state_dict[key] = torch.where(torch.from_numpy(statedict_mask[key]), value, decompressed_tensor.reshape(value.shape))
            num_true = np.broadcast_to(statedict_mask[key], value.shape).sum()
            num_false = value.numel() - num_true
            compressed_bits += num_true * 32 + num_false * (mantissa_bits + exponent_bits + 1)
            original_bits += value.numel() * 32
        else:
            state_dict[key] = value

    kqi_mask_dict = {key: value * (1 - statedict_mask[key]) for key, value in statedict_kqi.items()}
    kqi_mask = sum(torch.sum(v).item() for v in kqi_mask_dict.values())
    model.load_state_dict(state_dict=state_dict)
    top1_accuracy, top5_accuracy = evaluate_model(model, transform)

    return model.__class__.__name__, top1_accuracy, top5_accuracy, kqi_mask, original_bits / (10 ** 6), compressed_bits / (10 ** 6)


def mask_squeezenet(per, top, mantissa_bits=3, exponent_bits=4):
    model = models.squeezenet1_0()
    x = torch.randn(1, 3, 224, 224)
    model_params = {var: name for name, var in model.named_parameters()}
    statedict_kqi = {}
    for grad_fn, kqis in torchKQI.KQI_generator(model, x):
        if 'AccumulateGrad' in grad_fn.name():
            assert len(kqis) == 1
            statedict_kqi[model_params[grad_fn.variable]] = kqis[0][0]
    if top:
        statedict_mask = topkqi_mask(statedict_kqi, per)
    else:
        statedict_mask = bottomkqi_mask(statedict_kqi, per)
    model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    state_dict = {}
    compressed_bits = 0
    original_bits = 0
    for key, value in model.state_dict().items():
        if key in statedict_mask:
            decompressed_tensor = compress(value, mantissa_bits, exponent_bits)
            state_dict[key] = torch.where(torch.from_numpy(statedict_mask[key]), value, decompressed_tensor.reshape(value.shape))
            num_true = np.broadcast_to(statedict_mask[key], value.shape).sum()
            num_false = value.numel() - num_true
            compressed_bits += num_true * 32 + num_false * (mantissa_bits + exponent_bits + 1)
            original_bits += value.numel() * 32
        else:
            state_dict[key] = value

    kqi_mask_dict = {key: value * (1 - statedict_mask[key]) for key, value in statedict_kqi.items()}
    kqi_mask = sum(torch.sum(v).item() for v in kqi_mask_dict.values())
    model.load_state_dict(state_dict=state_dict)
    top1_accuracy, top5_accuracy = evaluate_model(model, transform)

    return model.__class__.__name__, top1_accuracy, top5_accuracy, kqi_mask, original_bits / (10 ** 6), compressed_bits / (10 ** 6)


def mask_vgg(per, top, mantissa_bits=3, exponent_bits=4):
    model = models.vgg16()
    x = torch.randn(1, 3, 224, 224)
    model_params = {var: name for name, var in model.named_parameters()}
    statedict_kqi = {}
    for grad_fn, kqis in torchKQI.KQI_generator(model, x):
        if 'AccumulateGrad' in grad_fn.name():
            assert len(kqis) == 1
            statedict_kqi[model_params[grad_fn.variable]] = kqis[0][0]
    if top:
        statedict_mask = topkqi_mask(statedict_kqi, per)
    else:
        statedict_mask = bottomkqi_mask(statedict_kqi, per)
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    state_dict = {}
    compressed_bits = 0
    original_bits = 0
    for key, value in model.state_dict().items():
        if key in statedict_mask:
            decompressed_tensor = compress(value, mantissa_bits, exponent_bits)
            state_dict[key] = torch.where(torch.from_numpy(statedict_mask[key]), value, decompressed_tensor.reshape(value.shape))
            num_true = np.broadcast_to(statedict_mask[key], value.shape).sum()
            num_false = value.numel() - num_true
            compressed_bits += num_true * 32 + num_false * (mantissa_bits + exponent_bits + 1)
            original_bits += value.numel() * 32
        else:
            state_dict[key] = value

    kqi_mask_dict = {key: value * (1 - statedict_mask[key]) for key, value in statedict_kqi.items()}
    kqi_mask = sum(torch.sum(v).item() for v in kqi_mask_dict.values())
    model.load_state_dict(state_dict=state_dict)
    top1_accuracy, top5_accuracy = evaluate_model(model, transform)

    return model.__class__.__name__, top1_accuracy, top5_accuracy, kqi_mask, original_bits / (10 ** 6), compressed_bits / (10 ** 6)


def mask_vit(per, top, mantissa_bits=3, exponent_bits=4):
    model = models.vit_b_16()
    x = torch.randn(1, 3, 224, 224)
    model_params = {var: name for name, var in model.named_parameters()}
    statedict_kqi = {}
    for grad_fn, kqis in torchKQI.KQI_generator(model, x):
        if 'AccumulateGrad' in grad_fn.name():
            assert len(kqis) == 1
            statedict_kqi[model_params[grad_fn.variable]] = kqis[0][0]

    if top:
        statedict_mask = topkqi_mask(statedict_kqi, per)
    else:
        statedict_mask = bottomkqi_mask(statedict_kqi, per)
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

    state_dict = {}
    compressed_bits = 0
    original_bits = 0
    for key, value in model.state_dict().items():
        if key in statedict_mask:
            decompressed_tensor = compress(value, mantissa_bits, exponent_bits)
            state_dict[key] = torch.where(torch.from_numpy(statedict_mask[key]), value, decompressed_tensor.reshape(value.shape))
            num_true = np.broadcast_to(statedict_mask[key], value.shape).sum()
            num_false = value.numel() - num_true
            compressed_bits += num_true * 32 + num_false * (mantissa_bits + exponent_bits + 1)
            original_bits += value.numel() * 32
        else:
            state_dict[key] = value

    kqi_mask_dict = {key: value * (1 - statedict_mask[key]) for key, value in statedict_kqi.items()}
    kqi_mask = sum(torch.sum(v).item() for v in kqi_mask_dict.values())
    model.load_state_dict(state_dict=state_dict)
    top1_accuracy, top5_accuracy = evaluate_model(model, transform)

    return model.__class__.__name__, top1_accuracy, top5_accuracy, kqi_mask, original_bits / (10 ** 6), compressed_bits / (10 ** 6)


def output(mask):
    for per in np.arange(0, 1.1, 0.1):
        name, top1_accuracy, top5_accuracy, kqi_mask, original_bits, compressed_bits = mask(per, True)
        result = (name, per * 100, top1_accuracy, top5_accuracy, kqi_mask, original_bits, compressed_bits)
        df = pd.DataFrame([result], columns=["Name", "Mask Percentage", "Top-1 Accuracy", "Top-5 Accuracy", "Mask KQI", "original bits(Mb)", "compressed bits(Mb)"])
        df["Mask Percentage"] = df["Mask Percentage"].map(lambda x: f"{x:.1f}%")
        df.to_csv('top8bit.csv', mode='a', header=False, index=False)

    for per in np.arange(0, 1.1, 0.1):
        name, top1_accuracy, top5_accuracy, kqi_mask, original_bits, compressed_bits = mask(per, False)
        result = (name, per * 100, top1_accuracy, top5_accuracy, kqi_mask, original_bits, compressed_bits)
        df = pd.DataFrame([result], columns=["Name", "Mask Percentage", "Top-1 Accuracy", "Top-5 Accuracy", "Mask KQI", "original bits(Mb)", "compressed bits(Mb)"])
        df["Mask Percentage"] = df["Mask Percentage"].map(lambda x: f"{x:.1f}%")
        df.to_csv('bottom8bit.csv', mode='a', header=False, index=False)


if __name__ == '__main__':
    if not os.path.exists('top8bit.csv'):
        pd.DataFrame(columns=['Name', 'Mask Percentage', 'Top-1 Accuracy', 'Top-5 Accuracy', 'Mask KQI', 'original bits(Mb)', 'compressed bits(Mb)']).to_csv('top8bit.csv', index=False)
    if not os.path.exists('bottom8bit.csv'):
        pd.DataFrame(columns=['Name', 'Mask Percentage', 'Top-1 Accuracy', 'Top-5 Accuracy', 'Mask KQI', 'original bits(Mb)', 'compressed bits(Mb)']).to_csv('bottom8bit.csv', index=False)

    for mask in [mask_alexnet, mask_densenet, mask_efficientnet, mask_googlenet, mask_inception, mask_mobilenet, mask_regnet, mask_shufflenet, mask_squeezenet, mask_resnet, mask_vgg, mask_vit]:
        output(mask)
