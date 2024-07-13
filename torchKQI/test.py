import torch

# 创建一个 PyTorch 张量
arr = torch.tensor([[1, 2, 3],
                    [4, 5, 6]])

# 在第一个维度（行）上求和
sum1 = torch.sum(arr, dim=1, keepdim=True)  # 沿着第一个维度（行）求和，并保持维度
print("Sum along axis 1:\n", sum1)

# 使用 expand_as 方法扩展形状
sum2 = sum1.expand_as(arr)
print("Expanded sum tensor:\n", sum2)
