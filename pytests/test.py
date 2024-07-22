import torch

# 示例数据
data = torch.tensor([[10, 20, 20],
                     [50, 40, 40]])

# 索引张量
index = torch.tensor([[0, 1, 1],
                      [1, 0, 0]])

# 创建一个全零的目标张量，形状与 data 相同
output = torch.tensor([[10, 20, 30],
                     [40, 50, 60]])

# 使用 scatter 将数据散布到目标张量
result = torch.scatter(output, dim=1, index=index, src=data)

print("原始数据：")
print(data)
print("索引张量：")
print(index)
print("scatter 结果：")
print(result)
