import torch

# 创建两个随机张量
tensor1 = torch.rand(1, 10)
tensor2 = torch.rand(1, 10)

# 创建 CosineSimilarity 对象
cosine_similarity = torch.nn.CosineSimilarity(dim=0)

# 计算余弦相似度
similarity = cosine_similarity(tensor1, tensor2)

# 打印结果
print(similarity)
