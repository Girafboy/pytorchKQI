import torchvision.models as models
import torchKQI
import torch
import numpy as np

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

model = models.alexnet()
x = torch.randn(1, 3, 224, 224)
model_params = {var: name for name, var in model.named_parameters()}
statedict_kqi = {}
for grad_fn, kqis in torchKQI.KQI_generator(model, x):
    if 'AccumulateGrad' in grad_fn.name():
        assert len(kqis) == 1
        statedict_kqi[model_params[grad_fn.variable]] = kqis[0][0]
statedict_mask = topkqi_mask(statedict_kqi, 0.1)

# Print the number of False values in each key of statedict_mask
for key, mask in statedict_mask.items():
    print(f"{key}: {torch.sum(~mask).item()} False values")

