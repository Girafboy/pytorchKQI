import os
import torch
import torchKQI
from transformers import AutoConfig, AutoModel

def get_all_model_names(base_dir):
    model_names = []
    for root, dirs, files in os.walk(base_dir):
        if root.count(os.sep) == base_dir.count(os.sep) + 1:
            for dir_name in dirs:
                model_names.append(os.path.join(root, dir_name))
    return model_names

model_names = get_all_model_names("llms")

for model_name in model_names:
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_config(config)

    batch_size = 1
    x = torch.randint(0, config.vocab_size, (batch_size, config.max_position_embeddings))
    y = model(x)
    print(type(y))
    try:
        kqi = torchKQI.KQI(model, x).item()
        print(kqi)
    except Exception as e:
        print(e)