import torch.nn as nn
import numpy as np 
import toGraph
import util.kqi as kqi
import datetime

def default_layer(G, index, Feature, layer):
    print("Found other layer:", layer)
    return index, Feature

def process_model(G, model, index, in_Feature):
    layer_map = {
        nn.Linear: toGraph.Linear_K,
        nn.Conv2d: toGraph.Conv2d_K,
        nn.MaxPool2d: toGraph.Pooling_K,
        nn.ReLU: toGraph.ReLU_K,
        nn.Flatten: toGraph.Flatten_K,
        nn.Softmax: toGraph.Softmax_K,
        nn.RNN: toGraph.RNN_K,
        nn.LSTM: toGraph.LSTM_K,
    }

    Feature = in_Feature
    for name, module in model.named_modules():
    #for name, module in model.named_children():
        print(name, module)
        layer_type = type(module)
        layer_func = layer_map.get(layer_type, default_layer)
        index, Feature = layer_func(G, index, Feature, module)
        print(index)
    return index


def Kqi(model, G, index, in_Feature):
   
    index = process_model(G, model, index, in_Feature)

    # De-cycling to form DAG
    G.remove_cycles()
    # Set the current date
    G.set_today(datetime.date.today())
    # Set the attenuation coefficient (1 that is, no attenuation, 0 that is, the maximum attenuation rate)
    G.set_decay(1)
    # Calculate KQI
    k = 0
    for i in range(1, index):
        k += G.kqi(i)
    return k

if __name__ == '__main__':
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.model = nn.Sequential(
                #The size of the picture is 28x28#The size of the picture is 28x28
                nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=3, stride = 1, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride = 2),

                nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, stride = 1, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride = 2),

                nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=3, stride = 1, padding = 1),
                nn.ReLU(),

                nn.Flatten(),
                nn.Linear(in_features = 128 * 7 * 7, out_features = 128),
                nn.ReLU(),
                nn.Linear(in_features = 128, out_features = 10),
                nn.Softmax(dim=1)
            )
    model = CNN()
    index = 1
    G = kqi.DiGraph()
    index, in_Feature = toGraph.Feature(G, index, size=28, channels=1)

    k = Kqi(model, G, index, in_Feature)
    print(k)