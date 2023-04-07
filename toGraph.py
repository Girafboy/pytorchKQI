import numpy as np 
import datetime
import util.kqi as kqi

#线性层
def linear(DiGraph, index, in_Feature, out_features):
    out_Feature = np.arange(index, index + out_features)
    index += out_features
    for v in out_Feature:
        DiGraph.add_node(v, in_Feature.tolist(), datetime.date.today())
    return index, out_Feature
#线性层(1d->1d)
def Linear_K(DiGraph, index, in_Feature, layer):
    return linear(DiGraph, index, in_Feature, layer.out_features)

#添加无入度的节点
def add_root(DiGraph, index, num):
    for i in range(index, index+num):
        DiGraph.add_node(i, [], datetime.date.today())

#生成特征矩阵(3d)
def Feature(DiGraph, index, size, channels):
    features = np.arange(index, index + channels*size*size).reshape([channels, size, size])
    add_root(DiGraph, index, channels*size*size)
    index += channels*size*size
    return index, features

def Padding(DiGraph, index, in_Feature, padding):
    #在行上padding
    for i in range(padding[0]):
        temp_Feature = []
        for channel in in_Feature:
            row_size = np.size(channel, 1)
            channel = np.row_stack((np.arange(index, index+row_size), channel))
            add_root(DiGraph, index, row_size)
            index += row_size
            channel = np.row_stack((channel, np.arange(index, index+row_size)))
            add_root(DiGraph, index, row_size)
            index += row_size
            
            temp_Feature.append(channel)
        in_Feature = np.asarray(temp_Feature)
    #在列上padding
    for i in range(padding[1]):
        temp_Feature = []
        for channel in in_Feature:
            column_size = np.size(channel, 0)
            channel = np.column_stack((np.arange(index, index+column_size), channel))
            add_root(DiGraph, index, column_size)
            index += column_size
            channel = np.column_stack((channel, np.arange(index, index+column_size)))
            add_root(DiGraph, index, column_size)
            index += column_size
            
            temp_Feature.append(channel)
        in_Feature = np.asarray(temp_Feature)
    return index, in_Feature

#卷积核操作
def kernel_op(Feature, kernel_size, x_index, y_index):
    kernel_Feature = Feature[..., y_index:y_index+kernel_size[0], x_index:x_index+kernel_size[1]]
    return kernel_Feature.flatten().tolist()

#卷积层
def Conv2d_K(DiGraph, index, in_Feature, layer):
    #输出特征
    out_Feature = []
    #padding
    index, in_Feature = Padding(DiGraph, index, in_Feature, layer.padding)
    for c in range(layer.out_channels):
        y_Feature = []
        for y in range(0, np.size(in_Feature, 1) - layer.kernel_size[0] + 1, layer.stride[0]):
            x_Feature = []
            for x in range(0, np.size(in_Feature, 2) - layer.kernel_size[1] + 1, layer.stride[1]):
                #获取当前卷积核位置对应的节点
                pred_list = kernel_op(in_Feature, layer.kernel_size, x, y)
                DiGraph.add_node(index, pred_list, datetime.date.today())
                x_Feature.append(index)
                index += 1
            y_Feature.append(x_Feature)
        out_Feature.append(y_Feature)
    return index, np.asarray(out_Feature)

#池化层
def Pooling_K(DiGraph, index, in_Feature, layer):
    #输出特征
    out_Feature = []
    for c in range(np.size(in_Feature, 0)):
        y_Feature = []
        for y in range(0, np.size(in_Feature, 1) - layer.kernel_size + 1, layer.stride):
            x_Feature = []
            for x in range(0, np.size(in_Feature, 2) - layer.kernel_size + 1, layer.stride):
                #复用卷积层的卷积核操作
                pred_list = kernel_op(in_Feature, (layer.kernel_size,layer.kernel_size), x, y)
                DiGraph.add_node(index, pred_list, datetime.date.today())
                x_Feature.append(index)
                index += 1
            y_Feature.append(x_Feature)
        out_Feature.append(y_Feature)
    return index, np.asarray(out_Feature)

#ReLU激活层
def ReLU_K(DiGraph, index, in_Feature, layer):
    shape = in_Feature.shape
    out_Feature = np.zeros(shape, dtype=int)
    for i in np.ndindex(shape):
        out_Feature[i] = index
        index += 1
        DiGraph.add_node(out_Feature[i], [in_Feature[i]], datetime.date.today())

    return index, out_Feature

#激活层
def Activation_K(DiGraph, index, in_Feature, layer):
    shape = in_Feature.shape
    out_Feature = np.zeros(shape, dtype=int)
    for i in np.ndindex(shape):
        out_Feature[i] = index
        index += 1
        DiGraph.add_node(out_Feature[i], [in_Feature[i]], datetime.date.today())

    return index, out_Feature

#softmax层(1d->1d)
def Softmax_K(DiGraph, index, in_Feature, layer):
    size = np.size(in_Feature, 0)
    out_Feature = np.arange(index, index + size)
    index += size
    for v in out_Feature:
        DiGraph.add_node(v, in_Feature.tolist(), datetime.date.today())
    return index, out_Feature

def Flatten_K(DiGraph, index, in_Feature, layer):
    return index, in_Feature.flatten()

def RNN_K(DiGraph, index, in_Feature, layer):
    h_in = np.arange(index, index + layer.hidden_size)
    index += layer.hidden_size
    #h_in无入度
    for v in h_in:
        DiGraph.add_node(v, [], datetime.date.today())

    h_out = np.arange(index, index + layer.hidden_size)
    index += layer.hidden_size

    index, hidden_Feature =  linear(DiGraph, index, np.append(h_in, in_Feature), layer.hidden_size)
    index, out_Feature =  Activation_K(DiGraph, index, hidden_Feature)

    for i in range(layer.hidden_size):
        DiGraph.add_node(h_out[i], [out_Feature[i]], datetime.date.today())

    return index, out_Feature

def LSTM_K(DiGraph, index, in_Feature, layer):
    #参考：http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    x_t = in_Feature
    hidden_size = layer.hidden_size

    h_t0 = np.arange(index, index + x_t.size)
    index += x_t.size
    #h_t0无入度
    for v in h_t0:
        DiGraph.add_node(v, [], datetime.date.today())
        
    C_t0 = np.arange(index, index + hidden_size)
    index += hidden_size
    #C_t0无入度
    for v in C_t0:
        DiGraph.add_node(v, [], datetime.date.today())

    index, f_t = linear(DiGraph, index, np.append(h_t0, x_t), hidden_size)
    index, f_t = Activation_K(DiGraph, index, f_t)

    index, i_t = linear(DiGraph, index, np.append(h_t0, x_t), hidden_size)
    index, i_t = Activation_K(DiGraph, index, i_t)

    index, Ct_hat = linear(DiGraph, index, np.append(h_t0, x_t), hidden_size)
    index, Ct_hat = Activation_K(DiGraph, index, Ct_hat)

    #f_t*C_t0
    temp1 = np.arange(index, index + hidden_size)
    index += hidden_size
    for i in range(hidden_size):
        DiGraph.add_node(temp1[i], [f_t[i], C_t0[i]], datetime.date.today())
    #i_t*Ct_hat
    temp2 = np.arange(index, index + hidden_size)
    index += hidden_size
    for i in range(hidden_size):
        DiGraph.add_node(temp2[i], [i_t[i], Ct_hat[i]], datetime.date.today())
    #C_t = f_t*C_t0 + i_t*Ct_hat
    C_t = np.arange(index, index + hidden_size)
    index += hidden_size
    for i in range(hidden_size):
        DiGraph.add_node(C_t[i], [temp1[i], temp2[i]], datetime.date.today())

    index, o_t = linear(DiGraph, index, np.append(h_t0, x_t), hidden_size)
    index, o_t = Activation_K(DiGraph, index, o_t)

    index, C_t = Activation_K(DiGraph, index, C_t)
    h_t = np.arange(index, index + hidden_size)
    index += hidden_size

    for i in range(hidden_size):
        DiGraph.add_node(h_t[i], [o_t[i], C_t[i]], datetime.date.today())
    
    return index, h_t
