import numpy as np 
import torch.nn as nn

# 计算kqi的方法
def caculate_Kqi(model, x):
    # 虚拟根节点连接的边
    W0 = np.prod(np.shape(x))
    # 计算输出的形状与整个网络的graph_size
    x, W = model.graph_size(x)
    W += W0

    alpha_volumes = np.zeros(np.shape(x))
    alpha_volumes, kqi = model.Kqi(alpha_volumes, W)
    # 虚拟根节点的Kqi
    kqi += root_Kqi(alpha_volumes, W)
    
    return kqi

def root_Kqi(alpha_volumes: list, W: int):
    # 计算顶层（无入度）节点的kqi
    V_root = W
    kqi = 0
    alpha_volumes_1d = np.array(alpha_volumes).flatten()
    for V in alpha_volumes_1d:
        d = 1
        kqi += - V / d / W * np.log2(V / d / V_root)
    return kqi

# Sequential
class Sequential(nn.Sequential):
    def graph_size(self, x, W=0):
        W1 = 0
        shape = np.shape(x)
        shape_list = [shape]

        for module in self:
            # 如果已经实现graph_size()方法
            if hasattr(module, 'graph_size'):
                w_, shape = module.graph_size(shape)
                W1 += w_
                shape_list.append(shape)

        shape_list.pop()
        self.___shape_list = shape_list
        x = self.forward(x)

        return x, W1 + W
        
    def Kqi(self, alpha_volumes, W):
        kqi = 0
        reversed_shape_list = list(reversed(self.___shape_list))
        for module, shape in zip(reversed(self), reversed_shape_list):
            # 如果已经实现Kqi()方法
            if hasattr(module, 'Kqi'):
                # shape: in_features_shape
                k, beta_volumes = module.Kqi(shape, alpha_volumes, W)
                kqi += k
                alpha_volumes = beta_volumes
            else:
                # 如果当前层未实现Kqi()方法，需要将shape添加回列表
                reversed_shape_list.insert(0, shape)

        return alpha_volumes, kqi

# Linear
class Linear(nn.Linear):
    def graph_size(self, shape):
        return self.in_features * self.out_features, self.out_features

    def Kqi(self, in_features_shape, alpha_volumes, W):
        # beta行
        beta_nodes = self.in_features
        # alpha行
        alpha_nodes = self.out_features

        # 一个节点的volume
        V_alpha = alpha_volumes[0]
        V_beta = alpha_nodes + alpha_nodes * (V_alpha / beta_nodes)
        # alpha节点的入度
        d = beta_nodes

        # 计算alpha行的Kqi
        if V_alpha == 0:
            kqi = 0
        else:
            kqi = alpha_nodes * beta_nodes * (- V_alpha / d / W * np.log2(V_alpha / d / V_beta))
        beta_volumes = [V_beta] * beta_nodes

        return kqi, beta_volumes
    

def Padding(in_Feature, padding):
    # 在行上padding
    for i in range(padding[0]):
        temp_Feature = []
        for channel in in_Feature:
            row_size = np.size(channel, 1)
            channel = np.row_stack((np.zeros(row_size), channel))
            channel = np.row_stack((channel, np.zeros(row_size)))

            temp_Feature.append(channel)
        in_Feature = np.asarray(temp_Feature)
    # 在列上padding
    for i in range(padding[1]):
        temp_Feature = []
        for channel in in_Feature:
            column_size = np.size(channel, 0)
            channel = np.column_stack((np.zeros(column_size), channel))
            channel = np.column_stack((channel, np.zeros(column_size)))
            
            temp_Feature.append(channel)
        in_Feature = np.asarray(temp_Feature)
    return in_Feature

def Convolution(in_Feature, out_channels, kernel_size, stride, padding=0):
    # 卷积核操作
    def kernel_op(Feature, kernel_size, x_index, y_index):
        kernel_Feature = Feature[..., y_index:y_index+kernel_size[0], x_index:x_index+kernel_size[1]]
        return kernel_Feature.flatten().tolist()
    
    out_Feature = []
    if padding:
        in_Feature = Padding(in_Feature, padding)
    for channel in range(out_channels):
            y_Feature = []
            for y in range(0, np.size(in_Feature, 1) - kernel_size[0] + 1, stride[0]):
                x_Feature = []
                for x in range(0, np.size(in_Feature, 2) - kernel_size[1] + 1, stride[1]):
                    # 获取当前卷积核位置对应的节点
                    pred_list = kernel_op(in_Feature, kernel_size, x, y)
                    x_Feature.append([v for v in pred_list if v!=0])

                y_Feature.append(x_Feature)
            out_Feature.append(y_Feature)
    return out_Feature

# Conv2d
class Conv2d(nn.Conv2d):
    def graph_size(self, in_features_shape):
        index = 1
        in_Feature = np.arange(index, index + np.prod(in_features_shape)).reshape(in_features_shape)
        out_Feature = Convolution(in_Feature, self.out_channels, self.kernel_size, self.stride, self.padding) #list
        W = 0
        for a in out_Feature:
            for b in a:
                for c in b:
                    W += len(c)
        out_features_shape = np.array(out_Feature, dtype=object).shape[:3]
        return W, out_features_shape
    
    def Kqi(self, in_features_shape, alpha_volumes, W):
        index = 1
        in_Feature = np.arange(index, index + np.prod(in_features_shape)).reshape(in_features_shape)
        out_Feature = Convolution(in_Feature, self.out_channels, self.kernel_size, self.stride, self.padding) #list

        alpha_volumes, out_Feature = np.array(alpha_volumes), np.array(out_Feature, dtype=object)
        if alpha_volumes.shape != out_Feature.shape:
            alpha_volumes = alpha_volumes.reshape(out_Feature.shape)
        # 计算beta层的volumes
        beta_volumes_dict = {} #存储volume
        for i,A1 in enumerate(out_Feature):
            for j,A2 in enumerate(A1):
                for k,preds in enumerate(A2):
                    # out_Feature中存储了每个节点的所有父亲节点
                    pred_list = preds
                    for l in pred_list:
                        if l not in beta_volumes_dict:
                            beta_volumes_dict[l] = 1 + alpha_volumes[i][j][k]/len(pred_list)
                        else:
                            beta_volumes_dict[l] += 1 + alpha_volumes[i][j][k]/len(pred_list)
        beta_volumes = in_Feature.copy()
        for i,A1 in enumerate(in_Feature):
            for j,A2 in enumerate(A1):
                for k,id in enumerate(A2):
                    beta_volumes[i][j][k] = beta_volumes_dict[id]


        # 计算alpha层的kqi
        kqi = 0
        if alpha_volumes[0][0][0] != 0:
            for i,A1 in enumerate(alpha_volumes):
                for j,A2 in enumerate(A1):
                    for k,volume in enumerate(A2):
                        pred_list = out_Feature[i][j][k]
                        for l in pred_list:
                            kqi += -(volume / len(pred_list) / W) * np.log2(volume / len(pred_list) / beta_volumes_dict[l])

        return kqi, beta_volumes



def Pool_Convolution(in_Feature, kernel_size, stride):
    # 卷积核操作
    def kernel_op(Feature, kernel_size, x_index, y_index):
        kernel_Feature = Feature[..., y_index:y_index+kernel_size, x_index:x_index+kernel_size]
        return kernel_Feature.flatten().tolist()
    
    out_Feature = []
    for channel in in_Feature:
            y_Feature = []
            for y in range(0, np.size(in_Feature, 1) - kernel_size + 1, stride):
                x_Feature = []
                for x in range(0, np.size(in_Feature, 2) - kernel_size + 1, stride):
                    # 获取当前卷积核位置对应的节点
                    pred_list = kernel_op(in_Feature, kernel_size, x, y)
                    # x_Feature.append([v for v in pred_list if v!=0])
                    x_Feature.append(pred_list)
                y_Feature.append(x_Feature)
            out_Feature.append(y_Feature)
    return out_Feature

# MaxPool2d
class MaxPool2d(nn.MaxPool2d):
    def graph_size(self, in_features_shape):
        index = 1
        in_Feature = np.arange(index, index + np.prod(in_features_shape)).reshape(in_features_shape)
        out_Feature = Pool_Convolution(in_Feature, self.kernel_size, self.stride) #list
        W = 0
        for a in out_Feature:
            for b in a:
                for c in b:
                    W += len(c)
        out_features_shape = np.array(out_Feature, dtype=object).shape[:3]
        return W, out_features_shape
    
    def Kqi(self, in_features_shape, alpha_volumes, W):
        index = 1
        in_Feature = np.arange(index, index + np.prod(in_features_shape)).reshape(in_features_shape)
        out_Feature = Pool_Convolution(in_Feature, self.kernel_size, self.stride) #list
        
        alpha_volumes, out_Feature = np.array(alpha_volumes), np.array(out_Feature, dtype=object)
        if alpha_volumes.shape != out_Feature.shape[:3]:
            alpha_volumes = alpha_volumes.reshape(out_Feature.shape[:3])
        # 计算beta层的volumes
        beta_volumes_dict = {} #存储volume
        for i,A1 in enumerate(out_Feature):
            for j,A2 in enumerate(A1):
                for k,preds in enumerate(A2):
                    # out_Feature中存储了每个节点的所有父亲节点
                    pred_list = preds
                    for l in pred_list:
                        if l not in beta_volumes_dict:
                            beta_volumes_dict[l] = 1 + alpha_volumes[i][j][k]/len(pred_list)
                        else:
                            beta_volumes_dict[l] += 1 + alpha_volumes[i][j][k]/len(pred_list)
        beta_volumes = in_Feature.copy()
        for i,A1 in enumerate(in_Feature):
            for j,A2 in enumerate(A1):
                for k,id in enumerate(A2):
                    beta_volumes[i][j][k] = beta_volumes_dict[id]


        # 计算alpha层的kqi
        kqi = 0
        if alpha_volumes[0][0][0] != 0:
            for i,A1 in enumerate(alpha_volumes):
                for j,A2 in enumerate(A1):
                    for k,volume in enumerate(A2):
                        pred_list = out_Feature[i][j][k]
                        for l in pred_list:
                            kqi += -(volume / len(pred_list) / W) * np.log2(volume / len(pred_list) / beta_volumes_dict[l])

        return kqi, beta_volumes


# 待实现
class ReLU(nn.ReLU):
    def non_callable_method(self):
        raise TypeError("This method is not callable.")


class Dropout(nn.Dropout):
    def non_callable_method(self):
        raise TypeError("This method is not callable.")