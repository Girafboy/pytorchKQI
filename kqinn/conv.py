import torch
import numpy as np

from .kqi import KQI


def _Padding(in_Feature, padding):
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


def _Convolution(in_Feature, out_channels, kernel_size, stride, padding=0):
    # 卷积核操作
    def kernel_op(Feature, kernel_size, x_index, y_index):
        kernel_Feature = Feature[..., y_index:y_index+kernel_size[0], x_index:x_index+kernel_size[1]]
        return kernel_Feature.flatten().tolist()
    
    out_Feature = []
    if padding:
        in_Feature = _Padding(in_Feature, padding)
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


class Conv2d(torch.nn.Conv2d, KQI):
    def KQIforward(self, x: torch.Tensor) -> torch.Tensor:
        index = 1
        in_Feature = np.arange(index, index + np.prod(in_features_shape)).reshape(in_features_shape)
        out_Feature = _Convolution(in_Feature, self.out_channels, self.kernel_size, self.stride, self.padding) #list
        W = 0
        for a in out_Feature:
            for b in a:
                for c in b:
                    KQI.W += len(c)
        out_features_shape = np.array(out_Feature, dtype=object).shape[:3]
        return self.forward(x)
    
    def Kqi(self, in_features_shape, alpha_volumes, W):
        index = 1
        in_Feature = np.arange(index, index + np.prod(in_features_shape)).reshape(in_features_shape)
        out_Feature = _Convolution(in_Feature, self.out_channels, self.kernel_size, self.stride, self.padding) #list

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
