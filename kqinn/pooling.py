import numpy as np 
import torch.nn as nn


def _Pool_Convolution(in_Feature, kernel_size, stride):
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


class MaxPool2d(nn.MaxPool2d):
    def graph_size(self, in_features_shape):
        index = 1
        in_Feature = np.arange(index, index + np.prod(in_features_shape)).reshape(in_features_shape)
        out_Feature = _Pool_Convolution(in_Feature, self.kernel_size, self.stride) #list
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
        out_Feature = _Pool_Convolution(in_Feature, self.kernel_size, self.stride) #list
        
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