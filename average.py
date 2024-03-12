import copy
import torch
# torch 的 nn 子模块提供了创建和操作神经网络层和模块的类和函数。
from torch import nn


def average_weights(w, s_num):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  # the nn layer loop
        for i in range(1, len(w)):  # the client loop
            w_avg[k] += torch.mul(w[i][k], s_num[i] / temp_sample_num)
        w_avg[k] = torch.mul(w_avg[k], temp_sample_num / total_sample_num)
    return w_avg
