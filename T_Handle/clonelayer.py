import copy
import torch
import torch.nn as nn


# 用来将XXX层复制多遍，满足如编码层由6层相同的层构成等的条件
# 在具体的如编码器模型的编写中，不使用nn.Sequential的方法封装达到多个相同层的效果，因为封装的多个相同的层的forward方法会自然调用
# nn.ModuleList相当于一个关于module的list，每个module的forward的方法在需要时才会调用
def clone_layers(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])
