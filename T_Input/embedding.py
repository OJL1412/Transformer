import math
import torch

import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, v_size, d_model):
        """
        一个普通的 embedding层，我们可以通过设置padding_idx=0来实现论文中的padding_mask
        :param v_size:
        :param d_model:
        """
        super(Embedding, self).__init__()

        self.d_model = d_model

        # self.embedding = nn.Embedding(v_size, d_model, padding_idx=0)
        self.embedding = nn.Embedding(v_size, d_model)

    def forward(self, x):
        """
        根据每个句子的长度，进行padding，短补长截
        :param x: (batch_size, seq_len)
        """
        x = self.embedding(x)  # (batch_size, seq_len) => (batch_size, seq_len, d_model)

        return x * math.sqrt(self.d_model)  # ?

