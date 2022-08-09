import math

import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        d_comp = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * d_comp)
        pe[:, 1::2] = torch.cos(pos * d_comp)
        # pe1 = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        # self.register_buffer('pe1', pe1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len, d_model]
        :return:
        """
        # x1 = x1 + self.pe1[:x1.size(0), :]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model):
#         super(PositionalEncoding, self).__init__()
#
#         self.d_model = d_model
#
#     def forward(self, seq_len, emb_dim):
#         pe = torch.zeros(seq_len, emb_dim)
#
#         for pos in range(pe.shape[0]):
#             for i in range(pe.shape[1]):
#                 if i % 2 == 0:
#                     pe[pos][i] = torch.sin(torch.tensor(pos / (10000 ** (2 * i / self.d_model))))
#                 else:
#                     pe[pos][i] = torch.cos(torch.tensor(pos / (10000 ** (2 * i / self.d_model))))
#
#         return pe.unsqueeze(0)  # (1, seq_len, emb_dim）(即(1, seq_len, d_model)


# if __name__ == '__main__':
#     pe = PositionalEncoding(2, 0.1)
#
#     t = torch.randn(3, 2, 2)
#
#     s1, s2 = pe(t.transpose(0, 1), t)
#
#     print(t)
#     print()
#     print(s1.transpose(0, 1))
#     print()
#     print(s2)
