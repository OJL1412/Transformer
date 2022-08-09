import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from T.T_IayerHandle.selfAttention import SelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)

        self.d_k = d_model // n_head  # 512 / 8 = 64
        self.d_v = d_model // n_head  # 512 / 8 = 64

        assert (self.d_k * self.n_head == self.d_model and self.d_v * self.n_head == self.d_model)

        # 嵌入层
        # 这里q,k必须维度相同，不然无法做点积
        self.W_q = nn.Linear(d_model, self.d_k * n_head, bias=False)  # 权值矩阵(512, 512)
        self.W_k = nn.Linear(d_model, self.d_k * n_head, bias=False)  # 权值矩阵(512, 512)
        self.W_v = nn.Linear(d_model, self.d_v * n_head, bias=False)  # 权值矩阵(512, 512)

        # 输出层
        self.o = nn.Linear(n_head * self.d_v, d_model)  # (512, 512)

    def forward(self, q, k, v, mask=None):
        """
        :param q: (batch_size, seq_len_q, d_model)
        :param k: (batch_size, seq_len_k, d_model)
        :param v: (batch_size, seq_len_v(=seq_len_k), d_model)
        :param mask: [batch_size, seq_len, seq_len]
        :return: (batch_size, emb_q, num_head, d_k)
        """

        # 因为是多头，所以mask矩阵要扩充成4维的
        # mask: [batch_size, seq_len, seq_len] => [batch_size, n_heads, seq_len, seq_len]
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        batch_size = q.size(0)

        # (batch_size, -1, 8, 64) => (batch_size, 8, -1, 64)
        # 为每个头分割输入，这里使用view方法对线性变换的结果进行维度重塑，多加了一个维度n_head，代表头数
        # 为了让代表句子长度维度和词向量维度能够相邻，对第二维和第三维进行转置操作,,这样注意力机制才能找到词义与句子位置的关系
        Q = self.W_q(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        # x: [batch_size, n_head, seq_len_q, d_v], attn: [batch_size, n_head, -1, -1]
        x = SelfAttention(Q, K, V, mask)

        # (batch_size, 8, -1, 64) => (batch_size, -1, 8, 64) => (batch_size, 8, 8*64)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_head * self.d_k)

        out = self.o(x)  # (batch_size, 8, 8*64) => (batch_size, 8, 512)

        return self.dropout(out)
