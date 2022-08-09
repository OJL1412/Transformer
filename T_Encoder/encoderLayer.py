import torch
import torch.nn as nn

from T.T_IayerHandle.feedForward import FeedForward
from T.T_IayerHandle.multiHeadAttention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout):
        """
        编码层，由2层子层构成：多头注意力机制 + 位置全连接的前馈网络
        :param d_model: 默认为512
        :param d_ff: 可设为为2048
        :param n_head: 头数
        :param dropout: 进行dropout操作时置0比率
        """
        super(EncoderLayer, self).__init__()

        # Multi-Head Attention
        self.attn = MultiHeadAttention(d_model, n_head, dropout)

        # Feed Forward
        self.FF = FeedForward(d_model, d_ff, dropout)

        # Norm
        self.norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, ec_inp, mask):
        """
        :param ec_inp: [batch_size, src_seq_len, d_model]
        :param mask: [batch_size, src_seq_len, src_seq_len]
        :return:
        """
        # Multi-Head Attention 和 Add & Norm
        x = ec_inp
        ec_out = self.attn(x, x, x, mask)  # (batch_size, 8, 512)
        ec_out = self.norm(ec_out + x)
        # print("编码层——多头注意力机制结果:")
        # print(x.size())
        # print(x.size())
        # print(x.size())
        # print("**" * 30)

        # Feed Forward 和 Add & Norm
        ff_out = self.FF(ec_out)
        ff_out = self.norm(ff_out + x)

        return self.dropout(ff_out)
