import torch
import numpy as np


def get_subsequence_mask(seq):
    """
    防止标签泄露，在t时刻不能看到t时刻之后的信息
    :param seq: [batch_size, tgt_len]
    :return:
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]    # [batch_size, tgt_len, tgt_len]

    subsequence_mask = np.tril(np.ones(attn_shape), k=0)  # 生成一个下三角矩阵，主队对角线以上全为0
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # [batch_size, tgt_len, tgt_len]

    return subsequence_mask

