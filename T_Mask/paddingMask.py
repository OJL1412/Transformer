import torch


def get_pad_mask(seq1, seq2):
    """
    用于处理非定长序列，区分padding和非padding部分，进行长度的统一，在对v向量加权平均的时候，可以让pad对应的位置为0，这样注意力就不会考虑到pad向量
    :param seq_1: [batch_size, seq1_len]，可用序列1
    :param seq_2: [batch_size, seq2_len]，可用序列2
    :return:
    """
    # 这2个序列用来扩展纬度，encoder和decoder都可能调用，可能不相等，其长度可能为src_len，也可能为tgt_len
    batch_size, seq1_len = seq1.size()
    batch_size, seq2_len = seq2.size()

    # 判断是否为0，是0则为True，True则masked，并扩一个维度
    padding_mask = seq2.eq(0).unsqueeze(1)  # [batch_size, 1, seq2_len]
    matrix = padding_mask.expand(batch_size, seq1_len, seq2_len)  # [batch_size, seq1_len, seq2_len]

    return matrix
