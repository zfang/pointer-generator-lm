""" attention functions """
import math

from torch.nn import functional as F


def dot_attention_score(key, query):
    """[B, Tk, D], [(Bs), B, Tq, D] -> [(Bs), B, Tq, Tk]"""
    d_k = query.size(-1)
    return query.matmul(key.transpose(1, 2)) / math.sqrt(d_k)


def attention_aggregate(value, score):
    """[B, Tv, D], [(Bs), B, Tq, Tv] -> [(Bs), B, Tq, D]"""
    output = score.matmul(value)
    return output


def step_attention_score(query, key, mem_mask=None):
    """ query[(Bs), B, D], key[B, T, D], value[B, T, D]"""
    score = dot_attention_score(key, query.unsqueeze(-2))
    if mem_mask is None:
        score = score.masked_fill(mem_mask == 0, -1e18)
    norm_score = F.softmax(score, dim=-1)

    return norm_score, score


def step_attention(query, key, value, mem_mask=None, return_raw_score=False):
    """ query[(Bs), B, D], key[B, T, D], value[B, T, D]"""
    norm_score, score = step_attention_score(query, key, mem_mask=mem_mask, squeeze=False)
    output = attention_aggregate(value, norm_score)
    if return_raw_score:
        return output.squeeze(-2), norm_score.squeeze(-2), score.squeeze(-2)
    else:
        return output.squeeze(-2), norm_score.squeeze(-2)
