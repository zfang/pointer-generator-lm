""" utility functions"""
import operator as op
import os
import re
from collections import Counter, defaultdict
from functools import reduce
from itertools import product
from os.path import basename

import gensim
import torch
from cytoolz import concat, curry
from torch import multiprocessing as mp
from torch import nn

from openai_transformer_lm.text_utils import TextEncoder


class ModelArguments:
    def __init__(self, **kwargs):
        for k, v in kwargs:
            setattr(self, k, v)


PAD = 0
UNK = 1
START = 2
END = 3

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

TRANSFORMER_LM_ENCODER_PATH = os.path.join(CURRENT_DIR, 'openai_transformer_lm/model/encoder_bpe_40000.json')
TRANSFORMER_LM_BPE_PATH = os.path.join(CURRENT_DIR, 'openai_transformer_lm/model/vocab_40000.bpe')
TRANSFORMER_LM_N_CTX = 512
TRANSFORMER_LM_N_SPECIAL = 2

_PRUNE = defaultdict(
    lambda: 2,
    {1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 4, 7: 3, 8: 3}
)


def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    names = os.listdir(path)
    n_data = len(list(filter(lambda name: bool(matcher.match(name)), names)))
    return n_data


def make_vocab(wc, vocab_size):
    word2id = {
        '<pad>': PAD,
        '<unk>': UNK,
        '<start>': START,
        '<end>': END,
    }
    for i, (w, _) in enumerate(wc.most_common(vocab_size), len(word2id)):
        word2id[w] = i
    return word2id


def make_embedding(id2word, w2v_file, initializer=None):
    attrs = basename(w2v_file).split('.')  # word2vec.{dim}d.{vsize}k.bin
    w2v = gensim.models.Word2Vec.load(w2v_file).wv
    vocab_size = len(id2word)
    emb_dim = int(attrs[-3][:-1])
    embedding = nn.Embedding(vocab_size, emb_dim).weight
    if initializer is not None:
        initializer(embedding)

    oovs = []
    with torch.no_grad():
        for i in range(len(id2word)):
            # NOTE: id2word can be list or dict
            word = None
            if i == START:
                word = '<s>'
            elif i == END:
                word = r'<\s>'
            elif id2word[i] in w2v:
                word = id2word[i]
            else:
                oovs.append(i)

            if word is not None:
                embedding[i, :] = torch.Tensor(w2v[word])

    return embedding, oovs


def rerank(all_beams, ext_inds, debug=False):
    beam_lists = (all_beams[i: i + n] for i, n in ext_inds if n > 0)
    return list(concat(map(rerank_one(debug=debug), beam_lists)))


def rerank_mp(all_beams, ext_inds, debug=False):
    beam_lists = [all_beams[i: i + n] for i, n in ext_inds if n > 0]
    with mp.Pool(os.cpu_count() or 1) as pool:
        reranked = pool.map(rerank_one(debug=debug), beam_lists)
    return list(concat(reranked))


@curry
def rerank_one(beams, debug=False):
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]

    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
    best_hyps = max(product(*beams), key=_compute_score)
    dec_outs = [h.sequence for h in best_hyps]

    if debug:
        return dec_outs, [[t.numpy() for t in hyp.attns[:-1]] for hyp in best_hyps]

    return dec_outs


def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i + n]) for i in range(len(sequence) - (n - 1)))


def _compute_score(hyps):
    all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
    repeat = sum(c - 1 for g, c in all_cnt.items() if c > 1)
    lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
    return -repeat, lp


def get_transformer_lm_encoder():
    text_encoder = TextEncoder(TRANSFORMER_LM_ENCODER_PATH, TRANSFORMER_LM_BPE_PATH)
    encoder = text_encoder.encoder
    n_vocab = len(encoder)
    encoder['_start_'] = n_vocab
    encoder['_classify_'] = n_vocab

    return text_encoder
