""" utility functions"""
from itertools import product

import gensim
import operator as op
import os
import re
import torch
from collections import Counter, defaultdict
from cytoolz import concat, curry
from functools import reduce
from os.path import basename
from torch import multiprocessing as mp
from torch import nn

from model.elmo import ElmoLM

PAD = 0
UNK = 1
START = 2
END = 3

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

ELMO_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
ELMO_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"

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
    with mp.get_context("spawn").Pool(os.cpu_count() or 1) as pool:
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


def get_elmo_lm(vocab_to_cache, args):
    new_args = dict(args)
    del new_args['type']
    elmo = ElmoLM(options_file=ELMO_OPTIONS_FILE,
                  weight_file=ELMO_WEIGHT_FILE,
                  vocab_to_cache=vocab_to_cache,
                  **new_args)

    return elmo
