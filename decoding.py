""" decoding utilities"""
import json
import logging
import pickle as pkl
from itertools import starmap

import numpy as np
import os
import re
import torch
from cytoolz import curry
from os.path import join

from data.batcher import convert2id, pad_batch_tensorize
from data.data import CnnDmDataset
from model.copy_summ import CopySumm
from utils import PAD, UNK, START, END, get_elmo_lm


class DecodeDataset(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""

    def __init__(self, split, dataset_dir):
        assert split in ['val', 'test']
        super().__init__(split, dataset_dir)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        return art_sents


def make_html_safe(s):
    """Rouge use html, has to make output html safe"""
    return s.replace("<", "&lt;").replace(">", "&gt;")


def load_best_ckpt(model_dir, reverse=False):
    """ reverse=False->loss, reverse=True->reward/score"""
    ckpts = os.listdir(join(model_dir, 'ckpt'))
    ckpt_matcher = re.compile('^ckpt-.*-[0-9]*')
    ckpts = sorted([c for c in ckpts if ckpt_matcher.match(c)],
                   key=lambda c: float(c.split('-')[1]), reverse=reverse)
    logging.info('loading checkpoint {}...'.format(ckpts[0]))
    ckpt = torch.load(
        join(model_dir, 'ckpt/{}'.format(ckpts[0])),
        map_location=lambda storage, loc: storage
    )['state_dict']
    return ckpt, ckpts[0]


def load_last_ckpt(model_dir):
    ckpts = os.listdir(join(model_dir, 'ckpt'))
    ckpt_matcher = re.compile('^ckpt-.*-[0-9]*')
    ckpts = sorted([c for c in ckpts if ckpt_matcher.match(c)],
                   key=lambda c: float(c.split('-')[2]), reverse=True)
    logging.info('loading checkpoint {}...'.format(ckpts[0]))
    ckpt = torch.load(
        join(model_dir, 'ckpt/{}'.format(ckpts[0])),
        map_location=lambda storage, loc: storage
    )['state_dict']
    return ckpt, ckpts[0]


class Abstractor(object):
    def __init__(self, abs_dir, max_len=30, cuda=True, restore_last_model=False):
        abs_meta = json.load(open(join(abs_dir, 'meta.json')))
        assert abs_meta['net'] == 'base_abstractor'
        abs_args = abs_meta['net_args']
        if restore_last_model:
            abs_ckpt, ckpt_name = load_last_ckpt(abs_dir)
        else:
            abs_ckpt, ckpt_name = load_best_ckpt(abs_dir)

        word2id = pkl.load(open(join(abs_dir, 'vocab.pkl'), 'rb'))

        language_model = None
        language_model_arg = abs_meta['language_model']
        if language_model_arg['type'] is not None:
            if language_model_arg['type'] == 'elmo':
                id2words = {i: w for w, i in word2id.items()}
                language_model = get_elmo_lm(vocab_to_cache=[id2words[i] for i in range(len(id2words))],
                                             args=language_model_arg)
            else:
                raise NotImplementedError(language_model_arg)

        abstractor = CopySumm(**abs_args, language_model=language_model)

        abstractor.load_state_dict(abs_ckpt)
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = abstractor.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}
        self._max_len = max_len
        self._abs_meta = abs_meta
        self._loaded_ckpt_name = ckpt_name

    def _prepro(self, raw_article_sents):
        ext_word2id = dict(self._word2id)
        ext_id2word = dict(self._id2word)
        for raw_words in raw_article_sents:
            for w in raw_words:
                if w not in ext_word2id:
                    ext_word2id[w] = len(ext_word2id)
                    ext_id2word[len(ext_id2word)] = w
        articles = convert2id(UNK, self._word2id, raw_article_sents)
        art_lens = [len(art) for art in articles]
        article = pad_batch_tensorize(articles, PAD, cuda=False
                                      ).to(self._device)
        extend_arts = convert2id(UNK, ext_word2id, raw_article_sents)
        extend_art = pad_batch_tensorize(extend_arts, PAD, cuda=False
                                         ).to(self._device)
        extend_vsize = len(ext_word2id)
        dec_args = (article, art_lens, extend_art, extend_vsize,
                    START, END, UNK, self._max_len)
        return dec_args, ext_id2word

    def __call__(self, raw_article_sents, debug=False):
        self._net.eval()
        dec_args, id2word = self._prepro(raw_article_sents)
        decs, attns = self._net.batch_decode(*dec_args)

        def argmax(arr, keys):
            return arr[max(range(len(arr)), key=lambda i: keys[i].item())]

        dec_sents = []
        for i, raw_words in enumerate(raw_article_sents):
            dec = []
            for id_, attn in zip(decs, attns):
                if id_[i] == END:
                    break
                elif id_[i] == UNK:
                    dec.append(argmax(raw_words, attn[i]))
                else:
                    dec.append(id2word[id_[i].item()])
            dec_sents.append(dec)

        if debug:
            abs_attns = np.array([t.numpy() for t in attns]).transpose((1, 0, 2))
            return dec_sents, [[t for t in attn[:len(dec_sents[i])]] for i, attn in enumerate(abs_attns)]

        return dec_sents


class BeamAbstractor(Abstractor):
    def __call__(self, raw_article_sents, beam_size=5, diverse=1.0):
        self._net.eval()
        dec_args, id2word = self._prepro(raw_article_sents)
        dec_args = (*dec_args, beam_size, diverse)
        all_beams = self._net.batched_beamsearch(*dec_args)
        all_beams = list(starmap(_process_beam(id2word),
                                 zip(all_beams, raw_article_sents)))
        return all_beams


@curry
def _process_beam(id2word, beam, art_sent):
    def process_hyp(hyp):
        seq = []
        for i, attn in zip(hyp.sequence[1:], hyp.attns[:-1]):
            if i == UNK:
                seq.append(art_sent[max(range(len(art_sent)), key=lambda j: attn[j].item())])
            else:
                seq.append(id2word[i])
        hyp.sequence = seq
        del hyp.hists
        return hyp

    return list(map(process_hyp, beam))


class ArticleBatcher(object):
    def __init__(self, word2id, cuda=True):
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._word2id = word2id
        self._device = torch.device('cuda' if cuda else 'cpu')

    def __call__(self, raw_article_sents):
        articles = convert2id(UNK, self._word2id, raw_article_sents)
        article = pad_batch_tensorize(articles, PAD, cuda=False
                                      ).to(self._device)
        return article
