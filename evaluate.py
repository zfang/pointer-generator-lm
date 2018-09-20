""" evaluation scripts"""
import json
import logging
import subprocess as sp
from itertools import starmap

import os
import re
from cytoolz import curry
from functools import reduce
from os.path import join
from os.path import normpath, basename
from pyrouge import Rouge155
from pyrouge.utils import log

from data.batcher import tokenize

try:
    _ROUGE_PATH = os.environ['ROUGE']
except KeyError:
    print('Warning: ROUGE is not configured')
    _ROUGE_PATH = None


def eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir,
               cmd='-c 95 -r 1000 -n 2 -m', system_id=1):
    """ evaluate by original Perl implementation"""
    assert _ROUGE_PATH is not None
    # silence pyrouge logging
    log.get_global_console_logger().setLevel(logging.WARNING)
    rouge_dec = join(dec_dir, '../rouge_dec')
    Rouge155.convert_summaries_to_rouge_format(
        dec_dir, rouge_dec)
    rouge_ref = join(ref_dir, '../rouge_{}_ref'.format(basename(normpath(ref_dir))))
    Rouge155.convert_summaries_to_rouge_format(
        ref_dir, rouge_ref)
    rouge_settings = join(dec_dir, '../rouge_settings.xml')
    Rouge155.write_config_static(
        rouge_dec, dec_pattern,
        rouge_ref, ref_pattern,
        rouge_settings, system_id
    )
    cmd = (join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
           + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
           + cmd
           + ' -a {}'.format(rouge_settings))
    output = sp.check_output(cmd.split(' '), universal_newlines=True)
    return output


try:
    _METEOR_PATH = os.environ['METEOR']
except KeyError:
    print('Warning: METEOR is not configured')
    _METEOR_PATH = None


def eval_meteor(dec_pattern, dec_dir, ref_pattern, ref_dir):
    """ METEOR evaluation"""
    assert _METEOR_PATH is not None
    ref_matcher = re.compile(ref_pattern)
    refs = sorted([r for r in os.listdir(ref_dir) if ref_matcher.match(r)],
                  key=lambda name: int(name.split('.')[0]))
    dec_matcher = re.compile(dec_pattern)
    decs = sorted([d for d in os.listdir(dec_dir) if dec_matcher.match(d)],
                  key=lambda name: int(name.split('.')[0]))

    @curry
    def read_file(file_dir, file_name):
        with open(join(file_dir, file_name)) as f:
            return ' '.join(f.read().split())

    meteor_dec = join(dec_dir, '../meteor_dec.txt')
    with open(meteor_dec, 'w') as dec_f:
        dec_f.write('\n'.join(map(read_file(dec_dir), decs)) + '\n')

    meteor_ref = join(ref_dir, '../meteor_{}_ref.txt'.format(basename(normpath(ref_dir))))
    with open(meteor_ref, 'w') as ref_f:
        ref_f.write('\n'.join(map(read_file(ref_dir), refs)) + '\n')

    cmd = 'java -Xmx2G -jar {} {} {} -l en -norm'.format(
        _METEOR_PATH, meteor_dec, meteor_ref)
    output = sp.check_output(cmd.split(' '), universal_newlines=True)
    return output


def eval_novel_ngrams(data_pattern, data_dir, dec_pattern, dec_dir, ref_pattern, ref_dir, max_n=4):
    """ Novel ngrams evaluation"""
    data_matcher = re.compile(data_pattern)
    data_files = sorted([d for d in os.listdir(data_dir) if data_matcher.match(d)],
                        key=lambda name: int(name.split('.')[0]))
    dec_matcher = re.compile(dec_pattern)
    dec_files = sorted([d for d in os.listdir(dec_dir) if dec_matcher.match(d)],
                       key=lambda name: int(name.split('.')[0]))
    ref_matcher = re.compile(ref_pattern)
    ref_files = sorted([r for r in os.listdir(ref_dir) if ref_matcher.match(r)],
                       key=lambda name: int(name.split('.')[0]))

    @curry
    def read_data_file(file_dir, file_name):
        with open(join(file_dir, file_name)) as f:
            return tokenize(None, json.load(f)['article'])

    @curry
    def read_file(file_dir, file_name):
        with open(join(file_dir, file_name)) as f:
            return tokenize(None, f.read().splitlines())

    arts = list(map(read_data_file, data_files))
    decs = list(map(read_file, dec_files))
    refs = list(map(read_file, ref_files))

    arts_ngrams = [[get_ngrams(sents, i) for i in range(1, max_n + 1)] for sents in arts]
    decs_ngrams = [[get_ngrams(sents, i) for i in range(1, max_n + 1)] for sents in decs]
    refs_ngrams = [[get_ngrams(sents, i) for i in range(1, max_n + 1)] for sents in refs]

    decs_novel_ngram_ratios = [list(starmap(find_novel_ngram_ratios, zip(d, a))) for d, a in
                               zip(decs_ngrams, arts_ngrams)]
    refs_novel_ngram_ratios = [list(starmap(find_novel_ngram_ratios, zip(r, a))) for r, a in
                               zip(refs_ngrams, arts_ngrams)]

    joined_arts = [' '.join(' '.join(tokens) for tokens in sents) for sents in arts]

    decs_novel_sent_ratios = [sum(map(lambda x: int(x not in joined_arts[i]),
                                      (' '.join(tokens) for tokens in sents))) / len(sents)
                              for i, sents in enumerate(decs)]

    refs_novel_sent_ratios = [sum(map(lambda x: int(x not in joined_arts[i]),
                                      (' '.join(tokens) for tokens in sents))) / len(sents)
                              for i, sents in enumerate(refs)]

    decs_novel_ngram_ratios_all = [n.append(s) for n, s in zip(decs_novel_ngram_ratios, decs_novel_sent_ratios)]

    refs_novel_ngram_ratios_all = [n.append(s) for n, s in zip(refs_novel_ngram_ratios, refs_novel_sent_ratios)]

    def compute_averages(ratios):
        ratio_sums = reduce(lambda x, y: [sum(a, b) for a, b in zip(x, y)], ratios)
        return [ratio_sum / len(ratios) for ratio_sum in ratio_sums]

    decs_novel_ngram_ratio_means = compute_averages(decs_novel_ngram_ratios_all)
    decs_novel_ngram_ratio_data = {
        'dec_novel_{}_gram_ratio'.format(i): r for i, r in enumerate(decs_novel_ngram_ratio_means)
        if i != len(decs_novel_ngram_ratio_means) - 1
    }
    decs_novel_ngram_ratio_data['dec_novel_sent_ratio'] = decs_novel_ngram_ratio_means[-1]

    refs_novel_ngram_ratio_means = compute_averages(refs_novel_ngram_ratios_all)
    refs_novel_ngram_ratio_data = {
        'ref_novel_{}_gram_ratio'.format(i): r for i, r in enumerate(refs_novel_ngram_ratio_means)
        if i != len(refs_novel_ngram_ratio_means) - 1
    }
    refs_novel_ngram_ratio_data['ref_novel_sent_ratio'] = refs_novel_ngram_ratio_means[-1]

    return {**decs_novel_ngram_ratio_data, **refs_novel_ngram_ratio_data}


def find_novel_ngrams(dec_ngrams, art_ngrams):
    return [dec - art for dec, art in zip(dec_ngrams, art_ngrams)]


def find_novel_ngram_ratios(dec_ngrams, art_ngrams):
    return [len(n) / len(d) for n, d in zip(find_novel_ngrams(dec_ngrams, art_ngrams), dec_ngrams)]


def get_ngrams(sents, n):
    assert n >= 1
    if n == 1:
        return set(token for sent in sents for token in sent)
    else:
        return set(' '.join(ngram) for sent in sents for ngram in zip(*[sent[j:] for j in range(n)]))
