""" pretrain a word2vec on the corpus"""
import argparse
import json
import logging
import os
from datetime import timedelta
from os.path import join
from time import time

import gensim
from cytoolz import concatv

from utils import count_data

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


class Sentences(object):
    """ needed for gensim word2vec training"""

    def __init__(self):
        self._path = join(DATA_DIR, 'train')
        self._n_data = count_data(self._path)

    def __iter__(self):
        for i in range(self._n_data):
            with open(join(self._path, '{}.json'.format(i))) as f:
                data = json.loads(f.read())
            for s in concatv(data['article'], data['abstract']):
                yield ['<s>'] + s.lower().split() + [r'<\s>']


def main(args):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    start = time()
    save_dir = args.path
    os.makedirs(save_dir, exist_ok=True)

    sentences = Sentences()
    model = gensim.models.Word2Vec(
        size=args.dim, min_count=args.min_count, workers=os.cpu_count() or 4, sg=1)
    model.build_vocab(sentences)
    print('vocab built in {}'.format(timedelta(seconds=time() - start)))

    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=model.iter)

    model.save(join(save_dir, 'word2vec.{}d.{}k.bin'.format(
        args.dim, len(model.wv.vocab) // 1000)))
    model.wv.save_word2vec_format(join(
        save_dir,
        'word2vec.{}d.{}k.w2v'.format(args.dim, len(model.wv.vocab) // 1000)
    ))

    print('word2vec trained in {}'.format(timedelta(seconds=time() - start)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train word2vec embedding used for model initialization'
    )
    parser.add_argument('--path', required=True, help='root of the model')
    parser.add_argument('--dim', action='store', type=int, default=128)
    parser.add_argument('--min-count', action='store', type=int, default=5)
    args = parser.parse_args()

    if args.use_pretrained:
        args.dim = 300

    main(args)
