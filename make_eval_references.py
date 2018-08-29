""" make reference text files needed for ROUGE evaluation """
import argparse
import json
from datetime import timedelta
from time import time

import os
from os.path import join
from tqdm import tqdm

from decoding import make_html_safe
from utils import count_data

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


def dump(split, dump_dir):
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    n_data = count_data(data_dir)
    for i in tqdm(range(n_data)):
        with open(join(data_dir, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        abs_sents = data['abstract']
        with open(join(dump_dir, '{}.ref'.format(i)), 'w') as f:
            f.write(make_html_safe(' '.join(abs_sents)))
    print('finished in {}'.format(timedelta(seconds=time() - start)))


def main(args):
    for split in ['val', 'test']:  # evaluation of train data takes too long
        if not os.path.exists(join(DATA_DIR, split)):
            continue
        ref_dir = args.ref_dir or join(DATA_DIR, 'refs')
        dump_dir = join(ref_dir, split)
        os.makedirs(dump_dir, exist_ok=True)
        dump(split, dump_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Make evaluation references'
    )
    parser.add_argument('--ref_dir')
    args = parser.parse_args()
    main(args)
