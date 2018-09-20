""" Evaluate model on ROUGE/METEOR"""
import argparse
import json

import os
from os.path import join, exists

from evaluate import eval_meteor, eval_rouge, eval_novel_ngrams

try:
    _DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


def main(args):
    dec_dir = join(args.decode_dir, 'output')
    with open(join(args.decode_dir, 'log.json')) as f:
        split = json.loads(f.read())['split']
    ref_dir = args.ref_dir or join(_DATA_DIR, 'refs', split)
    assert exists(ref_dir)

    if args.rouge:
        dec_pattern = r'(\d+).dec'
        ref_pattern = '#ID#.ref'
        output = eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir)
        metric = 'rouge'
    elif args.meteor:
        dec_pattern = '[0-9]+.dec'
        ref_pattern = '[0-9]+.ref'
        output = eval_meteor(dec_pattern, dec_dir, ref_pattern, ref_dir)
        metric = 'meteor'
    elif args.novel_ngrams:
        eval_novel_ngrams_args = {
            'data_dir': join(_DATA_DIR, split),
            'data_pattern': '[0-9]+.json',
            'dec_dir': dec_dir,
            'dec_pattern': '[0-9]+.dec',
            'ref_dir': ref_dir,
            'ref_pattern': '[0-9]+.ref',
        }

        output = json.dumps(eval_novel_ngrams(**eval_novel_ngrams_args), indent=4, ensure_ascii=False)
        metric = 'novel-ngrams'
    else:
        raise NotImplementedError()

    print(output)
    with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
        f.write(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the output files for the models')

    # choose metric to evaluate
    metric_opt = parser.add_mutually_exclusive_group(required=True)
    metric_opt.add_argument('--rouge', action='store_true',
                            help='ROUGE evaluation')
    metric_opt.add_argument('--meteor', action='store_true',
                            help='METEOR evaluation')
    metric_opt.add_argument('--novel-ngrams', action='store_true',
                            help='Novel ngrams evaluation')

    parser.add_argument('--decode_dir', action='store', required=True,
                        help='directory of decoded summaries')
    parser.add_argument('--ref_dir', action='store',
                        help='directory of reference summaries')

    args = parser.parse_args()
    main(args)
