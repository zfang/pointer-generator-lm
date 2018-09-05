import argparse
import json
from datetime import timedelta
from time import time

import os
import torch
from os.path import join
from torch.utils.data import DataLoader

from data.batcher import tokenize
from data.data import CnnDmDataset
from decoding import Abstractor, BeamAbstractor, make_html_safe
from utils import rerank_mp, count_parameters

try:
    DATASET_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


class DecodeDataset(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""

    def __init__(self, split, dataset_dir):
        assert split in ['val', 'test']
        super().__init__(split, dataset_dir)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        return [' '.join(art_sents)]


class MatchDataset(CnnDmDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """

    def __init__(self, split, dataset_dir):
        super().__init__(split, dataset_dir)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, abs_sents, extracts = (
            js_data['article'], js_data['abstract'], js_data['extracted'])
        matched_arts = [art_sents[i] for i in extracts]
        return matched_arts


def decode(save_path, model_dir, split, batch_size,
           beam_size, diverse, max_len, cuda):
    start = time()
    # setup model
    with open(join(model_dir, 'meta.json')) as f:
        meta = json.loads(f.read())

    if beam_size == 1:
        abstractor = Abstractor(model_dir,
                                max_len,
                                cuda)
    else:
        abstractor = BeamAbstractor(model_dir,
                                    max_len,
                                    cuda)

    # setup loader
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles

    dataset = MatchDataset(split, DATASET_DIR) if args.use_matched else DecodeDataset(split, DATASET_DIR)

    n_data = len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=coll
    )

    # prepare save paths and logs
    os.makedirs(join(save_path, 'output'), exist_ok=True)
    dec_log = {
        'abstractor': meta['net_args'],
        'split': split,
        'beam': beam_size,
        'diverse': diverse
    }
    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)

    print('trainable parameters: {}'.format(count_parameters(abstractor._net, only_trainable=True)))
    print('total parameters: {}'.format(count_parameters(abstractor._net, only_trainable=False)))

    # Decoding
    i = 0
    with torch.no_grad():
        for i_debug, raw_article_batch in enumerate(loader):
            tokenized_article_batch = map(tokenize(None), raw_article_batch)
            ext_arts = []
            ext_inds = []
            for raw_art_sents in tokenized_article_batch:
                ext_inds += [(len(ext_arts), len(raw_art_sents))]
                ext_arts += raw_art_sents
            if beam_size > 1:
                all_beams = abstractor(ext_arts, beam_size, diverse)
                dec_outs = rerank_mp(all_beams, ext_inds)
            else:
                dec_outs = abstractor(ext_arts)
            assert i == batch_size * i_debug
            for j, n in ext_inds:
                decoded_sents = [' '.join(dec) for dec in dec_outs[j:j + n]]
                with open(join(save_path, 'output/{}.dec'.format(i)),
                          'w') as f:
                    f.write(make_html_safe('\n'.join(decoded_sents)))
                i += 1
                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                    i, n_data, i / n_data * 100,
                    timedelta(seconds=int(time() - start))
                ), end='')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='run decoding of the full model (RL)')
    parser.add_argument('--path', required=True, help='path to store/eval')
    parser.add_argument('--model_dir', help='root of the full model')

    # dataset split
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--val', action='store_true', help='use validation set')
    data.add_argument('--test', action='store_true', help='use test set')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=16,
                        help='batch size of faster decoding')
    parser.add_argument('--beam', type=int, action='store', default=4,
                        help='beam size for beam-search (reranking included)')
    parser.add_argument('--div', type=float, action='store', default=0,
                        help='diverse ratio for the diverse beam-search')
    parser.add_argument('--max_dec_word', type=int, action='store', default=120,
                        help='maximun words to be decoded for the abstractor')

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--use-matched', action='store_true')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    data_split = 'test' if args.test else 'val'
    decode(args.path, args.model_dir,
           data_split, args.batch, args.beam, args.div,
           args.max_dec_word, args.cuda)
