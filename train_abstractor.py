""" train the abstractor"""
import argparse
import json
import pickle as pkl

import numpy as np
import os
import random
import torch
from cytoolz import compose, curry
from os.path import join, exists
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data.batcher import BucketedGenerater
from data.batcher import coll_fn, prepro_fn
from data.batcher import convert_batch_copy, batchify_fn_copy
from data.data import CnnDmDataset
from decoding import Abstractor
from model.copy_summ import CopySumm
from model.util import sequence_loss
from training import BasicPipeline, BasicTrainer
from training import get_basic_grad_fn, basic_validate
from utils import PAD, UNK, START, END, get_elmo_lm
from utils import make_vocab, make_embedding

# NOTE: bucket size too large may sacrifice randomness,
#       to low may increase # of PAD tokens
BUCKET_SIZE = 6400

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


class ConcatenatedDataset(CnnDmDataset):
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, abs_sents = js_data['article'], js_data['abstract']
        return [' '.join(art_sents)], [' '.join(abs_sents)]


def configure_training(opt, lr, clip_grad, lr_decay, batch_size, lm_coef):
    """ supports Adam optimizer only"""
    assert opt in ['adam']
    opt_kwargs = {'lr': lr}

    train_params = {
        'optimizer': (opt, opt_kwargs),
        'clip_grad_norm': clip_grad,
        'batch_size': batch_size,
        'lr_decay': lr_decay
    }

    def nll(logit, target):
        return F.nll_loss(logit, target, reduce=False)

    @curry
    def criterion(output, targets, training):
        logits = output['logit']
        loss = sequence_loss(logits, targets, nll, pad_idx=PAD)

        lm_args = output.get('lm')
        if training and lm_coef > 0 and lm_args is not None:
            article, lm_output = lm_args
            lm_loss = sequence_loss(lm_output, article, nll, pad_idx=PAD)
            lm_loss = lm_coef * lm_loss.mean()
            lm_loss.backward(retain_graph=True)

        return loss

    return criterion, train_params


def build_batchers(word2id, cuda, debug, dataset):
    prepro = prepro_fn(args.max_art, args.max_abs)

    def sort_key(sample):
        src, target = sample
        return len(target), len(src)

    batchify = compose(
        batchify_fn_copy(PAD, START, END, cuda=cuda),
        convert_batch_copy(UNK, word2id)
    )

    train_loader = DataLoader(
        dataset('train'), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=not debug)

    val_loader = DataLoader(
        dataset('val'), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                    single_run=True, fork=not debug)
    return train_batcher, val_batcher


def main(args):
    # configure training setting
    criterion, train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch, args.lm_coef
    )

    # make net
    if args.restore_model:
        abstractor = Abstractor(args.path, args.max_abs, args.cuda)
        word2id = abstractor._word2id
        meta = abstractor._abs_meta
        net = abstractor._net
        meta['training_params'] = train_params
    else:
        with open(join(DATA_DIR, 'vocab_cnt.pkl'), 'rb') as f:
            wc = pkl.load(f)
        word2id = make_vocab(wc, args.vsize)

        net_args = {
            'vocab_size': len(word2id),
            'emb_dim': args.emb_dim,
            'n_hidden': args.n_hidden,
            'bidirectional': args.bi,
            'n_layer': args.n_layer,
            'dropout': args.dropout,
        }

        language_model_args = {
            'type': args.lm,
            'requires_grad': args.lm_requires_grad,
            'do_layer_norm': args.lm_layer_norm,
            'dropout': args.lm_dropout,
        }

        id2words = {i: w for w, i in word2id.items()}
        language_model = None
        if language_model_args['type'] == 'elmo':
            language_model = get_elmo_lm(vocab_to_cache=[id2words[i] for i in range(len(id2words))],
                                         args=language_model_args)

        net = CopySumm(**net_args, language_model=language_model)

        meta = {
            'net': 'base_abstractor',
            'net_args': net_args,
            'training_params': train_params,
            'language_model': language_model_args,
        }

        if args.w2v:
            # NOTE: the pretrained embedding having the same dimension
            #       as args.emb_dim should already be trained
            embedding, _ = make_embedding(
                id2words, args.w2v)
            net.set_embedding(embedding)

        # save experiment setting
        if not exists(args.path):
            os.makedirs(args.path)
        with open(join(args.path, 'vocab.pkl'), 'wb') as f:
            pkl.dump(word2id, f, pkl.HIGHEST_PROTOCOL)

    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    # create data batcher, vocabulary
    dataset = ConcatenatedDataset
    train_batcher, val_batcher = build_batchers(word2id,
                                                args.cuda,
                                                args.debug,
                                                dataset)

    # prepare trainer
    val_fn = basic_validate(net, criterion(training=False))
    grad_fn = get_basic_grad_fn(net, args.clip)
    parameters_opt = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(parameters_opt, **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)

    if args.cuda:
        net = net.cuda()
    pipeline = BasicPipeline(meta['net'], net,
                             train_batcher, val_batcher, args.batch, val_fn,
                             criterion(training=True), optimizer, grad_fn)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler)

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of the abstractor (ML)'
    )
    parser.add_argument('--path', required=True, help='root of the model')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--vsize', type=int, action='store', default=30000,
                        help='vocabulary size')
    parser.add_argument('--emb_dim', type=int, action='store', default=128,
                        help='the dimension of word embedding')
    parser.add_argument('--w2v', action='store',
                        help='use pretrained word2vec embedding')
    parser.add_argument('--n_hidden', type=int, action='store', default=256,
                        help='the number of hidden units of LSTM')
    parser.add_argument('--n_layer', type=int, action='store', default=1,
                        help='the number of layers of LSTM')
    parser.add_argument('--no-bi', action='store_true',
                        help='disable bidirectional LSTM encoder')

    # length limit
    parser.add_argument('--max_art', type=int, action='store', default=400,
                        help='maximun words in a single article sentence')
    parser.add_argument('--max_abs', type=int, action='store', default=100,
                        help='maximun words in a single abstract sentence')
    # training options
    parser.add_argument('--lr', type=float, action='store', default=1e-3,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=4,
                        help='patience for learning rate decay')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=16,
                        help='the training batch size')
    parser.add_argument('--dropout', type=float, default=0,
                        help='the probability for dropout')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=3000,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=5,
                        help='patience for early stopping')

    parser.add_argument('--debug', action='store_true',
                        help='run in debugging mode')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--restore-model', action='store_true',
                        help='Restore from the best model')
    parser.add_argument('--lm', default=None, choices=('elmo',),
                        help='Use pre-trained language model')
    parser.add_argument('--lm-coef', type=float, default=0)
    parser.add_argument('--lm-requires-grad', action='store_true')
    parser.add_argument('--lm-layer-norm', action='store_true')
    parser.add_argument('--lm-dropout', type=float, default=0)
    args = parser.parse_args()
    args.bi = not args.no_bi
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    main(args)
