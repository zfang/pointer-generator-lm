""" batching """
import random
import torch
import torch.multiprocessing as mp
from collections import defaultdict
from cytoolz import curry, concat
from toolz.sandbox import unzip


# Batching functions
def coll_fn(data):
    source_lists, target_lists = unzip(data)
    # NOTE: independent filtering works because
    #       source and targets are matched properly by the Dataset
    sources = list(filter(bool, concat(source_lists)))
    targets = list(filter(bool, concat(target_lists)))
    assert all(sources) and all(targets)
    return sources, targets


def coll_fn_extract(data):
    def is_good_data(d):
        """ make sure data is not empty"""
        source_sents, extracts = d
        return source_sents and extracts

    batch = list(filter(is_good_data, data))
    assert all(map(is_good_data, batch))
    return batch


@curry
def tokenize(max_len, texts):
    return [t.lower().split()[:max_len] for t in texts]


def convert2id(unk, word2id, words_list):
    word2id = defaultdict(lambda: unk, word2id)
    return [[word2id[w] for w in words] for words in words_list]


@curry
def prepro_fn(max_src_len, max_tgt_len, batch):
    sources, targets = batch
    sources = tokenize(max_src_len, sources)
    targets = tokenize(max_tgt_len, targets)
    batch = list(zip(sources, targets))
    return batch


@curry
def convert_batch_copy(unk, word2id, batch):
    sources, targets = map(list, unzip(batch))
    ext_word2id = dict(word2id)
    for source in sources:
        for word in source:
            if word not in ext_word2id:
                ext_word2id[word] = len(ext_word2id)
    sources_ids = convert2id(unk, word2id, sources)
    src_exts_ids = convert2id(unk, ext_word2id, sources)
    tar_ins_ids = convert2id(unk, word2id, targets)
    targets_ids = convert2id(unk, ext_word2id, targets)

    batch = list(zip(sources_ids, src_exts_ids, tar_ins_ids, targets_ids))

    return batch


@curry
def pad_batch_tensorize(inputs, pad, cuda=True):
    """pad_batch_tensorize

    :param inputs: List of size B containing torch tensors of shape [T, ...]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    batch_size = len(inputs)
    max_len = max(len(ids) for ids in inputs)
    tensor_shape = (batch_size, max_len)
    tensor = tensor_type(*tensor_shape)
    tensor.fill_(pad)
    for i, ids in enumerate(inputs):
        tensor[i, :len(ids)] = tensor_type(ids)
    return tensor


@curry
def batchify_fn_copy(pad, start, end, data, cuda=True):
    sources, ext_srcs, tar_ins, targets = tuple(map(list, unzip(data)))

    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    src_lens = tensor_type([len(src) for src in sources])

    sources = [src for src in sources]
    ext_srcs = [ext for ext in ext_srcs]

    tar_ins = [[start] + tgt for tgt in tar_ins]
    targets = [tgt + [end] for tgt in targets]

    source = pad_batch_tensorize(sources, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)
    ext_src = pad_batch_tensorize(ext_srcs, pad, cuda)

    ext_vsize = ext_src.max().item() + 1

    fw_args = (source, src_lens, tar_in, ext_src, ext_vsize)
    loss_args = (target,)

    return fw_args, loss_args


def _batch2q(loader, prepro, q, single_run=True):
    epoch = 0
    while True:
        for batch in loader:
            q.put(prepro(batch))
        if single_run:
            break
        epoch += 1
        q.put(epoch)
    q.put(None)


class BucketedGenerater(object):
    def __init__(self, loader, prepro,
                 sort_key, batchify,
                 single_run=True, queue_size=8, fork=True):
        self._loader = loader
        self._prepro = prepro
        self._sort_key = sort_key
        self._batchify = batchify
        self._single_run = single_run
        if fork:
            ctx = mp.get_context('forkserver')
            self._queue = ctx.Queue(queue_size)
        else:
            # for easier debugging
            self._queue = None
        self._process = None

    def __call__(self, batch_size: int):
        def get_batches(hyper_batch):
            indexes = list(range(0, len(hyper_batch), batch_size))
            if not self._single_run:
                # random shuffle for training batches
                random.shuffle(hyper_batch)
                random.shuffle(indexes)
            hyper_batch.sort(key=self._sort_key)
            for i in indexes:
                batch = self._batchify(hyper_batch[i:i + batch_size])
                yield batch

        if self._queue is not None:
            ctx = mp.get_context('forkserver')
            self._process = ctx.Process(
                target=_batch2q,
                args=(self._loader, self._prepro,
                      self._queue, self._single_run)
            )
            self._process.start()
            while True:
                d = self._queue.get()
                if d is None:
                    break
                if isinstance(d, int):
                    print('\nepoch {} done'.format(d))
                    continue
                yield from get_batches(d)
            self._process.join()
        else:
            i = 0
            while True:
                for batch in self._loader:
                    yield from get_batches(self._prepro(batch))
                if self._single_run:
                    break
                i += 1
                print('\nepoch {} done'.format(i))

    def terminate(self):
        if self._process is not None:
            self._process.terminate()
            self._process.join()
