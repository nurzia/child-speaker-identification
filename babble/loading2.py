import errno
import json
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from . import utils

from keras.utils import to_categorical
from math import inf


class DataGenerator:

    def __init__(self, batch_size, folder, seed, pad,
                 cutoff=None, CI_exclude=True, oldcam=False,
                 width=2, min_reduction=None, max_reduction=None,
                 kids_list=None, threshold=None, below_over=None):

        self.folder = folder
        self.batch_size = batch_size
        self.scaler = None
        self.mean_length = None
        self.max_length = None
        self.stddev = None
        self.batch_lens = {s: None for s in ('train', 'dev', 'test')}

        np.random.seed(seed)
        random.seed(seed)

        self.streams = self.init_stream(cutoff=cutoff,
                                        CI_exclude=CI_exclude,
                                        oldcam=oldcam,
                                        min_reduction=min_reduction,
                                        max_reduction=max_reduction,
                                        kids_list=kids_list,
                                        threshold=threshold,
                                        below_over=below_over)

        if pad == 'max':
            self.pad_length = self.max_length
        elif pad == 'mean':
            self.pad_length = self.mean_length
        elif pad == 'interval':
            self.pad_length = int(self.mean_length + width * self.stddev)
        else:
            try:
                self.pad_length = int(pad)
            except ValueError:
                print('invalid padding length')
                return False
                
        self.encoder = LabelEncoder()
        self.encoder.fit([l for _, l in self.streams['train']])

        print(self.encoder)
        print('-> working on classes:', self.encoder.classes_)

    def init_stream(self,
                    cutoff=None, CI_exclude=False, oldcam=False,
                    min_reduction=None, max_reduction=None,
                    kids_list=None, threshold=None, below_over=None):

        meta = pd.read_excel(self.folder + '/meta_wlc.xlsx')
        meta = self.limit_meta(meta=meta, oldcam=oldcam, CI_exclude=CI_exclude,
                               min_reduction=min_reduction, max_reduction=max_reduction,
                               kids_list=kids_list, threshold=threshold, below_over=below_over)
        data = self.unbiased_split(meta=meta, cutoff=cutoff)
        return data

    def limit_meta(self, meta,
                   CI_exclude, oldcam,
                   min_reduction, max_reduction,
                   kids_list=None,
                   threshold=None,
                   below_over=None):

        if CI_exclude == True:
            meta = meta.loc[meta['NH/CI'] == 'NH']
        elif CI_exclude == 2:
            meta = meta.loc[meta['NH/CI'] == 'CI']
        if not oldcam:
            meta = meta.loc[meta['camera'] == 0]
        elif oldcam == 2:
            meta = meta.loc[meta['camera'] == 1]
        if min_reduction:
            meta = meta.loc[meta['length'] > min_reduction]
        if max_reduction:
            meta = meta.loc[meta['length'] < max_reduction]
        if kids_list:
            meta = meta.loc[meta['child'].isin(kids_list)]
        if threshold and below_over and (below_over == 'below'):
            meta = meta.loc[meta['age'] < threshold]
        elif threshold and below_over and (below_over == 'over'):
            meta = meta.loc[meta['age'] > threshold]
        meta_fin = meta
        return meta_fin

    def init_settings(self, data):

        self.mean_length = int(np.mean([data[i]['length'].mean() for i in data]))
        self.max_length = np.max([data[i]['length'].max() for i in data])
        self.stddev = np.mean([data[i]['length'].std() for i in data])

    def unbiased_split(self, meta, cutoff=1000):

        children = set(meta['child'])
        children_meta = pd.DataFrame()
        for child in children:
            child_list = meta.loc[meta['child'] == child]
            if len(child_list.index) < cutoff:
                continue
            else:
                child_list = child_list.sample(cutoff, random_state=123)
            children_meta = pd.concat([children_meta, child_list])
        children_meta = children_meta.sample(frac=1)

        train_meta, dev_meta, test_meta = utils.metadata_split(children_meta)

        data = {'train': train_meta, 'dev': dev_meta, 'test': test_meta}
        self.init_settings(data=data)
        data = {i: list(zip(data[i]['filename'], data[i]['child'])) for i in data}
        return data

    def get_batches(self, stream,  fitting=False,
                    endless=False, sample_size=None):
        if not fitting and not self.scaler:
            raise ValueError('Scaler not fitted yet: call fit_scaler() first!')
        while True:
            X, Y = [], []
            random.shuffle(self.streams[stream])
            num_batches = 0
            if not sample_size:
                list_ = self.streams[stream]
            else:
                list_ = random.sample(self.streams[stream], sample_size)
            for idx, (fn, y) in enumerate(list_):
                try:
                    spec = np.load(self.folder + '/spec/' + fn + '.npy')
                    X.append(spec)
                    Y.append(y)
                except OSError as e:
                    if e.errno == errno.ENOENT:
                        continue
                    else:
                        raise

                if len(Y) == self.batch_size or idx == len(self.streams[stream]) - 1:
                    if fitting:
                        yield (X, Y)
                    else:
                        X = [self.scaler.transform(x) for x in X]
                        X = [utils.pad(x, self.pad_length) for x in X]

                        Y = self.encoder.transform(Y)
                        Y = to_categorical(Y, num_classes=len(self.encoder.classes_))
                        yield (np.array(X, dtype=np.float32), np.array(Y, dtype=np.int32))
                    num_batches += 1

                    X, Y = [], []

            if not self.batch_lens[stream]:
                self.batch_lens[stream] = num_batches
            if not endless:
                break

    def fit_scaler(self, stream='train'):
        self.scaler = StandardScaler()
        for X, _ in self.get_batches(stream, fitting=True, endless=False):
            self.scaler.partial_fit(np.vstack(X))

    def get_num_batches(self, stream='train'):
        if not self.batch_lens[stream]:  # do idle pass pass over stream:
            [b for b in self.get_batches(stream, fitting=False, endless=False)]
        return self.batch_lens[stream]
