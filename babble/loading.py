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
                 train_names=[], dev_names=[], test_names=[],
                 nh_ci=False, task='child_id', eps=1e-8,
                 threshold=None, confines=None,
                 cutoff=None, CI_exclude=False,
                 width=2, min_reduction=None, max_reduction=None,
                 oldcam=False, kids_cut=None, unbiased=False):

        self.folder = folder
        self.batch_size = batch_size
        self.eps = eps
        self.scaler = None

        self.mean_length = None
        self.max_length = None
        self.stddev = None

        print('threshold: ', threshold)
        print('confines: ', confines)
        print('cutoff: ', cutoff)

        if threshold:
            self.batch_lens = {s: None for s in ('young_train', 'young_dev', 'young_test',
                                                 'old_train', 'old_dev', 'old_test')}
        elif confines:
            self.batch_lens = {s: None for s in ('young_train', 'young_dev', 'young_test',
                                                 'median_train', 'median_dev', 'median_test',
                                                 'old_train', 'old_dev', 'old_test')}
        else:
            self.batch_lens = {s: None for s in ('train', 'dev', 'test')}

        np.random.seed(seed)
        random.seed(seed)
        print(folder + '/stats.json')

        if nh_ci:
            self.streams = self.init_stream_folds(train_names=train_names,
                                                  dev_names=dev_names,
                                                  test_names=test_names,
                                                  min_reduction=min_reduction,
                                                  max_reduction=max_reduction)
        elif task == 'child_id':
            self.streams = self.init_stream(threshold=threshold,
                                            confines=confines,
                                            cutoff=cutoff,
                                            CI_exclude=CI_exclude,
                                            oldcam=oldcam,
                                            min_reduction=min_reduction,
                                            max_reduction=max_reduction,
                                            kids_cut=kids_cut,
                                            unbiased=unbiased)
        elif task == 'prelex_id':
            self.folder = 'data/annotated'
            self.streams = self.init_stream(task='prelex_id')

        if pad == 'max':
            self.pad_length = self.max_length
        elif pad == 'mean':
            self.pad_length = self.mean_length
        elif pad == 'interval':
            self.pad_length = int(self.mean_length + width * self.stddev)

        self.encoder = LabelEncoder()
        if threshold or confines:
            self.encoder.fit([l for _, l in self.streams['young_train']])
        else:
            self.encoder.fit([l for _, l in self.streams['train']])

        print(self.encoder)
        print('-> working on classes:', self.encoder.classes_)

    def init_stream(self, task='child_id', threshold=None, confines=None,
                    cutoff=None, CI_exclude=False, oldcam=False,
                    min_reduction=None, max_reduction=None,
                    kids_cut=None,
                    unbiased=False):

        meta = pd.read_excel(self.folder + '/meta_wlc.xlsx')
        meta = self.limit_meta(meta=meta, cutoff=cutoff, oldcam=oldcam, CI_exclude=CI_exclude,
                               min_reduction=min_reduction, max_reduction=max_reduction,
                               kids_cut=kids_cut)
        print(set(meta['child']))
        print('length of cutoff: ', len(meta.index))

        print('meta length: ', len(meta.index))

        if task == 'prelex_id':
            data = {'train': list(zip(train_meta['filename'], train_meta['category'])),
                    'dev': list(zip(dev_meta['filename'], dev_meta['category'])),
                    'test': list(zip(test_meta['filename'], test_meta['category']))}

        elif task == 'child_id':
            if threshold:
                data = self.threshold_divide(meta=meta,
                                             threshold=threshold,
                                             cutoff=cutoff)
            elif confines:
                data = self.compartments(meta=meta,
                                         confines=confines,
                                         cutoff=cutoff)
            elif unbiased:
                data = self.unbiased_split(meta=meta,
                                           cutoff=cutoff)
            else:
                print('TYPE META: ', type(meta))
                train_meta, dev_meta, test_meta = utils.metadata_split(meta)
                data = {'train': train_meta, 'dev': dev_meta, 'test': test_meta}
                self.init_settings(data=data)
                data = {i: list(zip(data[i]['filename'], data[i]['child'])) for i in data}

        # print('len data: ', len(data['train']))
        return data

    def limit_meta(self, meta,
                   cutoff, CI_exclude, oldcam,
                   min_reduction, max_reduction,
                   kids_cut=None, kids_list=None,
                   threshold=None):

        if CI_exclude == True:
            print(meta.columns)
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
            meta = meta.loc[meta['child'] in kids_list]
        if below_threshold:
            meta = meta.loc[meta['age'] < threshold]
        elif over_threshold:
            meta = meta.loc[meta['age'] > threshold]

        meta_fin = pd.DataFrame(columns=meta.columns)
        # if kids_cut and (kids_cut < len(set(meta['child']))):
        #     kids_set = random.sample(set(meta['child']), kids_cut)
        # else:
        #     kids_set = set(meta['child'])
        if cutoff:
            for child in set(meta['child']):
                if len(meta[meta['child'] == child]) < cutoff:
                    continue
                else:
                    meta_fin = pd.concat([meta_fin,
                                          meta.loc[meta['child'] == child]])
            tot_kids = set(meta_fin['child'])
            if kids_cut and (kids_cut < len(tot_kids)):
                for ckid in random.sample(tot_kids, len(tot_kids) - kids_cut):
                    meta_fin = meta_fin[meta_fin['child'] != ckid]

        else:
            meta_fin = meta
        print(meta_fin.columns)
        return meta_fin

    def init_stream_folds(self, train_names, dev_names, test_names, oldcam=False,
                          min_reduction=None, max_reduction=None):

        meta = pd.read_excel(self.folder + '/meta_wlc.xlsx')
        if not oldcam:
            meta = meta.loc[meta['camera'] == 0]
        if min_reduction:
            meta = meta.loc[meta['length'] > min_reduction]
        if max_reduction:
            meta = meta.loc[meta['length'] < max_reduction]

        train_meta = meta[meta['child'].isin(train_names)]
        train_meta.sample(frac=1)  # shuffles the rows

        dev_meta = meta[meta['child'].isin(dev_names)]
        dev_meta.sample(frac=1)  # shuffles the rows

        test_meta = meta[meta['child'].isin(test_names)]
        test_meta.sample(frac=1)  # shuffles the rows

        data = {'train': train_meta, 'dev': dev_meta, 'test': test_meta}
        self.init_settings(data=data)
        data = {i: list(zip(data[i]['filename'], data[i]['NH/CI'])) for i in data}

        return data

    def init_settings(self, data):

        self.mean_length = int(np.mean([data[i]['length'].mean() for i in data]))
        self.max_length = np.max([data[i]['length'].max() for i in data])
        self.stddev = np.mean([data[i]['length'].std() for i in data])

    def threshold_divide(self, meta, threshold, cutoff=1000):

        children = set(meta['child'])
        young_children_meta = pd.DataFrame()
        old_children_meta = pd.DataFrame()
        for child in children:
            young_child_list = meta.loc[meta['child'] == child]
            young_child_list = young_child_list.loc[young_child_list['age'] < threshold]
            old_child_list = meta.loc[meta['child'] == child]
            old_child_list = old_child_list.loc[old_child_list['age'] > threshold]
            if len(young_child_list.index) < cutoff or len(old_child_list.index) < cutoff:
                continue
            else:
                old_child_list = old_child_list.sample(cutoff, random_state=123)
                young_child_list = young_child_list.sample(cutoff, random_state=123)

            young_children_meta = pd.concat([young_children_meta, young_child_list])
            old_children_meta = pd.concat([old_children_meta, old_child_list])
        young_children_meta = young_children_meta.sample(frac=1)
        old_children_meta = old_children_meta.sample(frac=1)

        young_train_meta, young_dev_meta, young_test_meta = \
            utils.metadata_split(young_children_meta)
        old_train_meta, old_dev_meta, old_test_meta = \
            utils.metadata_split(old_children_meta)

        data = {'young_train': young_train_meta, 'young_dev': young_dev_meta, 'young_test': young_test_meta,
                'old_train': old_train_meta, 'old_dev': old_dev_meta, 'old_test': old_test_meta}
        self.init_settings(data=data)
        data = {i: list(zip(data[i]['filename'], data[i]['child'])) for i in data}
        return data

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

    def compartments(self, meta, confines, cutoff=None):

        if len(confines)==2:
            confines = [0, confines[0], confines[1], inf]
        children = set(meta['child'])
        print(confines)
        young_children_meta = pd.DataFrame()
        median_children_meta = pd.DataFrame()
        old_children_meta = pd.DataFrame()

        for child in children:
            young_child_list = meta[(meta['child'] == child) & (confines[0] <= meta['age']) & (meta['age'] < confines[1])]
            median_child_list = meta[(meta['child'] == child) & (confines[1] <= meta['age']) & (meta['age'] < confines[2])]
            old_child_list = meta[(meta['child'] == child) & (confines[2] <= meta['age']) & (meta['age'] < confines[3])]
            # young_child_list = young_child_list.loc[(young_child_list['age'] >= confines[0]) & (young_child_list['age'] < confines[1])]
            # median_child_list = meta.loc[meta['child'] == child]
            # median_child_list = median_child_list.loc[(median_child_list['age'] >= confines[1]) & (median_child_list['age'] < confines[2])]
            # old_child_list = meta.loc[meta['child'] == child]
            # old_child_list = old_child_list.loc[(old_child_list['age'] >= confines[2]) & (old_child_list['age'] <= confines[3])]
            if len(young_child_list.index) < cutoff or \
               len(median_child_list.index) < cutoff or \
               len(old_child_list.index) < cutoff:
                continue
            else:
                old_child_list = old_child_list.sample(cutoff, random_state=123)
                median_child_list = median_child_list.sample(cutoff, random_state=123)
                young_child_list = young_child_list.sample(cutoff, random_state=123)

            young_children_meta = pd.concat([young_children_meta, young_child_list])
            median_children_meta = pd.concat([median_children_meta, median_child_list])
            old_children_meta = pd.concat([old_children_meta, old_child_list])
        young_children_meta = young_children_meta.sample(frac=1)
        median_children_meta = median_children_meta.sample(frac=1)
        old_children_meta = old_children_meta.sample(frac=1)

        young_train_meta, young_dev_meta, young_test_meta = \
            utils.metadata_split(young_children_meta)
        median_train_meta, median_dev_meta, median_test_meta = \
            utils.metadata_split(median_children_meta)
        old_train_meta, old_dev_meta, old_test_meta = \
            utils.metadata_split(old_children_meta)
        data = {'young_train': young_train_meta, 'young_dev': young_dev_meta, 'young_test': young_test_meta,
                'median_train': median_train_meta, 'median_dev': median_dev_meta, 'median_test': median_test_meta,
                'old_train': old_train_meta, 'old_dev': old_dev_meta, 'old_test': old_test_meta}
        self.init_settings(data=data)
        data = {i: list(zip(data[i]['filename'], data[i]['child'])) for i in data}

        return data

    def get_batches(self, stream,  fitting=False,
                    endless=False, sample_size=None, shuffle=False):
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
                        if shuffle:
                            X = [random.shuffle(x) for x in X]
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
