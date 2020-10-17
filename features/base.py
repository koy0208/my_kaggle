import re
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import contextmanager

import pandas as pd

import argparse
import inspect

@contextmanager
def timer(name):
    t0 = time.time()
    print('[{}] start'.format(name))
    yield
    print("[{}] done in {:.0f} s".format(name,time.time() - t0))


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / '{}_train.ftr'.format(self.name)
        #self.test_path = Path(self.dir) / '{}_test.ftr'.format(self.name)
    
    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            #self.test.columns = prefix + self.test.columns + suffix
        return self
    
    @abstractmethod
    def create_features(self):
        raise NotImplementedError
    
    def save(self):
        self.train.to_feather(str(self.train_path))
        #self.test.to_feather(str(self.test_path))

def generate_features_all(train,test=None,dir=None):
    dir = Feature.dir
    for col in train.columns:
        train[[col]].to_feather('{}/{}_train.ftr'.format(dir,col))
        #test[[col]].to_feather('{}/{}_test.ftr'.format(dir,col))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    return parser.parse_args()


def get_features(namespace):
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()



def generate_features(namespace, overwrite):
    for f in get_features(namespace):
        #テストなし
        if f.train_path.exists() and not overwrite:
        #テストあり
        #if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()


