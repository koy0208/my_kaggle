from sklearn.linear_model import LogisticRegression

import copy
from .util import Util
from .model_base import Model


class ModelLogistic(Model):
    def __init__(self):
        self.model  = LogisticRegression()

    
    def train(self, x, y, va_x=None, va_y=None, params=None,weight=None):
        # インスタンス
        self.model.fit(x, y)

    def predict(self, te_x):
        dtest = te_x
        return self.model.predict(te_x)