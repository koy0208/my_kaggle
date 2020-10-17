import xgboost as xgb
import copy
from .util import Util
from .model_base import Model

class ModelXGB(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None,params=None,weight=None):

        # データのセット
        validation = va_x is not None
        dtrain = xgb.DMatrix(tr_x, label=tr_y,weight = weight)
        if validation:
            dvalid = xgb.DMatrix(va_x, label=va_y)

        # ハイパーパラメータの設定

        params_c = copy.deepcopy(params)
        num_round = params_c.pop('num_round')
        early_stopping_rounds = params_c.pop('early_stopping_rounds')

        # 学習
        if validation:
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            self.model = xgb.train(params_c, dtrain, num_round, evals=watchlist,
                                   early_stopping_rounds=early_stopping_rounds,
                                   verbose_eval=1000)
        else:
            watchlist = [(dtrain, 'train')]
            self.model = xgb.train(params_c, dtrain, num_round, evals=watchlist,verbose_eval=1000)

    def predict(self, te_x):
        dtest = xgb.DMatrix(te_x)
        return self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)