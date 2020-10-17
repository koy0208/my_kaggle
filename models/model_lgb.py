import lightgbm as lgb

import copy
from .util import Util
from .model_base import Model


class ModelLGB(Model):
    
    def train(self, x, y, va_x=None, va_y=None, params=None,weight=None):

        # データのセット
        validation = va_x is not None
        
        dtrain = lgb.Dataset(x,label=y,weight=weight)
        if validation:
            dvalid = lgb.Dataset(va_x, label=va_y)

        # ハイパーパラメータの設定
        
        params_c = copy.deepcopy(params)
        num_round = params_c.pop('num_round')
        early_stopping_rounds = params_c.pop('early_stopping_rounds')
       

        # 学習
        if validation:
            #early_stopping_rounds = params.pop('early_stopping_rounds')
    
            self.model = lgb.train(params_c,train_set=dtrain,
            num_boost_round=num_round,
            valid_sets=dvalid,
            early_stopping_rounds=early_stopping_rounds,verbose_eval=1000)
        else:
            self.model = lgb.train(params_c, dtrain, num_round)

    def predict(self, te_x):
        dtest = te_x
        return self.model.predict(dtest, num_iteration=self.model.best_iteration)
        

    