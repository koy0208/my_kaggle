import copy
import xgboost as xgb
from sklearn.model_selection import train_test_split
from model_base import Model


class ModelXGB(Model):
    def train(self, x, y, params, va_x=None, va_y=None):
        # データのセット
        validation = va_x is not None
        dtrain = xgb.DMatrix(x, label=y)
        if validation:
            dvalid = xgb.DMatrix(va_x, label=va_y)
        else:
            tmp_x, tmp_va_x, tmp_y, tmp_va_y = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)
            # dtrain = lgb.Dataset(x,label=y,weight=weight)
            tmp_dvalid = xgb.DMatrix(tmp_va_x, label=tmp_va_y)
            tmp_dtrain = xgb.DMatrix(tmp_x, label=tmp_y)

        # ハイパーパラメータの設定

        params_c = copy.deepcopy(params)
        num_round = params_c.pop("num_round")
        early_stopping_rounds = params_c.pop("early_stopping_rounds")

        # 学習
        if validation:
            watchlist = [(dtrain, "train"), (dvalid, "eval")]
            self.model = xgb.train(
                params_c,
                dtrain,
                num_round,
                evals=watchlist,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=1000,
            )
        else:
            watchlist = [(tmp_dtrain, "train"), (tmp_dvalid, "eval")]
            pre_model = xgb.train(
                params_c,
                dtrain,
                num_round,
                evals=watchlist,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=1000,
            )
            # validationでのbest_paramを使う
            watchlist = [(dtrain, "train")]
            num_round = pre_model.best_iteration
            self.model = xgb.train(params_c, dtrain, num_round, evals=watchlist, verbose_eval=1000)

    def predict(self, te_x):
        dtest = xgb.DMatrix(te_x)
        return self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)
