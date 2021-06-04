import copy
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from model_base import Model


class ModelLGB(Model):
    def train(self, x, y, params, va_x=None, va_y=None):
        # データのセット
        validation = va_x is not None

        dtrain = lgb.Dataset(x, label=y)
        if validation:
            dvalid = lgb.Dataset(va_x, label=va_y)
        else:
            tmp_x, tmp_va_x, tmp_y, tmp_va_y = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)
            tmp_dvalid = lgb.Dataset(tmp_va_x, label=tmp_va_y)
            tmp_dtrain = lgb.Dataset(tmp_x, label=tmp_y)

        # ハイパーパラメータの設定

        params_c = copy.deepcopy(params)
        num_round = params_c.pop("num_round")
        early_stopping_rounds = params_c.pop("early_stopping_rounds")

        # 学習
        if validation:
            self.model = lgb.train(
                params_c,
                train_set=dtrain,
                num_boost_round=num_round,
                valid_sets=dvalid,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=1000,
            )
        else:
            pre_model = lgb.train(
                params_c,
                train_set=tmp_dtrain,
                num_boost_round=num_round,
                valid_sets=tmp_dvalid,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=1000,
            )
            # validationでのbest_paramを使う
            num_round = pre_model.best_iteration
            self.model = lgb.train(params_c, train_set=dtrain, num_boost_round=num_round)

    def predict(self, te_x):
        return self.model.predict(te_x)
