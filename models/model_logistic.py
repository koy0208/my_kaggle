import copy
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from model_base import Model


class ModelLogistic(Model):
    def train(self, x, y, params, va_x=None, va_y=None):
        # ハイパーパラメータの設定
        self.model = LogisticRegression(**params)
        self.model.fit(x, y)

    def predict(self, te_x):
        return self.model.predict_proba(te_x)[:, 1]
