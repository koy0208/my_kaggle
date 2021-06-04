import copy
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from model_base import Model


class ModelRFRegressor(Model):
    def train(self, x, y, params, va_x=None, va_y=None):
        # ハイパーパラメータの設定
        self.model = RandomForestRegressor(**params)
        self.model.fit(x, y)

    def predict(self, te_x):
        return self.model.predict(te_x)


class ModelRFClassifier(Model):
    def train(self, x, y, params, va_x=None, va_y=None):
        # ハイパーパラメータの設定
        self.model = RandomForestRegressor(**params)
        self.model.fit(x, y)

    def predict(self, te_x):
        return self.model.predict(te_x)