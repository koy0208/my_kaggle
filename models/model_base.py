import os

import numpy as np
import pandas as pd
import copy
from .util import Util
from sklearn.metrics import log_loss, recall_score
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from typing import Callable, List, Optional, Tuple, Union
#import imblearn


class Model():
    def load_model(self,load_name):
        """モデルの読み込みを行う
        load_nameにモデル名を渡す
        """
        model_path = os.path.join('./model_learned', '{}.model'.format(load_name))
        self.model = Util.load(model_path)

    def save_model(self,save_name):
        model_path = os.path.join('./model_learned', '{}.model'.format(save_name))
        Util.dump(self.model, model_path)
        
    def train_test(self, x, y,params = None,weight=None,test_size=0.2,stratified=False,run_name = "model_split"):

        print('{} - start training cv'.format(run_name))

        x = pd.DataFrame(x)
        y = pd.Series(y)

        tr_x, va_x, tr_y, va_y = train_test_split(x, y, test_size=test_size)
        

        tr_idx = tr_x.index
        va_idx = va_x.index

        self.train(tr_x,tr_y,va_x,va_y,params,weight)
        va_pred = self.predict(va_x)
        score = recall_score(va_y, np.round(va_pred))

        self.save_model("{}".format(run_name))
        
        print("{} end training cv".format(run_name))
        print('score {}'.format(score))


        return tr_idx, va_idx, va_pred




    def train_cv(self, x, y,te_x=None,te_y=None,params = None,weight=None,n_fold=4,stratified=False,run_name = "model_cv"):
        """クロスバリデーションでの学習・評価を行う

        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        print('{} - start training cv'.format(run_name))


        if stratified:
            dummy_x = np.zeros(len(y))
            skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=71)
        else:
            kf = KFold(n_splits=n_fold, shuffle=True, random_state=71)
        
        scores = []
        va_idxes = []
        va_preds = []

        te_preds = []

        # 各foldで学習を行う
        for i_fold in range(n_fold):
            print(" ")
            print('{} fold {} - start training'.format(run_name,i_fold))

            if stratified:
                tr_idx, va_idx = list(skf.split(dummy_x,y))[i_fold]
            else:
                tr_idx, va_idx = list(kf.split(x))[i_fold]

            

            #cvindexを代入
            x = pd.DataFrame(x)
            y = pd.Series(y)

            tr_x, tr_y = x.iloc[tr_idx], y.iloc[tr_idx]
            va_x, va_y = x.iloc[va_idx], y.iloc[va_idx]


            # 学習を行う
            self.train(x,y,va_x,va_y,params,weight)
            va_pred = self.predict(va_x)
            score = recall_score(va_y, np.round(va_pred))

            # モデルを保存する
            self.save_model("{}_{}".format(run_name,i_fold))

            # 結果を保持する
            va_idxes.append(va_idx)
            scores.append(score)
            va_preds.append(va_pred)

            print('{} fold {} - end training - score {}'.format(run_name,i_fold,score))

            if te_x is not None:
                te_pred = self.predict(te_x)
                te_preds.append(te_pred)


        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        va_preds = np.concatenate(va_preds, axis=0)
        va_preds = va_preds[order]
    

        print(" ")
        print("{} end training cv".format(run_name))
        print('score {}'.format(np.mean(scores)))

        return va_preds,  np.mean(te_preds,axis=0)







