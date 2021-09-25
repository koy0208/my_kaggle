import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import multiprocessing

cores = multiprocessing.cpu_count()
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

import sklearn.neighbors._base

sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base
from missingpy import MissForest


# ターゲットエンコーディング
def tg_encoding(pre_tr_x, pre_te_x, tr_y, group_cols):
    tr_x = pre_tr_x.copy()
    te_x = pre_te_x.copy()
    df = pd.concat([tr_x, tr_y], axis=1)
    outcome = pd.DataFrame(tr_y).columns[0]
    for g_col in group_cols:
        group_df = pd.DataFrame(df.groupby(g_col)[outcome].mean())
        group_df.columns = [f"tge_{g_col}_{outcome}"]
        tr_x = tr_x.join(group_df, on=g_col)
        te_x = te_x.join(group_df, on=g_col)
    return tr_x, te_x


# 集団の平均値
def tg_group_agg(pre_tr_x, pre_te_x, target_col, group_cols, how="mean"):
    tr_x = pre_tr_x.copy()
    te_x = pre_te_x.copy()
    if how == "mean":
        group_df = tr_x.groupby(group_cols)[target_col].mean()
    elif how == "mean":
        group_df = tr_x.groupby(group_cols)[target_col].median()
    group_df.columns = [f"g_{c}_{how}" for c in target_col]
    tr_x = tr_x.join(group_df, on=group_cols)
    te_x = te_x.join(group_df, on=group_cols)
    return tr_x, te_x


# カテゴリから数値への変換
def cat_to_num(pre_tr_x, pre_te_x=False, target_col=False, how="one_hot", save_name=False, model=False):
    if model:
        tr_x = pre_tr_x.reset_index(drop=True).copy()
        if how == "one_hot":
            tr_en = model.transform(tr_x[target_col])
            # 列名を取得
            label = model.get_feature_names(target_col)
            # データフレーム化
            tr_en = pd.DataFrame(tr_en, columns=label)
            # データフレームを結合
            tr_x = pd.concat([tr_x, tr_en], axis=1)
        return tr_x.drop(columns=target_col, axis=1)
    else:
        tr_x = pre_tr_x.reset_index(drop=True).copy()
        te_x = pre_te_x.reset_index(drop=True).copy()
        if how == "one_hot":
            ohe = OneHotEncoder(sparse=False, handle_unknown="error", drop="first")
            tr_en = ohe.fit_transform(tr_x[target_col])
            te_en = ohe.transform(te_x[target_col])
            # 列名を取得
            label = ohe.get_feature_names(target_col)
            # データフレーム化
            tr_en = pd.DataFrame(tr_en, columns=label)
            te_en = pd.DataFrame(te_en, columns=label)
            # データフレームを結合
            tr_x = pd.concat([tr_x, tr_en], axis=1)
            te_x = pd.concat([te_x, te_en], axis=1)
        if save_name:
            pickle.dump(ohe, open(save_name, "wb"))
        return tr_x.drop(columns=target_col, axis=1), te_x.drop(columns=target_col, axis=1)


# 欠損値補完
def super_fillna(pre_tr_x, pre_te_x, target_col, how="mean", save_name=False):
    tr_x = pre_tr_x.copy()
    te_x = pre_te_x.copy()
    if how == "mean":
        fill_value = tr_x[target_col].mean()
        tr_x.fillna(fill_value, inplace=True)
        te_x.fillna(fill_value, inplace=True)
    elif how == "median":
        fill_value = tr_x[target_col].median()
        tr_x.fillna(fill_value, inplace=True)
        te_x.fillna(fill_value, inplace=True)
    elif how == "rf":
        fill_value = MissForest()
        tr_x[target_col] = fill_value.fit_transform(tr_x[target_col])
        te_x[target_col] = fill_value.transform(te_x[target_col])
    if save_name:
        pickle.dump(fill_value, open(save_name, "wb"))
    return tr_x, te_x


def standerize(pre_tr_x, pre_te_x, target_col, save_name=False):
    tr_x = pre_tr_x.copy()
    te_x = pre_te_x.copy()
    sc = StandardScaler()
    tr_x[target_col] = sc.fit_transform(tr_x[target_col])
    te_x[target_col] = sc.transform(te_x[target_col])
    if save_name:
        pickle.dump(sc, open(save_name, "wb"))
    return tr_x, te_x


def eval_scores(true_y, pred_y, probability=True, th=0.5):
    """性能評価指標をまとめて返す。

    Args:
        true_y ([type]): [description]
        pred_y ([type]): [description]
        probability (bool, optional): 予測値は確率か0,1か. Defaults to True.

    Returns:
        [type]: [description]
    """
    pred = np.where(pred_y > th, 1, 0)
    conf_mat = confusion_matrix(true_y, pred)
    precision = precision_score(true_y, pred) * 100
    recall = recall_score(true_y, pred) * 100
    f1 = f1_score(true_y, pred) * 100
    print(conf_mat)
    print(f"precision(陽性的中率): {precision:.1f}%")
    print(f"recall(感度): {recall:.1f}%")
    print(f"f1: {f1:.1f}%")
    if probability:
        roc_auc = roc_auc_score(true_y, pred_y)
        print(f"roc_auc {roc_auc}")
        scores = {"precision": precision, "recall": recall, "f1": f1, "auc": roc_auc}
    else:
        scores = {"precision": precision, "recall": recall, "f1": f1}
    return scores


def train_cv(x, y, model, params, te_x=None, te_y=None, n_fold=4, stratified=True, score="auc"):
    """クロスバリデーションでの学習・評価を行う

    学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
    """
    print("{} - start training cv".format(run_name))

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
        print(f"fold {i_fold} - start training")
        tr_idx, va_idx = list(skf.split(dummy_x, y))[i_fold]

        # cvindexを代入
        x = pd.DataFrame(x)
        y = pd.Series(y)

        tr_x, tr_y = x.iloc[tr_idx], y.iloc[tr_idx]
        va_x, va_y = x.iloc[va_idx], y.iloc[va_idx]

        # 学習を行う
        model.train(tr_x, tr_y, params, va_x, va_y)

        # スコアを保存する
        score.append(eval_scores(va_y, model.predict(va_x)))

    # 各foldの結果をまとめる
    va_idxes = np.concatenate(va_idxes)
    order = np.argsort(va_idxes)
    va_preds = np.concatenate(va_preds, axis=0)
    va_preds = va_preds[order]

    print(" ")
    print(f"end training cv")
    print(f"score {np.mean(scores)}")

    return va_preds, np.mean(te_preds, axis=0)


# ハイパラメーターチューニング
def tuningOfRF(X, y):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 500)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 1, 1000)
        max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])
        clf = RandomForestClassifier(
            random_state=2021, n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, n_jobs=cores // 2
        )

        # ランダムフォレストモデルを交差検証してその平均スコア（今回はAccuracy）を返す
        return cross_val_score(clf, X, y, n_jobs=cores // 2, cv=4, scoring="roc_auc").mean()

    return objective