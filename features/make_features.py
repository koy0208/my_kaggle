import pandas as pd
import numpy as np
from base import Feature, get_arguments, generate_features, generate_features_all




class bmi(Feature):
    def create_features(self):
        self.train['bmi'] = train["bw"]/np.square((train["hi"]/100))
        #self.test['bmi'] =  test["bw"]/np.square((test["hi"]/100))


def col_LHrate(ldl, hdl):
        try:
            LHrate = ldl / hdl
            return LHrate
        except ZeroDivisionError:
            pass
        
class LHrate(Feature):
    def create_features(self):
        self.train["LHrate"] = list(map(col_LHrate,train["ldl"],train["hdl"]))
        #self.test['LHrate'] =  list(map(col_LHrate,test["ldl"],test["hdl"]))


if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_csv("../data/input/train.csv")
    #test = pd.read_csv("../data/input/test.csv")

    #generate_features(train.columns, args.force)
    generate_features(globals(), args.force)

    #テストなし
    generate_features_all(train,None)
    #テストあり
    #generate_features_all(train,test)