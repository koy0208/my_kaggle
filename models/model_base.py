import os
import pickle


class Model:
    def load_model(self, load_name):
        """モデルの読み込みを行う
        Args:
            load_name ([type]): [description]
        """
        model_path = os.path.join("../model_learned", f"{load_name}.pickle")
        self.model = pickle.load(open(model_path, "rb"))

    def save_model(self, save_name):
        """モデルの保存を行う
        Args:
            save_name ([type]): [description]
        """
        model_path = os.path.join("../model_learned", f"{save_name}.pickle")
        pickle.dump(self.model, open(model_path, "wb"))
