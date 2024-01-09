from mlops.datas.BaseSeqToClassDt import BaseSeqToClassDt
from sklearn.preprocessing import MinMaxScaler
import numpy as np



class VolumeBreakOutClassDt(BaseSeqToClassDt):
    def __init__(self, y_threshold=10, **kwargs):
        super().__init__(**kwargs)
        self.y_threshold = y_threshold


    def categoric_engineering(self, raw_data, split_name='train'):
        X_arr_list = super().categoric_engineering(raw_data, split_name)

        # X_list = [np.median(MinMaxScaler().fit_transform(X), axis=1).reshape(1, -1) for X in X_arr_list]
        X_list = [np.median(X, axis=1).reshape(1, -1) for X in X_arr_list]
        X_cate = np.concatenate(X_list, axis=0)
        return X_cate


    def feature_engineering(self, raw_data, split_name='train'):
        X_arr_list, y_list = super().feature_engineering(raw_data, split_name)

        X_arr_list = [MinMaxScaler().fit_transform(X).reshape(1, -1) for X in X_arr_list]
        X = np.concatenate(X_arr_list, axis=0)

        if self.categoric_engineering is not None:
            X_cate = self.categoric_engineering(raw_data, split_name=split_name)
            X = np.concatenate([X, X_cate], axis=1)

        if split_name == 'train':
            y = np.array([1 if np.max(y) >= self.y_threshold / 100 else 0 for y in y_list])
            return X, y
        else:
            return X