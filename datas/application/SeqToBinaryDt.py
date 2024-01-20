from mlops.datas.BaseSeqToClassDt import BaseSeqToClassDt
from sklearn.preprocessing import MinMaxScaler
import numpy as np



class SeqToBinaryDt(BaseSeqToClassDt):
    '''
    主要特征：将时间序列预测问题变为二分类问题
    '''
    def __init__(self, y_threshold=10, **kwargs):
        '''
        # y_threshold: 将序列 y 转换为 class 的阈值
        '''
        super().__init__(**kwargs)
        self.y_threshold = y_threshold


    def categoric_engineering(self, raw_data, split_name='train'):
        '''
        将序列汇聚成类别变量, 聚合函数: np.amax, np.amin, np.median, ...;
        '''
        X_arr_list = super().categoric_engineering(raw_data, split_name)

        # X_list = [np.median(MinMaxScaler().fit_transform(X), axis=1).reshape(1, -1) for X in X_arr_list]
        X_list = [np.median(X, axis=1).reshape(1, -1) for X in X_arr_list]
        X_cate = np.concatenate(X_list, axis=0)
        return X_cate


    def feature_engineering(self, raw_data, split_name='train'):
        if self.preprocess_func is not None:
            raw_data = self.preprocess_func(raw_data)

        # base dt class return base X, y datas
        X_arr_list, y_list = super().feature_engineering(raw_data, split_name)

        # 将 X  concatenate & reshape
        X_arr_list = [MinMaxScaler().fit_transform(X).reshape(1, -1) for X in X_arr_list]
        X = np.concatenate(X_arr_list, axis=0)

        # 如果有分类变量，则一起concat
        if self.categoric_features:
            X_cate = self.categoric_engineering(raw_data, split_name=split_name)
            X = np.concatenate([X, X_cate], axis=1)

        if split_name == 'train':
            y = np.array([1 if np.max(y) >= self.y_threshold / 100 else 0 for y in y_list])
            return X, y
        else:
            return X