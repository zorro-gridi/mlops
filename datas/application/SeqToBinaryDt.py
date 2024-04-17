from mlops.datas.BaseSeqToClassDt import BaseSeqToClassDt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging


class SeqToBinaryDt(BaseSeqToClassDt):
    '''
    Desc:
        主要特征：将时间序列预测问题变为二分类问题
    '''
    def __init__(self, y_threshold=10, **kwargs):
        '''
        Args:
            y_threshold: 将序列 y 转换为 class 的阈值
        '''
        super().__init__(**kwargs)
        self.y_threshold = y_threshold


    def categoric_engineering(self, raw_data, split_name='train'):
        '''
        Desc:
            将部分 feature 的序列汇聚成 categoric 类别变量
            可选的聚合函数: ["np.amax", "np.amin", "np.median", ...], amax, amin 表示沿数轴的最值
        TODO:
            1. 此处的聚合函数需要支持自定义
        '''
        X_arr_list = super().categoric_engineering(raw_data, split_name)

        # 原始写法过死，不够灵活，子类无法修改
        # # X_list = [np.median(MinMaxScaler().fit_transform(X), axis=1).reshape(1, -1) for X in X_arr_list]
        # X_list = [np.median(X, axis=1).reshape(1, -1) for X in X_arr_list]
        # X_cate = np.concatenate(X_list, axis=0)

        # 使用类方法单独定义分类变量的处理逻辑，方便子类继承重写逻辑
        X_cate = self.categoric_processing(X_arr_list)
        return X_cate


    def categoric_processing(self, cate_arr_list):
        '''
        Desc:
            categoric_engineering 方法的 postprocessing 方法，对数据进行后处理。
            同样可以通过继承改写该方法
        '''
        # X_list = [np.median(MinMaxScaler().fit_transform(X), axis=1).reshape(1, -1) for X in X_arr_list]
        X_list = [np.median(X, axis=1).reshape(1, -1) for X in cate_arr_list]
        X_cate = np.concatenate(X_list, axis=0)
        return X_cate


    def feature_engineering(self, raw_data, split_name='train'):
        '''
        Desc:
            特征预处理函数
        Args:
            raw_data: 原始数据
            split_name: 返回的数据集的名称
        '''
        # 如果预处理函数不为空，则先对原始数据进行数据预处理
        if self.preprocess_func is not None:
            raw_data = self.preprocess_func(raw_data)

        # base dt class return base X, y datas
        X_arr_list, y_list = super().feature_engineering(raw_data, split_name)

        # # 将 X  concatenate & reshape
        # X_arr_list = [MinMaxScaler().fit_transform(X).reshape(1, -1) for X in X_arr_list]
        # X = np.concatenate(X_arr_list, axis=0)
        # y = np.array([1 if np.max(y) >= self.y_threshold / 100 else 0 for y in y_list])

        # 使用单独定义 seq2class_fn 类方法，方便子类继承重写
        X, y = self.seq2class_fn(X_arr_list, y_list)

        # 如果有分类变量，则一起concat
        if self.categoric_features:
            X_cate = self.categoric_engineering(raw_data, split_name=split_name)
            # 将汇聚后形成的分类变量按列 stack
            X = np.concatenate([X, X_cate], axis=1)

        # 将目标连续变量变为离散的分类变量
        if split_name == 'train':
            return X, y
        else:
            return X, None


    def seq2class_fn(self, X_list, y_list):
        '''
        Desc:
            定义 seq2class 的具体逻辑。该方法可以通过继承重写
        Guide: Important !!!
            如果需要针对数据集进行定制化的处理，可以继续继承该类，修改对应的类方法
        Args:
            X_list, y_list: self.feature_engineering 返回的结果
        '''
        # 将 X  concatenate & reshape
        X_arr_list = [MinMaxScaler().fit_transform(X).reshape(1, -1) for X in X_list]
        X = np.concatenate(X_arr_list, axis=0)

        if y_list:
            y = np.array([1 if np.max(y) >= self.y_threshold / 100 else 0 for y in y_list])
        else:
            y = None
        return X, y