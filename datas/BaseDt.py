from abc import ABCMeta, abstractclassmethod
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    )

import xgboost as xgb
import numpy as np
from tqdm import tqdm
import logging

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    )
import torch
from sklearn.preprocessing import MinMaxScaler
from functools import partial



class AbstractDatasetFactory(metaclass=ABCMeta):
    '''
    基础数据集抽象工厂方法
    '''
    def __init__(self, features=None, categoric_features=None, target=None, preprocess_func=None):
        '''
        Args:
            features: 数据默认输入特征
            target: str or int
                when is str: 表示目标变量名称
                when is int: 目标变量的索引值; [= 1: 表示单步预测; > 1: 表示多步预测]
            preprocess_func: 数据进行特征工程之前的预处理函数
            categoric_features: 数据中的分类特征
        '''
        self.features = features
        self.target = target
        self.preprocess_func = preprocess_func
        self.is_imbalanced = True
        self.categoric_features = categoric_features


    @abstractclassmethod
    def feature_engineering(self):
        pass


    def set_attr(self, inst_config):
        new_config = {k: v for k, v in inst_config.items() if k in self.__dict__.keys()}
        self.__dict__.update(new_config)


    def data_split(self, X, y, test_size=0.2, random_state=42):
        if self.is_imbalanced:
            logging.warning(f'the dataset is set imbalanced! ')
            # 因为数据集的标签不均衡，所以使用 StratifiedShuffleSplit 分层抽样
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            for i, (train_idx, test_idx) in enumerate(sss.split(X, y)):
                x_train, y_train = X[train_idx], y[train_idx]
                x_test, y_test = X[test_idx], y[test_idx]

            assert len(X) == len(x_train) + len(x_test)
            train_pos_ratio = sum(y_train) / len(y_train)
            test_pos_ratio = sum(y_test) / len(y_test)

            logging.warning(f'train 训练集数量: {len(x_train)}, 正样本比例: {train_pos_ratio:,.3f}')
            logging.warning(f'test  测试集数量: {len(x_test)}, 正样本比例: {test_pos_ratio:,.3f}')

        else:
            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test


    def load_test_data(self, data, inst_config=None):
        '''
        # inst_config: 类实例化的参数;可为历史模型参数, 或new model 的参数
        '''
        if inst_config is not None:
            self.set_attr(inst_config)

        model_datas = self.feature_engineering(data)
        # 同时传入 X, y
        if isinstance(model_datas, tuple):
            x_train, x_test, y_train, y_test = self.data_split(*model_datas)
            test_data = (x_test, y_test)
        # 只传入 X
        else:
            train_data, test_data = self.data_split(model_datas)
        return test_data
