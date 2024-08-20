from abc import ABCMeta, abstractclassmethod
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    )
from typing import List, Union
from copy import copy
import pandas as pd

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

StrOrInt = Union[str, int]
StrOrIntList = List[StrOrInt]


class AbstractDatasetFactory(metaclass=ABCMeta):
    '''
    基础数据集抽象工厂方法
    '''
    def __init__(self,
        features=None,
        categoric_features=None,
        target=None,
        preprocess_func=None,
        is_imbalanced=False,
        **kwargs,
        ):
        '''
        Args:
            features: StrOrIntList, 数据的输入部特征列表
            target: str or int, 目标变量的索引或者名称
                when is str: 表示目标变量名称
                when is int: 目标变量的索引值; [= 1: 表示单步预测; > 1: 表示多步预测]
            categoric_features: StrOrIntList, 指定数据中的分类特征列表
                when is str: 表示特征名称列表
                when is int: 表示特征名在 dataframe columns 中的索引
        '''
        self.features = features
        self.target = target
        self.is_imbalanced = is_imbalanced # 默认为 False, 因为分层抽样会打乱数据集
        self.categoric_features = categoric_features
        self.preprocess_func = preprocess_func
        self.target = self.target if isinstance(self.target, int) else self.features.index(self.target) if self.target else None


    @abstractclassmethod
    def feature_engineering(self):
        pass


    def set_attr(self, inst_config: dict):
        '''
        Desc:
            更新类的属性
        '''
        new_config = copy(
            {k: v for k, v in inst_config.items() if k in self.__dict__.keys()})
        self.__dict__.update(inst_config)
        logging.warning(f'-------> 数据示例切换为历史模型评测模式')


    def data_split(self, X, y, test_size=0.2, random_state=42, shuffle=True, **kwargs):
        '''
        Desc:
            可处理不平衡数据集
        '''
        if self.is_imbalanced:
            logging.warning(f'the dataset is set imbalanced! ')
            # 因为数据集的标签不均衡，所以使用 StratifiedShuffleSplit 分层抽样
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            for i, (train_idx, test_idx) in enumerate(sss.split(X, y)):
                if isinstance(X, pd.DataFrame):
                    x_train, x_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
                else:
                    x_train, x_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

            assert len(X) == len(x_train) + len(x_test)
            train_pos_ratio = sum(y_train) / len(y_train)
            test_pos_ratio = sum(y_test) / len(y_test)

            logging.warning(f'train 训练集数量: {len(x_train)}, 正样本比例: {train_pos_ratio:,.3f}')
            logging.warning(f'test  测试集数量: {len(x_test)}, 正样本比例: {test_pos_ratio:,.3f}')

        else:
            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, shuffle=shuffle, **kwargs)
        return x_train, x_test, y_train, y_test


    def load_test_data(self, data, inst_config=None):
        '''
        Desc:
            data: 默认可提供 mlops 的 raw_data
            inst_config: 类实例化的参数; 可为历史模型参数, 或new model 的参数
        Return:
            返回提供给历史模型评估的最新的测试集
        Remark: Important !!! 设计模式
            如果对于需要额外处理的数据集，可以通过继承重写的方式，来实现加载历史模型的测试数据集
        Remark: TODO
            1. 此处有个问题, 即加载历史模型评估损失，由于数据集被打乱，无法获取历史的损失数据，对比有可能失真
        '''
        if inst_config is not None:
            self.set_attr(inst_config)

        model_datas = self.feature_engineering(data)
        # 同时传入 X, y
        if isinstance(model_datas, tuple):
            x_train, x_test, y_train, y_test = self.data_split(*model_datas, )
            test_data = (x_test, y_test)
        # 只传入 X
        else:
            train_data, test_data = self.data_split(model_datas)
        return test_data
