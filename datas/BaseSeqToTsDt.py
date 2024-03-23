from mlops.datas.BaseDt import AbstractDatasetFactory
from functools import partial
import torch
import numpy as np
from copy import copy

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    )
from sklearn.model_selection import train_test_split
import logging


class BaseSeqToTsDt(AbstractDatasetFactory):
    '''
    Desc:
        序列到时间序列
    '''
    def __init__(self, input_features=None, **kwargs):
        '''
        Args:
            input_features: 时间序列的外部变量列表
        '''
        super(BaseSeqToTsDt, self).__init__(**kwargs)
        # input_features 作为外部变量
        self.input_features = input_features

        # 时间序列的特征是窗口长度
        if isinstance(self.features, list):
            self.features = len(self.features)


    def feature_engineering(self, vars_datas: np.array, pred_len=0, split_name='train'):
        '''
        Desc:
            构造序列特征的步骤:
                1. 分割序列。以输入features长度+预测targe长度作为一条(X, y)序列样本的Window
                2. Window数据切片, 获得 (X, y) 样本
        Args:
            vars_datas: 输入序列的数组
            pred_len: 预测样本的长度。例如，保留最后 1 条数据作为预测决策样本
            split_name: 返回的数据集类型: "train", "pred"
        Return:
            shape 为 (-1, window, features) 的多维数组, 还未分出(X, y)数据集
                -1:  表示数据集的size
                window: 时间序列的窗口长度 = features + target; 因此, X, y = dataset[:, features], dataset[:, target]
                features: 输入的特征数量 (Unit单变量 or MultVars多变量)
        '''
        # 时间序列的特征定义
        window = self.features + self.target
        vars_datas_copy = copy(vars_datas)
        # 时间序列处理的核心
        vars_datasets = [
            vars_datas_copy[i:i+window]
            for i in range(len(vars_datas_copy))
            if len(vars_datas_copy[i:i+window]) == window
            ]
        # 将不满足长度的数据全部删除，数据信息没有影响
        vars_datasets = [v for v in vars_datasets if len(v) == window]
        vars_datasets = np.array(vars_datasets)

        if pred_len > 0 and split_name == 'train':
            logging.warning(f'return training datasets')
            vars_datasets = vars_datasets[:-pred_len]
        if split_name == 'pred':
            logging.warning(f'return prediction datasets')
            vars_datasets = vars_datasets[-pred_len:]
        return vars_datasets


    def data_split(self, raw_dataset:np.ndarray, test_size=0.2, random_state=42):
        '''
        Desc:
            切分数据集, 并区分出X, y
        Args:
            raw_dataset: 输入数据集
        Return:
            train_data: (X_train, y_train)
            test_data: (X_test, y_test)
        '''
        logging.warning(f'raw datasets shape: {raw_dataset.shape}')
        # 此处 self.target 表示预测变量的长度, 一般为最后 1 个索引位置, 即 target = 1
        X, y = raw_dataset[:, :-self.target], raw_dataset[:, -self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        logging.warning(f'train_dataset len: {len(X_train)}')
        train_data = (X_train, y_train)
        test_data = (X_test, y_test)
        return train_data, test_data


class SeqToTsDt_NN(BaseSeqToTsDt):
    def __init__(self, dt_class, **kwargs):
        '''
        Desc:
            dt_class: 神经网络加载数据集的class
        '''
        super().__init__(**kwargs)
        self.dt_class = dt_class


    def feature_engineering(self, vars_datas, **kwargs):
        vars_datasets = super().feature_engineering(vars_datas, **kwargs)

        window = self.features + self.target
        # 神经网络模型需要 batch_size 三维数据
        vars_datasets = vars_datasets.reshape(len(vars_datasets), window, -1)
        vars_datasets = vars_datasets.astype(np.float32)
        vars_datasets = torch.from_numpy(vars_datasets)
        vars_datasets = self.dt_class(vars_datasets)
        return vars_datasets


    def data_split(self, raw_dataset, test_size=0.2, random_state=42):
        generator = torch.Generator().manual_seed(random_state)
        train_dataset, test_dataset = random_split(
            raw_dataset, lengths=[1-test_size, test_size], generator=generator)

        logging.warning(f'train_dataset len: {len(train_dataset)}')
        return train_dataset, test_dataset


    def load_test_data(self, data, inst_config=None):
        '''
        # inst_config: 类实例化的参数;可为历史模型参数, 或new model 的参数
        '''
        if inst_config is not None:
            self.set_attr(inst_config)

        model_datas = self.feature_engineering(data)
        train_dataset, test_dataset = self.data_split(model_datas)
        return test_dataset