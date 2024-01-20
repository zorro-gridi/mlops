from mlops.datas.BaseDt import AbstractDatasetFactory
from functools import partial
import torch
import numpy as np

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    )
from sklearn.model_selection import train_test_split
import logging



class BaseSeqToTsDt(AbstractDatasetFactory):
    '''
    主要特征：序列到时间序列
    '''
    def __init__(self, input_features=None, **kwargs):
        '''
        # input_features: 时间序列的外部变量列表
        '''
        super(BaseSeqToTsDt, self).__init__(**kwargs)
        # input_features 作为外部变量
        self.input_features = input_features

        # 时间序列的特征是窗口长度
        if isinstance(self.features, list):
            self.features = len(self.features)


    def feature_engineering(self, vars_datas):
        '''
        # vars_datas: 数组对象
        '''
        # 时间序列的特征定义
        window = self.features + self.target
        vars_datas_copy = vars_datas.copy()
        # 时间序列处理的核心
        vars_datasets = [vars_datas_copy[i:i+window] for i in range(len(vars_datas_copy))]

        # 将不满足长度的数据全部删除，数据信息没有影响
        vars_datasets = [v for v in vars_datasets if len(v) == window]
        vars_datasets = np.array(vars_datasets)
        return vars_datasets


    def data_split(self, raw_dataset, test_size=0.2, random_state=42):
        logging.warning(f'datasets len: {len(raw_dataset)}')
        train_dataset, test_dataset = train_test_split(
            raw_dataset, test_size=test_size, random_state=random_state)

        logging.warning(f'train_dataset len: {len(train_dataset)}')

        train_data = (train_dataset[:, :-1], train_dataset[:, -1])
        test_data = (test_dataset[:, :-1], test_dataset[:, -1])
        return train_data, test_data


class SeqToTsDt_NN(BaseSeqToTsDt):
    def __init__(self, dt_class, **kwargs):
        '''
        # dt_class: 神经网络加载数据集的class
        '''
        super().__init__(**kwargs)
        self.dt_class = dt_class


    def feature_engineering(self, vars_datas):
        vars_datasets = super().feature_engineering(vars_datas)

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