from mlops.datas.datas import AbstractDatasetFactory
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


    def categoric_engineering(self, ):
        pass


    def feature_engineering(self, vars_datas):
        '''
        # vars_datas: 数组对象
        # preprocess_func: 对原始数据进行特征呢工程之前，预处理的函数。可以将定义的预处理函数封装成偏函数
        '''
        # 时间序列的特征定义
        window = self.features + self.target
        if self.preprocess_func is not None and self.input_features is not None:
            prep_func = partial(self.preprocess_func, input_features=self.input_features)
            vars_datas = prep_func(vars_datas)

        elif self.preprocess_func is not None and self.input_features is None:
            vars_datas = self.preprocess_func(vars_datas)

        logging.warning(f'var datas preview: {vars_datas[0]}')

        model_types_options = ['nn', 'boost']
        if self.model_type not in model_types_options:
            logging.warning(f'Supported model types options: {model_types_options}')
            raise Exception

        vars_datas_copy = vars_datas.copy()
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
        return train_dataset, test_dataset
