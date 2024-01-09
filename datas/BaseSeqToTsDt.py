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
    def __init__(self, **kwargs):
        super(BaseSeqToTsDt, self).__init__(**kwargs)
        # 时间序列的特征是窗口长度
        if isinstance(self.features, list):
            self.features = len(self.features)


    def categoric_engineering(self, ):
        pass


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
        return train_dataset, test_dataset
