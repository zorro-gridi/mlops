from mlops.datas.BaseSeqToTsDt import (
    BaseSeqToTsDt,
    SeqToTsDt_NN,
    )

from functools import partial
import torch
import numpy as np

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    )
import logging


class PreProcessSeqToTsDt_Base(BaseSeqToTsDt):
    '''
    Desc:
        1. 处理需要 preprocess 的数据集。
            进行时间序列预测前，需要对输入数据进行一定的自定义变换，再进行特征工程的数据集
            BASE 主要应用于一般机器学习的数值模型
    '''
    def __init__(self, preprocess_func, fn_config=None, **kwargs):
        '''
        Args:
            preprocess_func: 对原始数据进行特征呢工程之前，预处理的函数。可以将定义的预处理函数转换成偏函数。自定义数据预处理函数，该类数据集必选属性
            fn_config: str of dict, preprocess_func 的关键字参数字典字符串(因为 log to mlfow 不能使用 dict 类型)
        '''
        super().__init__(**kwargs)
        self.preprocess_func = preprocess_func
        self.fn_config = fn_config


    def feature_engineering(self, vars_datas, **kwargs):
        '''
        Desc:
            vars_datas: 数组对象
        Return:
            vars_datasets, 数组数据集
        '''
        # 外部变量列表input_features 不为空，也需要交给 preprocess_func 预处理
        # if self.input_features is not None:
        #     prep_func = partial(self.preprocess_func, input_features=self.input_features)
        if self.fn_config is not None:
            fn_config = eval(self.fn_config)
            prep_func = partial(self.preprocess_func, **fn_config)
            vars_datas = prep_func(vars_datas)
        else:
            vars_datas = self.preprocess_func(vars_datas)

        vars_datasets = super().feature_engineering(vars_datas, **kwargs)
        return vars_datasets



class PreProcessSeqToTsDt_NN(PreProcessSeqToTsDt_Base):
    '''
    Desc:
        NN 表示此数据集专门应用于神经网络模型
    '''
    def __init__(self, dt_class, **kwargs):
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