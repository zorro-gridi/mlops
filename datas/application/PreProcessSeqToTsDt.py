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
    PreProcessSeqToTsDt: 进行时间序列预测前，需要对输入数据进行一定的自定义变换，再进行特征工程的数据集
    # preprocess_func 自定义数据预处理函数，属性必选
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def feature_engineering(self, vars_datas):
        '''
        # vars_datas: 数组对象
        # preprocess_func: 对原始数据进行特征呢工程之前，预处理的函数。可以将定义的预处理函数转换成偏函数
        '''
        if self.input_features is not None:
            prep_func = partial(self.preprocess_func, input_features=self.input_features)
            vars_datas = prep_func(vars_datas)

        else:
            vars_datas = self.preprocess_func(vars_datas)

        logging.warning(f'var datas preview: {vars_datas[0]}')

        vars_datasets = super().feature_engineering(vars_datas)
        return vars_datasets



class PreProcessSeqToTsDt_NN(PreProcessSeqToTsDt_Base):
    def __init__(self, dt_class, **kwargs):
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