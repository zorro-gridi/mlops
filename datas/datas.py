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
    def __init__(self, features=[], categoric_features=[], target=None, preprocess_func=None):
        '''
        # preprocess_func: 数据进行特征工程之前的预处理函数
        # categoric_features: 数据中的分类特征
        # features: 数据默认输入特征
        '''
        self.features = features
        self.target = target
        self.preprocess_func = preprocess_func
        self.is_imbalanced = True
        self.categoric_features = categoric_features


    @abstractclassmethod
    def feature_engineering(self, *args, **kwargs):
        pass


    def set_attr(self, inst_config):
        [setattr(self, k, v)
         for k, v in inst_config.items()
         if k in self.__dict__.keys()
         ]
        # self.__dict__.update(inst_config)


    def data_split(self, X, y, test_size=0.2, random_state=42):
        if self.is_imbalanced:
            logging.warning(f'the dataset is imbalanced! ')
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
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test


    def load_test_data(self, data, inst_config=None):
        '''
        # inst_config: 类实例化的参数;可为历史模型参数, 或new model 的参数
        '''
        if inst_config is not None:
            self.set_attr(inst_config)

        model_datas = self.feature_engineering(data)
        if isinstance(model_datas, tuple):
            x_train, x_test, y_train, y_test = self.data_split(*model_datas)
            test_data = (x_test, y_test)
        else:
            train_data, test_data = self.data_split(model_datas)
        return test_data



class SeqToSeqClassDt(AbstractDatasetFactory):
    def __init__(self, X_seq_len=10, y_seq_len=10, y_threshold=3, **kwargs):
        super(SeqToSeqClassDt, self).__init__(**kwargs)
        self.X_seq_len = X_seq_len
        self.y_seq_len = y_seq_len
        self.y_threshold = y_threshold


    def categoric_engineering(self, raw_data, split_name='train'):
        # 固定灿哥参数
        X_seq_len = self.X_seq_len
        y_seq_len = self.y_seq_len
        y_threshold = self.y_threshold
        # X_seq_len, y_seq_len, y_threshold = data_args
        # logging.warning(f'data args: {data_args}')

        # 读取模型的的数值特征
        if isinstance(self.categoric_features[0], int):
            groups_arr = [group.iloc[:, self.categoric_features].values for idx, group in raw_data]
        else:
            groups_arr = [group.loc[:, self.categoric_features].values for idx, group in raw_data]

        if split_name == 'train':
            X_data_arr = [
                np.min(g_arr[i:i+X_seq_len, :]).reshape(1, -1)
                for g_arr in tqdm(
                    groups_arr,
                    desc='get categoric features data -->',
                    )
                for i in range(len(g_arr) - X_seq_len - y_seq_len)
                ]
            X = np.concatenate(X_data_arr, axis=0)
            return X

        else:
            X_data_arr = [
                np.min(g_arr[-X_seq_len:, :]).reshape(1, -1)
                for g_arr in tqdm(
                    groups_arr,
                    desc='get categoric features data -->',
                    )
                ]
            X = np.concatenate(X_data_arr, axis=0)
            return X


    def feature_engineering(self, raw_data, split_name='train'):
        '''
        # X_seq_len: x 输入序列长度; 例如近30天的股价趋势
        # y_seq_len: y 序列 to label 的观察长度; 例如10天内股价的最高涨幅
        # y_threshold: label y 的阈值
        '''
        # 固定灿哥参数
        X_seq_len = self.X_seq_len
        y_seq_len = self.y_seq_len
        y_threshold = self.y_threshold
        # X_seq_len, y_seq_len, y_threshold = data_args
        # logging.warning(f'data args: {data_args}')

        # 读取模型的的数值特征
        if isinstance(self.features[0], int):
            groups_arr = [group.iloc[:, self.features].values for idx, group in raw_data]
        else:
            groups_arr = [group.loc[:, self.features].values for idx, group in raw_data]
        target = self.target if isinstance(self.target, int) else self.features.index(self.target)

        # 处理分类特征
        X_cate = None
        if len(self.categoric_features) > 0:
            X_cate = self.categoric_engineering(raw_data, split_name=split_name)

        if split_name == 'train':
            X_y_data_arr = [
                (
                    MinMaxScaler().fit_transform(g_arr[i:i+X_seq_len, :]),
                    # 计算持有期内的最大涨跌幅
                    np.max(
                        g_arr[i+X_seq_len+1:i+X_seq_len+y_seq_len, target]) / g_arr[i+X_seq_len, target] -  1
                    )
                for g_arr in tqdm(
                    groups_arr,
                    desc='get features data -->',
                    )
                for i in range(len(g_arr) - X_seq_len - y_seq_len)
                ]
            X_arr_list = [x for x, _ in X_y_data_arr]
            X_arr_list = [x.reshape(1, -1) for x in X_arr_list]
            X = np.concatenate(X_arr_list, axis=0)
            y = np.array([1 if y >= y_threshold / 100 else 0  for x, y in X_y_data_arr])

            X = np.concatenate([X, X_cate], axis=1) if X_cate is not None else X
            return X, y

        else:
            X_data_arr = [
                MinMaxScaler().fit_transform(g_arr[-X_seq_len:, :]).reshape(1, -1)
                for g_arr in tqdm(
                    groups_arr,
                    desc='get features data -->',
                    )
                ]
            X = np.concatenate(X_data_arr, axis=0)
            X = np.concatenate([X, X_cate], axis=1) if X_cate is not None else X
            return X



class SeqToTsDt(AbstractDatasetFactory):
    def __init__(self, input_features=None, dt_class=None, model_type='nn', **kwargs):
        '''
        # input_features: 时间序列的外部变量列表
        # dt_class: 加载数据为 torch datasets class
        # model_type: 模型的类型
        '''
        super(SeqToTsDt, self).__init__(**kwargs)
        # input_features 作为外部变量
        self.input_features = input_features
        # 时间序列的特征是窗口长度
        if isinstance(self.features, list):
            self.features = len(self.features)

        self.dt_class = dt_class
        self.model_type = model_type


    def feature_engineering(self, vars_datas):
        '''
        # preprocess_func: 对原始数据进行特征呢工程之前，预处理的函数。可以将定义的预处理函数封装成偏函数
        '''
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


        if self.model_type == 'nn':
            # 神经网络模型需要 batch_size 三维数据
            vars_datasets = vars_datasets.reshape(len(vars_datasets), window, -1)
            vars_datasets = vars_datasets.astype(np.float32)
            vars_datasets = torch.from_numpy(vars_datasets)

        if self.dt_class is not None:
            # 装载成 torch datasets 格式
            vars_datasets = self.dt_class(vars_datasets)
        return vars_datasets



    def data_split(self, raw_dataset, test_size=0.2, random_state=42):
        logging.warning(f'datasets len: {len(raw_dataset)}')

        if self.model_type == 'nn':
            generator = torch.Generator().manual_seed(random_state)
            train_dataset, test_dataset = random_split(
                raw_dataset, lengths=[1-test_size, test_size], generator=generator)

        else:
            train_dataset, test_dataset = train_test_split(
                raw_dataset, test_size=test_size, random_state=random_state)

        logging.warning(f'train_dataset len: {len(train_dataset)}')
        return train_dataset, test_dataset