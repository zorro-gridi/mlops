from mlops.datas.BaseDt import AbstractDatasetFactory
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import logging
import numpy as np
import pandas as pd
from typing import Union


class BaseSeqToClassDt(AbstractDatasetFactory):
    '''
    Desc:
        主要特征：序列到类别
    '''
    def __init__(self, X_seq_len=10, y_seq_len=10, **kwargs):
        '''
        Args:
            X_seq_len: x 输入序列窗口长度; 例如, 近30天的股价趋势
            y_seq_len: y 序列的观察窗口长度; 例如, 10天内股价的每日涨跌幅
        '''
        super().__init__(**kwargs)

        self.X_seq_len = X_seq_len
        self.y_seq_len = y_seq_len

    def build_train_data(self, data: np.ndarray):
        '''
        Desc:
            构造 train 训练的原始数据
        '''
        logging.warning(f'-----> split name: "train" mode')
        X_y_arr_list = [
            (
                g_arr[i-self.X_seq_len:i, :],
                # 计算持有期（y_seq_len）序列内的涨跌幅
                # 此处是用数组除标量，所以得到的 y 先是一个 list 对象
                # 为什么额外 +1：因为计算的是次日的涨跌幅，当日的数据无用
                g_arr[i+1:i+self.y_seq_len, self.target] / g_arr[i, self.target] -  1
                )
            for g_arr in tqdm(
                data,
                desc='get features data ---->',
                )
            # 每一个 group 是一个股票/基金的历史指标集合趋势
            for i in range(self.X_seq_len, len(g_arr)-self.y_seq_len)
            # 保证 g_arr >= X seq + y seq 的长度
            if len(g_arr) >= self.X_seq_len+self.y_seq_len
            ]
        if len(X_y_arr_list) == 0:
            logging.warning(f'\n======> 序列分块数为零！请增大样本数量，或者减少分块序列的长度')
            raise

        logging.warning(f'X_y_arr_list length ------------------> {len(X_y_arr_list)}')
        return X_y_arr_list

    def build_test_data(self, data: np.ndarray):
        '''
        Desc:
            构造 test 全量测试数据
        '''
        logging.warning(f'-----> split name: "test" mode, 默认使用所有记录')
        X_test_list = [
            g_arr[i-self.X_seq_len:i, :]
            for g_arr in tqdm(
                data,
                desc='get features data ---->',
                )
            for i in range(self.X_seq_len, len(g_arr))
            # 保证 g_arr >= X seq
            if len(g_arr) >= self.X_seq_len
            ]
        return X_test_list

    def build_pred_data(self, data: np.ndarray):
        '''
        Desc:
            构造 pred 最新预测数据
        '''
        logging.warning(f'-----> split name: "pred" mode, 默认只预测最后一条记录')
        X_pred_list = [
            g_arr[-self.X_seq_len:, :]
            for g_arr in tqdm(data, desc='get features data -->',)
            # 仍要保证 g_arr 的长度大于等于要求输入序列的长度
            if len(g_arr) >= self.X_seq_len
            ]
        return X_pred_list


    def categoric_engineering(self, raw_data: Union[pd.DataFrame, pd.Grouper], split_name='train'):
        '''
        Args:
            raw_data: pd.Dataframe group 对象
            split_name: 数据集名称, Union['train', 'test', 'pred']
        Remark:
            test 数据集的作用是测试全量的数据集, test 数据比 trian 数据多1份末尾序列的数据
        '''
        if type(raw_data).__name__ == 'DataFrameGroupBy':
            # 读取模型的的数值特征
            if isinstance(self.categoric_features[0], int):
                groups_arr = [group.iloc[:, self.categoric_features].values for idx, group in raw_data]
            else:
                groups_arr = [group.loc[:, self.categoric_features].values for idx, group in raw_data]
        elif isinstance(raw_data, pd.DataFrame):
            groups_arr = raw_data.to_numpy()

        if split_name == 'train':
            X_y_arr_list = self.build_train_data(groups_arr)
            # 提取 X arr list 加工分类变量
            X_arr_list = [x for x, _ in X_y_arr_list]
            return X_arr_list

        elif split_name == 'test':
            X_arr_list = self.build_test_data(groups_arr)
            return X_arr_list

        else:
            X_arr_list = self.build_pred_data(groups_arr)
            return X_arr_list


    def feature_engineering(self, raw_data: Union[pd.DataFrame, pd.Grouper], split_name='train'):
        '''
        Args:
            raw_data: pd.Dataframe grouper 对象
            split_name: 数据集的名称, Union['train', 'test', 'pred']
        '''
        if split_name not in ['train', 'test', 'pred']:
            logging.warning(f'{split_name}, No define dataset name error !')
            raise Exception

        if type(raw_data).__name__ == 'DataFrameGroupBy':
            # 读取模型的的数值特征&目标标签
            if isinstance(self.features[0], int):
                groups_arr = [group.iloc[:, self.features].values for idx, group in raw_data]
            else:
                groups_arr = [group.loc[:, self.features].values for idx, group in raw_data]
        elif isinstance(raw_data, pd.DataFrame):
            groups_arr = raw_data.to_numpy()
        # logging.warning(f'groups_arr length ---------------------------> {len(groups_arr)}')

        if split_name == 'train':
            X_y_arr_list = self.build_train_data(groups_arr)
            # 提取 X, y arr list seperately
            X_arr_list = [x for x, _ in X_y_arr_list]
            y_list = [y for _, y in X_y_arr_list]
            return X_arr_list, y_list

        elif split_name == 'test':
            X_arr_list = self.build_test_data(groups_arr)
            return X_arr_list, None

        else:
            X_arr_list = self.build_pred_data(groups_arr)
            return X_arr_list, None
