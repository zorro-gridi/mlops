from mlops.datas.datas import AbstractDatasetFactory
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import numpy as np


import logging



class BaseSeqToClassDt(AbstractDatasetFactory):
    def __init__(self, X_seq_len=10, y_seq_len=10, **kwargs):
        '''
        # X_seq_len: x 输入序列窗口长度; 例如, 近30天的股价趋势
        # y_seq_len: y 序列的观察窗口长度; 例如, 10天内股价的每日涨跌幅
        '''
        super().__init__(**kwargs)

        self.X_seq_len = X_seq_len
        self.y_seq_len = y_seq_len


    def categoric_engineering(self, raw_data, split_name='train'):
        '''
        # raw_data: pd.Dataframe group 对象
        '''
        X_seq_len = self.X_seq_len
        y_seq_len = self.y_seq_len

        # 读取模型的的数值特征
        if isinstance(self.categoric_features[0], int):
            groups_arr = [group.iloc[:, self.categoric_features].values for idx, group in raw_data]
        else:
            groups_arr = [group.loc[:, self.categoric_features].values for idx, group in raw_data]

        if split_name == 'train':
            X_arr_list = [
                g_arr[i:i+X_seq_len, :]
                for g_arr in tqdm(
                    groups_arr,
                    desc='get categoric features data -->',
                    )
                for i in range(len(g_arr) - X_seq_len - y_seq_len)
                ]
            return X_arr_list

        else:
            X_arr_list = [
                g_arr[-X_seq_len:, :]
                for g_arr in tqdm(
                    groups_arr,
                    desc='get categoric features data -->',
                    )
                ]
            return X_arr_list



    def feature_engineering(self, raw_data, split_name='train'):
        '''
        # raw_data: pd.Dataframe group 对象
        '''
        # 固定灿哥参数
        X_seq_len = self.X_seq_len
        y_seq_len = self.y_seq_len

        # 读取模型的的数值特征
        if isinstance(self.features[0], int):
            groups_arr = [group.iloc[:, self.features].values for idx, group in raw_data]
        else:
            groups_arr = [group.loc[:, self.features].values for idx, group in raw_data]
        target = self.target if isinstance(self.target, int) else self.features.index(self.target)


        if split_name == 'train':
            X_y_data_arr = [
                (
                    g_arr[i:i+X_seq_len, :],
                    # 计算持有期内的最大涨跌幅
                    g_arr[i+X_seq_len+1:i+X_seq_len+y_seq_len, target] / g_arr[i+X_seq_len, target] -  1
                    )
                for g_arr in tqdm(
                    groups_arr,
                    desc='get features data -->',
                    )
                for i in range(len(g_arr) - X_seq_len - y_seq_len)
                ]
            X_arr_list = [x for x, _ in X_y_data_arr]
            y_list = [y for _, y in X_y_data_arr]
            return X_arr_list, y_list

        else:
            X_arr_list = [
            g_arr[-X_seq_len:, :]
            for g_arr in tqdm(
                groups_arr,
                desc='get features data -->',
                )
            ]
            return X_arr_list, None