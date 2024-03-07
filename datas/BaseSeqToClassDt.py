from mlops.datas.BaseDt import AbstractDatasetFactory
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import logging
import numpy as np



class BaseSeqToClassDt(AbstractDatasetFactory):
    '''
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

        # 读取模型的的数值特征&目标标签
        if isinstance(self.features[0], int):
            groups_arr = [group.iloc[:, self.features].values for idx, group in raw_data]
        else:
            groups_arr = [group.loc[:, self.features].values for idx, group in raw_data]
        logging.warning(f'groups_arr length ---------------------------> {len(groups_arr)}')
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
                    desc='get features data ---->',
                    )
                # 每一个 group 是一个股票/基金的历史指标集合趋势
                for i in range(len(g_arr) - X_seq_len - y_seq_len)
                if len(g_arr) >= X_seq_len + y_seq_len
                ]
            if len(X_y_data_arr) == 0:
                logging.warning(f'\n=======> 序列分块数为零！请增大样本数量，或者减少分块序列的长度')
                raise

            logging.warning(f'X_y_data_arr length -----------------------> {len(X_y_data_arr)}')
            X_arr_list = [x for x, _ in X_y_data_arr]
            y_list = [y for _, y in X_y_data_arr]
            return X_arr_list, y_list

        else:
            X_arr_list = [
                g_arr[-X_seq_len:, :]
                for g_arr in tqdm(groups_arr, desc='get features data -->',)
            ]
            return X_arr_list, None