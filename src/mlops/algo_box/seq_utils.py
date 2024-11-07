import numpy as np
from sklearn.cluster import KMeans
import logging
import pandas as pd
from typing import Union

import os
import sys
from pathlib import Path

home_dir = Path(os.path.expanduser('~'))
proj_path = (home_dir / 'project/pycharm').as_posix()
sys.path.append(proj_path)


"""
@Desc:
    序列数据分析处理工具箱
@Author: Zorro
@Date: 2024-03-12
"""



def phase_series_point(data: Union[pd.Series, list, np.array], start_point, n_clusters=3):
    '''
    Desc:
        本函数实现将一段序列聚类为几类关键点位。
        重点: KMeans 预测后的标签需要重拍, 使得所有的标签满足
             0: 熊; 1: 平; 2: 牛 的固定位置关系
    Args:
        data: 需要分段的序列
        start_point: 序列中需要聚类的点的起始位置。
            这个参数的潜在bug: 如果需要例如近180天的数据，
            对于交易日来说，没有180个，因此导致错位，所以在实际查询数据的时候，需要适当的放宽查询的日期范围
        n_clusters: phase point 聚类的个数
    Return:
        phase_point: 序列中每个点对应的聚类类别
        sorted_label_centers: 经过排序的族类的中心点, 与phase_point匹配
    TODO:
        1. 是否需要动态计算牛熊分位线?
    '''
    # 用于聚类分组的序列
    point_kmeans_data = [
        np.array(data[i:i+start_point]).reshape(-1, 1)
        for i in range(len(data)-start_point)]

    # 只能在循环中实例化: 因为不同族群点位不同
    # point_datas 中不能有 NaN 值
    eatimator_list = [
        KMeans(n_clusters=n_clusters, n_init=2).fit(point_datas)
        for point_datas in point_kmeans_data]

    # 取每组的最后 1 个点标签作为结果。因为，参照的是近期的整体数据
    k_labels = np.array([eatimator.labels_[-1] for eatimator in eatimator_list])
    assert len(data) - start_point == len(k_labels)

    # 获取【原始的】每个聚类族的中心点
    label_centers = np.array([eatimator.cluster_centers_.squeeze() for eatimator in eatimator_list])
    # 获取【原始的】聚类标签的中心点
    point_label_center = [center[label] for center, label in zip(label_centers, k_labels)]

    # 重排序后的标签中心点
    sorted_label_centers = np.array([np.sort(center) for center in label_centers])
    # 重新【统一分配】聚类标签，满足牛熊平的固定标签类别
    point_phase = np.array([
        list(center).index(p) for center, p in zip(sorted_label_centers, point_label_center)
        ])
    return point_phase, sorted_label_centers




# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import logging


class Peak_and_Trough_Detecter:
    '''
    Desc:
        分析一段序列的Peak(最高点)和Trough(最低点)
    '''
    def __init__(self, min_chg) -> None:
        if min_chg < 0.5 / 100:
            logging.warning(f'-------> min_chg 设置的太小, min_chg 阈值最小值: {min_chg}')
            raise Exception
        self.min_chg = min_chg
        self.reverse_points = None
        self.Ei = None


    def fit(self, sequence, point_type, base_point, start_idx=0, make_plot=True):
        '''
        Desc:
            执行寻找最大回撤、收益点流程
        '''
        gain_down_list = self.cal_sequence_peak_and_trough_point(sequence, start_idx=start_idx)
        reverse_points = self.get_batch_top_point(gain_down_list)
        self.reverse_points = reverse_points

        detecter_result = self.get_max_return_or_loss(point_type=point_type, base_point=base_point)

        if make_plot:
            self.make_plot()

        return detecter_result


    def make_plot(self):
        '''
        Desc:
            画出序列的最大回撤和收益点
        '''
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(np.arange(len(self.Ei)), self.Ei)
        for i, point in enumerate(self.reverse_points):
            if point['type'] == 'peak':
                color = 'r'
                label = 'peak' if i <= 1 else None
            else:
                color = 'g'
                label = 'trough' if i <= 1 else None
            ax.scatter(point['indx'], point['sum_chg'], c=color, label=label)

        plt.xlabel('time tick')
        plt.ylabel(f'YTD Return')

        plt.suptitle(f'Sequence "Peak"&"Trough" Point Detect Plot', fontsize=18)
        plt.title(f'trend reverse "min_chg": {(self.min_chg * 100):.1f}%', fontdict={"fontsize": 14})

        plt.legend(loc='best', fontsize=10)
        plt.show()


    def cal_sequence_peak_and_trough_point(self, sequence, start_idx=0):
        '''
        Desc:
            生成初始的序列回撤点、与反弹收益点
        Args:
            sequence: 需要分析的序列
            min_chg: 阶段上涨或下跌的趋势逆转的的阈值
            start_idx: 序列分析的起始点, 默认为 0
        '''
        Ei = np.array(sequence[start_idx:] / sequence[start_idx]) - 1
        Ei_diff = np.diff(Ei)

        self.Ei = Ei
        self.Ei_diff = Ei_diff

        Ei_peak, Ei_trough = 0, 0
        gain_down_list = []

        for j, (diff_j, sum_chg_j) in enumerate(zip(Ei_diff, Ei[1:])):
            if diff_j > 0:
                if sum_chg_j - Ei_trough >= self.min_chg:
                    Ei_peak = sum_chg_j
                    gain_down_list.append({'indx': j+1, 'type': 'peak', 'sum_chg': Ei_peak})
            else:
                if sum_chg_j - Ei_peak <= -self.min_chg:
                    Ei_trough = sum_chg_j
                    gain_down_list.append({'indx': j+1, 'type': 'trough', 'sum_chg': Ei_trough})

        if len(gain_down_list) == 0:
            logging.warning(f'-------> 回撤、收益点列表返回为空!!! 请缩小统计阶段回撤、或收益的阈值参数"min_chg", 适当调小')
            raise Exception

        # logging.warning(f'-------> gain_down_list 回撤、收益点初始记录列表:')
        # pprint(gain_down_list)
        return gain_down_list


    def find_batch_point_indx(self, gain_down_list):
        '''
        Desc:
            将peak、trough点batch批次化, 用于后续计算每个批次的最大收益、回撤点
        Args:
            gain_down_list: 回撤点和收益点的信息列表
        Return:
            reverse_indxs: 批次的起始索引和结尾索引号的列表
        '''
        reverse_indxs = [
            i for i in range(1, len(gain_down_list))
            if gain_down_list[i-1]['type'] != gain_down_list[i]['type']
            ]
        if not reverse_indxs:
            unique_point_type = gain_down_list[0]['type']
            logging.warning(f'-------> gain_down_list 回撤、收益点列表只有 {unique_point_type} 一种极值点')
            return [0, len(gain_down_list)]

        reverse_indxs.insert(0, 0)
        return reverse_indxs


    def find_the_top_point(self, point_list):
        '''
        Desc:
            寻找每个最大回撤、收益批次里面的top点
        Args:
            point_list: 最大回撤、或最高收益批次点
        Return:
            top_point: 批次中的最大的一个回撤或收益点
        '''
        top_point = None
        point_type = point_list[0]['type']
        top_point_chg = -np.inf if point_type == 'peak' else np.inf

        if point_type == 'peak':
            for point in point_list:
                if point['sum_chg'] > top_point_chg:
                    top_point_chg = point['sum_chg']
                    top_point = point

        if point_type == 'trough':
            for point in point_list:
                if point['sum_chg'] < top_point_chg:
                    top_point_chg = point['sum_chg']
                    top_point = point

        return top_point


    def get_batch_top_point(self, gain_down_list):
        '''
        Desc:
            获取每一个批次的最大收益、回撤点
        Args:
            gain_down_list: 序列的初始检测的收益点和回撤点(未去重)
            reverse_indxs: find_batch_point_indx() 方法返回的索引号列表
        Return:
            reverse_points: 过滤后的有效最大回撤、最大收益点记录列表
        '''
        if len(gain_down_list) == 1:
            logging.warning(f'-------> gain_down_list 只有 1 条回撤、或收益点记录, 直接返回')
            return gain_down_list

        reverse_indxs = self.find_batch_point_indx(gain_down_list)
        reverse_points = []

        for i in range(1, len(reverse_indxs)):
            s_indx = reverse_indxs[i-1]
            e_indx = reverse_indxs[i]
            top_point = self.find_the_top_point(gain_down_list[s_indx:e_indx])
            reverse_points.append(top_point)
        return reverse_points


    def get_top_point_indx(self, reverse_points_data: pd.DataFrame, point_type='peak'):
        '''
        Desc:
            返回最大回撤、或最大收益点的记录点索引。注意, 不是 df 的行索引记录
        Args:
            reverse_points_data: 经过去重后的回撤、收益点的 df. 也可以是该 df 的子集
            point_type:
                1. peak: 计算最大收益点日期点
                2. trough: 计算最大回撤日期点
        '''
        if point_type == 'peak':
            top_point = reverse_points_data['sum_chg'].max()
            top_cond = reverse_points_data['sum_chg'] == top_point
            row_indx = reverse_points_data.loc[top_cond].index
            top_indx = reverse_points_data.loc[row_indx, 'indx'].max()

        elif point_type == 'trough':
            top_point = reverse_points_data['sum_chg'].min()
            top_cond = reverse_points_data['sum_chg'] == top_point
            row_indx = reverse_points_data.loc[top_cond].index
            top_indx = reverse_points_data.loc[row_indx, 'indx'].max()

        else:
            raise Exception
        return top_indx


    def get_phase_max_return_or_loss(self, point_type='peak'):
        '''
        Desc:
            计算局部价格区间的最大收益、最小回撤
        Args:
            point_type:
                1. trough, 计算最大回撤, 从上一个收益点开始
                2. peak, 计算最大收益, 从上一个回撤点开始
        '''
        reverse_points_data_copy = pd.DataFrame(self.reverse_points)

        # 获取第1条记录的 point_type, 分析是从【收益点】 -> 【回撤点】, 还是【回撤点】 -> 【收益点】
        first_row_type = reverse_points_data_copy['type'].loc[0]

        reverse_points_data_copy['next_sum_chg'] = reverse_points_data_copy['sum_chg'].shift(-1)
        reverse_points_data_copy['next_indx'] = reverse_points_data_copy['indx'].shift(-1)
        reverse_points_data_copy.dropna(how='any', inplace=True)

        reverse_points_data_copy['diff'] = reverse_points_data_copy['next_sum_chg'] - reverse_points_data_copy['sum_chg']
        reverse_points_data_copy['return_days'] = reverse_points_data_copy['next_indx'] - reverse_points_data_copy['indx']

        if point_type == 'peak':
            start_type = 'trough'
        else:
            start_type = 'peak'

        if first_row_type != start_type:
            reverse_points_data_copy = reverse_points_data_copy.loc[1:, :]
            if len(reverse_points_data_copy) == 0:
                logging.warning(f'-------> point type: {start_type} 筛选结果为空!!! 请检查序列分割阈值参数"min_cgh", 适当调小')
                raise Exception

        if start_type == 'peak':
            max_diff = reverse_points_data_copy['diff'].min()
        elif start_type == 'trough':
            max_diff = reverse_points_data_copy['diff'].max()
        else:
            raise Exception

        logging.warning(f'-------> 阶段回撤与收益统计的结果表:')
        logging.warning(reverse_points_data_copy)
        logging.warning('\n')

        max_diff_cond = reverse_points_data_copy['diff'] == max_diff
        max_diff_indx = reverse_points_data_copy.loc[max_diff_cond, 'next_indx'].max()
        start_indx = reverse_points_data_copy.loc[max_diff_cond, 'indx'].max()
        return_days = max_diff_indx - start_indx

        detecter_result = {
            'max_return': max_diff,
            'return_days': return_days,
            'start_indx': int(start_indx),
            'end_indx': int(max_diff_indx),
            }
        return detecter_result


    def get_max_return_or_loss(self, point_type='peak', base_point='start'):
        '''
        Desc:
            计算全局价格区间的最大收益、或最大回撤
        Args:
            reverse_points_data:
            point_type: 判断是计算最大收益、还是最大回撤
                1. peak: 计算最大收益
                2. trough: 计算最大回撤
            base_point: 判断计算收益、或回撤的起始点
                1. start: 从起始点开始计算
                2. peak: 从最大收益点开始计算
                3. phase: 阶段的最大回撤
        NOTE:
            最大回撤计算的不对
        '''
        if point_type == 'peak':
            stat_name = '最高收益'
        elif point_type == 'trough':
            stat_name = '最大回撤'
        else:
            raise Exception

        reverse_points_data = pd.DataFrame(self.reverse_points)
        if base_point == 'phase':
            logging.warning(f'-------> 统计阶段的最大回撤与最高收益')
            detecter_result = self.get_phase_max_return_or_loss(point_type=point_type)
            phase_return = detecter_result['max_return']
            return_days = detecter_result['return_days']
            logging.warning(f'-------> {stat_name}: {phase_return}, 持续 {return_days} 个交易日\n')
            return detecter_result

        s2peak_return = reverse_points_data['sum_chg'].max()
        s2trough_loss = reverse_points_data['sum_chg'].min()
        # 根据point_type参数，动态变化
        # 1. 如果是 'peak', 则返回最大收益点的日期点
        # 2. 如果是 'trough', 则返回最大回撤点的日期点
        top_peak_indx = self.get_top_point_indx(reverse_points_data, point_type='peak')
        top_trough_indx = self.get_top_point_indx(reverse_points_data, point_type='trough')

        # 如果计算最大收益
        if point_type == 'peak':
            # 从起始点开始计算
            if base_point == 'start':
                logging.warning(f'-------> 从【起始点】开始')
                logging.warning(f'-------> 最大收益: {s2peak_return}, 持续 {top_peak_indx} 个交易日')
                detecter_result = {
                    'max_return': s2peak_return,
                    'return_days': top_peak_indx,
                    'start_indx': 0,
                    'end_indx': int(top_peak_indx),
                    }
                return detecter_result

            # 从最大回撤点开始计算
            elif base_point == 'trough':
                # 从最大收益点之前的数据，获取新的最大回撤点
                before_peak_cond = reverse_points_data['indx'] < top_peak_indx
                reverse_points_data_before_peak = reverse_points_data.loc[before_peak_cond, :].copy()
                logging.warning(f'-------> 统计从【最大回撤】-> 【最高收益】的结果表:\n{reverse_points_data_before_peak}')

                top_trough_loss_before_peak = reverse_points_data_before_peak['sum_chg'].min()
                top_trough_indx_before_peak = self.get_top_point_indx(reverse_points_data_before_peak, point_type='trough')

                trough2peak_return = s2peak_return - top_trough_loss_before_peak
                return_days = top_peak_indx - top_trough_indx_before_peak

                logging.warning(f'-------> 从【最大回撤点】开始')
                logging.warning(f'-------> 最大收益: {trough2peak_return}, 持续 {return_days} 个交易日')
                detecter_result = {
                    'max_return': trough2peak_return,
                    'return_days': return_days,
                    'start_indx': int(top_trough_indx_before_peak),
                    'end_indx': int(top_peak_indx),
                    }
                return detecter_result

            else:
                raise Exception

        # 如果计算最大回撤
        elif point_type == 'trough':
            # 从起始点开始计算
            if base_point == 'start':
                if s2trough_loss > 0:
                    logging.warning(f'-------> 从起始点以来, 最大回撤大于0, 因此, 阶段最大回撤替代')
                    logging.warning(f'-------> 最大回撤: {s2trough_loss}, 持续 {top_trough_indx} 个交易日')
                    return self.get_phase_max_return_or_loss(reverse_points_data, point_type=point_type)
                else:
                    logging.warning(f'-------> 从【起始点】开始')
                    logging.warning(f'-------> 最大回撤: {s2trough_loss}, 持续 {top_trough_indx} 个交易日')
                    detecter_result = {
                        'max_return': s2trough_loss,
                        'return_days': top_trough_indx,
                        'start_indx': 0,
                        'end_indx': int(top_trough_indx),
                        }
                    return detecter_result

            # 从最大收益点开始计算
            elif base_point == 'peak':
                # 从最大收益点之后的数据，获取新的最大回撤点
                after_peak_cond = reverse_points_data['indx'] >= top_peak_indx
                reverse_points_data_after_peak = reverse_points_data.loc[after_peak_cond, :].copy()
                logging.warning(f'-------> 统计从【最高收益】 -> 【最大回撤】的结果表:\n{reverse_points_data_after_peak}')

                top_trough_indx_after_peak = self.get_top_point_indx(reverse_points_data_after_peak, point_type='trough')
                top_trough_loss_after_peak = reverse_points_data_after_peak['sum_chg'].min()

                peak2trough_loss = top_trough_loss_after_peak - s2peak_return
                loss_days = top_trough_indx_after_peak - top_peak_indx
                logging.warning(f'-------> 从【最高点】开始')
                logging.warning(f'-------> 最大回撤: {peak2trough_loss}, 持续 {loss_days} 个交易日')
                detecter_result = {
                    'max_return': peak2trough_loss,
                    'return_days': loss_days,
                    'start_indx': int(top_peak_indx),
                    'end_indx': int(top_trough_indx_after_peak),
                    }
                return detecter_result

            else:
                raise Exception

        else:
            raise Exception
