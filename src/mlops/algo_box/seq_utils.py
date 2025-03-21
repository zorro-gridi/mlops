# %%
import numpy as np
from sklearn.cluster import KMeans
import logging
import pandas as pd
from typing import Union
import time

import matplotlib.pyplot as plt
from pprint import pprint

import os
import sys
from pathlib import Path

home_dir = Path(os.path.expanduser('~'))
proj_path = (home_dir / 'project/pycharm').as_posix()
sys.path.append(proj_path)

from tools.DB_Client import DB_Client


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
        KMeans(n_clusters=n_clusters, n_init=2, random_state=42).fit(point_datas)
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
        self.gain_down_list = None
        self.reverse_indxs = None
        self.Ei = None
        self.last_top_point = {'indx': 0}
        self.reverse_points = []

    def fit(self, sequence, point_type=None, base_point=None, end_point='current', start_idx=0, make_plot=True):
        '''
        Desc:
            执行寻找最大回撤、收益点流程
        Args:
            sequence: 待分析的序列
            point_type: 指明找最大回撤 trough、或最大收益点 peak.
            base_point: 收益区间的起始点
                options: ["start", "default", "peak", "trough", "max_peak", "max_trough", ]
                # 具体含义参考 curr_return_base_hist_reverse_point() 方法定义
            end_point: 收益区间的结束点
                options:
                    1. "phase": 表示阶段点, 可能是最低回撤点, 也可能是阶段高点; 因此，得到的收益统计可能是最大回撤、或最大收益
                    2. "current": 表示收益区间的结束点为当前最新点
                # 具体含义参考 curr_return_base_hist_reverse_point() 方法定义
            start_idx: 索引切片的起始点
        '''
        base_point_options = [
            "start",
            "default", "peak", "trough", "max_peak", "max_trough",
            ]
        if base_point not in base_point_options:
            logging.warning(f'-------> Invalid args value! Args "base_point" Params Options: {base_point_options}')
            raise Exception

        end_point_options = ["phase", "current"]
        if end_point not in end_point_options:
            logging.warning(f'-------> Invalid args value! Args "end_point" Params Options: {end_point_options}')
            raise Exception

        while True:
            try:
                gain_down_list = self.cal_sequence_peak_and_trough_point(sequence, start_idx=self.last_top_point['indx'])
                self.gain_down_list = gain_down_list
                self.get_batch_top_point(gain_down_list)
            except:
                logging.warning(f'----------> gain_down_list 循环完成..., 准备作图')
                break

        detecter_result = None
        if end_point == 'phase':
            if base_point is None or point_type is None:
                logging.warning(f'-------> end_point 为 "phase" 的模式下, point_type 和 base_point 参数不能为空！')
                raise Exception

            detecter_result = self.get_max_return_or_loss(point_type=point_type, base_point=base_point)

        elif end_point == 'current':
            detecter_result = self.curr_return_base_hist_reverse_point(base_point=base_point)

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
        ax.plot(np.arange(len(self.Ei)), [self.Ei[-1]] * len(self.Ei), ls='--')
        ax.scatter(len(self.Ei), self.Ei[-1], c='blue')
        for i, point in enumerate(self.reverse_points):
        # for unit-test
        # for i, point in enumerate(self.gain_down_list):
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
        self.Ei = np.array(sequence / sequence[0]) - 1

        Ei = np.array(sequence[start_idx:] / sequence[start_idx]) - 1
        Ei_diff = np.diff(Ei)
        Ei_diff = np.around(Ei_diff, 4)

        # self.Ei = Ei
        self.Ei_diff = Ei_diff

        Ei_peak, Ei_trough = 0, 0
        gain_down_list = []

        is_peak_reverse = False
        is_trough_reverse = False

        last_point_type = None
        phase_peak, phase_trough = 0, 0

        for j, (diff_j, sum_chg_j) in enumerate(zip(Ei_diff, Ei[1:])):
            if diff_j > 0:
                if any([
                    # 1. 当前点位相比前期最低点累计上涨超过 self.min_chg
                    sum_chg_j - Ei_trough >= self.min_chg and not is_peak_reverse,
                    sum_chg_j - phase_trough >= self.min_chg,
                    ]):
                    Ei_peak = sum_chg_j
                    gain_down_list.append({'indx': j+1, 'type': 'peak', 'sum_chg': Ei_peak})

                    # 始终保持阶段的收益点为局部最大值，而Ei_peak需要迭代更新，目的是判断趋势是否持续上涨，从而判断 peak 反转点
                    if Ei_peak > phase_peak:
                        phase_peak = Ei_peak

                    if last_point_type and last_point_type != 'peak':
                        is_trough_reverse = True
                        # 下跌趋势被反转，进入上涨通道，所以应该重新定义收益的基点。phase_peak 目的的是定义新基点
                        phase_peak = sum_chg_j
                        is_peak_reverse = False

            elif diff_j < 0:
                if any([
                    # 1. 当前点位相比前期最高点累计下跌超过 self.min_chg
                    sum_chg_j - Ei_peak <= -self.min_chg and not is_trough_reverse,
                    sum_chg_j - phase_peak <= -self.min_chg,
                    ]):
                    Ei_trough = sum_chg_j
                    gain_down_list.append({'indx': j+1, 'type': 'trough', 'sum_chg': Ei_trough})

                    # 始终保持阶段的回撤点为局部最小值，而 Ei_trough 需要迭代更新，目的是判断下跌趋势是否持续，从而判断 trough 反转点
                    if Ei_trough < phase_trough:
                        phase_trough = Ei_trough

                    if last_point_type and last_point_type != 'trough':
                        is_peak_reverse = True
                        # 上涨趋势被反转，进入下跌通道，所以应该重新定义回撤点的基点
                        phase_trough = sum_chg_j
                        is_trough_reverse = False

            if gain_down_list:
                last_point_type = gain_down_list[-1]['type']


        if len(gain_down_list) == 0:
            logging.warning(f'-------> 回撤、收益点列表返回为空!!! 请缩小统计阶段回撤、或收益的阈值参数"min_chg", 适当调小')
            raise Exception

        # for unit-test
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

        # TODO: 第1批、和最后1批的 point_type 批次在循环中没有加入
        reverse_indxs.insert(0, 0)
        reverse_indxs.append(len(gain_down_list))
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
        self.reverse_indxs = reverse_indxs
        # self.reverse_points = []

        # # version1: 一次性提取完
        # for i in range(1, len(reverse_indxs)):
        #     s_indx = reverse_indxs[i-1]
        #     e_indx = reverse_indxs[i]
        #     patch_phase = gain_down_list[s_indx:e_indx]
        #     if len(patch_phase) == 0:
        #         continue
        #     top_point = self.find_the_top_point(patch_phase)
        #     self.reverse_points.append(top_point)

        # version2: 为保证准确性，一次提取一个
        s_indx = reverse_indxs[0]
        e_indx = reverse_indxs[1]
        patch_phase = gain_down_list[s_indx:e_indx]
        top_point = self.find_the_top_point(patch_phase)
        top_point['indx'] += self.last_top_point['indx']
        top_point['sum_chg'] = self.Ei[top_point['indx']]
        self.reverse_points.append(top_point)
        print(f'-------------> test: {top_point}')
        self.last_top_point = top_point
        return top_point


    def get_top_point_indx(self, reverse_points_data: pd.DataFrame, point_type='peak'):
        '''
        Desc:
            返回最大回撤、或最大收益记录点的索引。注意: 不是 df 的行索引记录
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
        first_row_type = reverse_points_data_copy['type'].iloc[0]

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
            reverse_points_data_copy = reverse_points_data_copy.iloc[1:, :]
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
            'return_type': 'phase2phase',
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
            logging.warning(f'-------> get_max_return_or_loss() point_type 参数不合法')
            raise Exception
        # 原始的反转点记录数据
        reverse_points_data = pd.DataFrame(self.reverse_points)

        if base_point == 'phase':
            logging.warning(f'-------> 统计阶段的最大回撤与最高收益')
            detecter_result = self.get_phase_max_return_or_loss(point_type=point_type)
            phase_return = detecter_result['max_return']
            return_days = detecter_result['return_days']
            logging.warning(f'-------> {stat_name}: {phase_return}, 持续 {return_days} 个交易日\n')
            return detecter_result

        # s2 标识从记录数据开始，s2peak 表示从记录开始到最高收益点
        s2peak_return = reverse_points_data['sum_chg'].max()
        s2trough_loss = reverse_points_data['sum_chg'].min()
        # 根据point_type参数，动态变化
        # 1. 如果是 'peak', 则返回最大收益点的日期点
        # 2. 如果是 'trough', 则返回最大回撤点的日期点
        top_peak_indx = self.get_top_point_indx(reverse_points_data, point_type='peak')
        top_trough_indx = self.get_top_point_indx(reverse_points_data, point_type='trough')

        # 如果计算最大收益
        if point_type == 'peak':
            # 从起始点开始, 计算最大收益
            if base_point == 'start':
                logging.warning(f'-------> 从【起始点】开始计算【最高收益】')
                logging.warning(f'-------> 最大收益: {s2peak_return}, 持续 {top_peak_indx} 个交易日')
                detecter_result = {
                    'max_return': s2peak_return,
                    'return_days': top_peak_indx,
                    'start_indx': 0,
                    'end_indx': int(top_peak_indx),
                    'return_type': 's2MaxPeak',
                    }
                return detecter_result

            # 从区间最大回撤点开始，计算最大收益
            elif base_point == 'trough':
                after_trough_cond = reverse_points_data['indx'] >= top_trough_indx
                reverse_points_data_after_trough = reverse_points_data.loc[after_trough_cond, :].copy()
                if len(reverse_points_data_after_trough) == 1:
                    logging.warning(f'-------> 该序列第1个点即是最高点, 此后持续下跌, 最高点还未收复!!! 从【最大回撤】-> 【最高收益】的结果为 0')
                    detecter_result = {
                        'max_return': 0,
                        'return_days': 0,
                        'start_indx': 0,
                        'end_indx': 0,
                        'return_type': 'stillTroughStatus',
                        }
                    return detecter_result

                logging.warning(f'-------> 统计从【最大回撤】-> 【最高收益】的结果表:\n{reverse_points_data_after_trough}')

                top_peak_return_after_trough = reverse_points_data_after_trough['sum_chg'].max()
                top_peak_indx_after_trough = self.get_top_point_indx(reverse_points_data_after_trough, point_type='peak')

                # 从记录的的最低回撤点，到回撤点右侧的最高收益点的区间收益
                trough2peak_return = top_peak_return_after_trough - s2trough_loss
                return_days = top_peak_indx_after_trough - top_trough_indx

                logging.warning(f'-------> 从【最大回撤】开始计算【最高收益】')
                logging.warning(f'-------> 最大收益: {trough2peak_return}, 持续 {return_days} 个交易日')
                detecter_result = {
                    'max_return': trough2peak_return,
                    'return_days': return_days,
                    'start_indx': int(top_trough_indx),
                    'end_indx': int(top_peak_indx_after_trough),
                    'return_type': 'maxTrough2maxPeak',
                    }
                return detecter_result

            else:
                logging.warning(f'-------> get_max_return_or_loss(point_type="peak") 模式下, base_point 参数可选值: ["start", "trough"]')
                raise Exception

        # 如果计算最大回撤
        elif point_type == 'trough':
            # 从起始点开始, 计算最大回撤
            if base_point == 'start':
                if s2trough_loss > 0:
                    logging.warning(f'-------> 从起始点以来, 最大回撤大于0, 因此, 阶段最大回撤替代')
                    logging.warning(f'-------> 最大回撤: {s2trough_loss}, 持续 {top_trough_indx} 个交易日')
                    return self.get_phase_max_return_or_loss(reverse_points_data, point_type=point_type)
                else:
                    logging.warning(f'-------> 从【起始点】开始计算【最大回撤】')
                    logging.warning(f'-------> 最大回撤: {s2trough_loss}, 持续 {top_trough_indx} 个交易日')
                    detecter_result = {
                        'max_return': s2trough_loss,
                        'return_days': top_trough_indx,
                        'start_indx': 0,
                        'end_indx': int(top_trough_indx),
                        'return_type': 's2MaxTrough',
                        }
                    return detecter_result

            # 从最大收益点开始, 计算最大回撤
            elif base_point == 'peak':
                # 从最大收益点之后的数据，获取新的最大回撤点
                after_peak_cond = reverse_points_data['indx'] >= top_peak_indx
                reverse_points_data_after_peak = reverse_points_data.loc[after_peak_cond, :].copy()

                if len(reverse_points_data_after_peak) == 1:
                    logging.warning(f'-------> 该序列最高点即是当前最新的数据点, 已创新高!!! 从【最大收益】-> 【最大回撤】的结果为 0')
                    detecter_result = {
                        'max_return': 0,
                        'return_days': 0,
                        'start_indx': 0,
                        'end_indx': 0,
                        'return_type': 'stillPeakStatus',
                        }
                    return detecter_result

                logging.warning(f'-------> 统计从【最高收益】 -> 【最大回撤】的结果表:\n{reverse_points_data_after_peak}')

                top_trough_indx_after_peak = self.get_top_point_indx(reverse_points_data_after_peak, point_type='trough')
                top_trough_loss_after_peak = reverse_points_data_after_peak['sum_chg'].min()

                peak2trough_loss = top_trough_loss_after_peak - s2peak_return
                loss_days = top_trough_indx_after_peak - top_peak_indx
                logging.warning(f'-------> 从【最高点】开始计算【最大回撤】')
                logging.warning(f'-------> 最大回撤: {peak2trough_loss}, 持续 {loss_days} 个交易日')
                detecter_result = {
                    'max_return': peak2trough_loss,
                    'return_days': loss_days,
                    'start_indx': int(top_peak_indx),
                    'end_indx': int(top_trough_indx_after_peak),
                    'return_type': 'maxPeak2maxTrough',
                    }
                return detecter_result

            else:
                logging.warning(f'-------> get_max_return_or_loss(point_type="trough") 模式下, base_point 参数可选值: ["start", "peak"]')
                raise Exception

        else:
            raise Exception


    def from_last_point_2_curr_return(self, reverse_points_data: pd.DataFrame):
        '''
        Desc:
            从最后一个 reverse point 点计算到当前的收益
        '''
        reverse_points_data_f = reverse_points_data.iloc[-1:, :].copy()
        if len(reverse_points_data_f) != 1:
            raise Exception

        last_point_type = reverse_points_data_f['type'].max()

        if last_point_type == 'peak':
            return_type = 'forePeak2curr'
        elif last_point_type == 'trough':
            return_type = 'foreTrough2curr'

        point2curr_return = self.Ei[-1] - reverse_points_data_f['sum_chg'].max()
        start_indx = int(reverse_points_data_f['indx'].max())
        end_indx = len(self.Ei) - 1
        detecter_result = {
            'max_return': point2curr_return,
            'return_days': end_indx - start_indx,
            'start_indx': start_indx,
            'end_indx': end_indx,
            'return_type': return_type,
            }
        return detecter_result

    def from_named_point_2_curr_return(self, reverse_points_data: pd.DataFrame, point_type: str):
        '''
        Desc:
            从最后一个 point 点到 curr 的累计收益
        Args:
            point_type:
                options: ["peak", "trough"]
        '''
        named_point_cond = reverse_points_data['type'] == point_type
        named_point_reverse_data = reverse_points_data.loc[named_point_cond].copy()

        if len(named_point_reverse_data) == 0:
            logging.warning(f'-------> 序列中没有 {point_type} 点, 默认从起始点开始 ...')
            base_return = 0
            start_indx = 0
        else:
            # 只有一条记录，不需要指定 stat_mothod 方法
            reverse_points_data_f = named_point_reverse_data.iloc[-1:, :]
            base_return = reverse_points_data_f['sum_chg'].min()
            start_indx = int(reverse_points_data_f['indx'].min())

        end_indx = len(self.Ei) - 1
        detecter_result = {
            'max_return': self.Ei[-1] - base_return,
            'return_days': end_indx - start_indx,
            'start_indx': start_indx,
            'end_indx': end_indx,
            'return_type': f'last{point_type.capitalize()}2curr',
            }
        return detecter_result

    def from_max_point_2_curr_return(self, reverse_points_data: pd.DataFrame, point_type: str):
        '''
        Desc:
            从最大的回撤、收益点到当前的累计收益
        Args:
            point_type:
                optinos: ["max_peak", "max_trough"]
        '''
        if point_type not in ["max_peak", "max_trough"]:
            logging.warning(f'-------> Arg "point_type" Parame Options: ["max_peak", "max_trough"]')
            raise Exception

        named_point_type = point_type.split('_')[1].lower()

        stat_method_map = {
            'peak': np.max,
            'trough': np.min,
            }
        stat_method = stat_method_map[named_point_type]

        max_point_type_cond = reverse_points_data['type'] == named_point_type
        reverse_points_data_f = reverse_points_data.loc[max_point_type_cond].copy()

        if len(reverse_points_data_f) == 0:
            logging.warning(f'-------> 序列中没有 {point_type} 点, 默认从起始点开始 ...')
            base_return = 0
            start_indx = 0
        else:
            base_return = stat_method(reverse_points_data_f['sum_chg'].tolist())
            max_point_row_cond = reverse_points_data_f['sum_chg'] == base_return
            start_indx = reverse_points_data_f.loc[max_point_row_cond, 'indx'].max()

        end_indx = len(self.Ei) - 1
        detecter_result = {
            'max_return': self.Ei[-1] - base_return,
            'return_days': end_indx - start_indx,
            'start_indx': start_indx,
            'end_indx': end_indx,
            'return_type': f'max{named_point_type.capitalize()}2curr',
            }
        return detecter_result

    def curr_return_base_hist_reverse_point(self, base_point='default'):
        '''
        Desc:
            统计从当前点的上一个回撤、或收益点以来的累计收益
        Args:
            base_point: 上一个 base 极值点的类型
                1. default: 不特别指定, 即，从当前点的上一点(可能是 peak / trough)计算累计收益
                2. peak: 从最后一个 peak 点计算累计收益
                3. trough: 从最后一个 trough 点计算累计收益
                4. max_peak: 从"最高收益点"开始
                5. max_trough: 从"最大回撤点"开始
        '''
        if not self.reverse_points:
            logging.warning(f'-------> self.reverse_points 还没有完成赋值')
            raise Exception

        reverse_points_data = pd.DataFrame(self.reverse_points)
        if base_point == 'default':
            logging.warning(f'-------> 计算从上一个 reverse point(可能 peak 或 trough 点) 到 current 点的累计收益')
            detect_result = self.from_last_point_2_curr_return(reverse_points_data)
        elif base_point in ['peak', 'trough']:
            logging.warning(f'-------> 计算从上一个 {base_point} 点到 current 点的累计收益')
            detect_result = self.from_named_point_2_curr_return(reverse_points_data, base_point)
        elif base_point in ['max_peak', 'max_trough']:
            logging.warning(f'-------> 计算从最大 {base_point} 点到 current 点的累计收益')
            detect_result = self.from_max_point_2_curr_return(reverse_points_data, base_point)
        else:
            logging.warning(f'-------> curr_return_base_hist_reverse_point() 未知计算模式')
            raise Exception

        logging.warning(f'-------> 回撤点、或最高收益点到 current 点的累计收益统计:')
        pprint(detect_result)
        return detect_result


def peak_trough_detect_table(fundcode, min_chg, drange=720, make_plot=False):
    '''
    Desc:
        提供 “Peak_and_Trough_Detecter” 回撤检测工具的使用教程
    Args:
        min_chg: 检测回撤点的最小波动幅度，债券基金一般为 1% 左右；股票基金一般为 3% 左右
    '''
    db_session = DB_Client('mysql_centos')
    fundnet_data = db_session.data_read(
        f'''
        select
             fsrq
            ,dwjz
        from fund.fund_networth_record_from_tt_web
        where 1=1
            and fundcode = '{fundcode}'
        order by
            fsrq
        ''')
    fundnet_data = fundnet_data.astype({'dwjz': 'float'})
    fundnet_data.tail()

    detect_results = []

    # %%
    # 获取基金的历史净值数据
    panel_data = fundnet_data.iloc[-drange:, :]
    date_list = panel_data['fsrq'].tolist()
    data_sdate = panel_data['fsrq'].min()
    fund_values = panel_data['dwjz'].tolist()
    fund_values = np.array(fund_values, dtype=float)

    # NOTE: 实例化 Peak_and_Trough_Detecter
    peak_trough_detecter = Peak_and_Trough_Detecter(min_chg)

    # NOTE: 含 point_type：起始到【最大】盈利点检测
    s2peak = peak_trough_detecter.fit(
        fund_values, point_type='peak', base_point='start', end_point='phase', make_plot=make_plot)
    print(f'Start s2peak:\n{s2peak}')
    detect_results.append(s2peak)

    # NOTE: 含 point_type：起始到【最大】回撤点检测
    s2trough = peak_trough_detecter.fit(
        fund_values, point_type='trough', base_point='start', end_point='phase', make_plot=False)
    print(f'Start s2trough:\n{s2trough}')
    detect_results.append(s2trough)

    # NOTE: 含 point_type：记录【最大】盈利点检测
    trough2peak = peak_trough_detecter.fit(
        fund_values, point_type='peak', base_point='trough', end_point='phase', make_plot=False)
    print(f'Max trough2peak:\n{trough2peak}')
    detect_results.append(trough2peak)

    # NOTE: 含 point_type：记录【最大】回撤点检测
    peak2trough = peak_trough_detecter.fit(
        fund_values, point_type='trough', base_point='peak', end_point='phase', make_plot=False)
    print(f'Max peak2trough:\n{peak2trough}')
    detect_results.append(peak2trough)

    # NOTE: 含 point_type：【阶段】盈利点检测
    trough2peak = peak_trough_detecter.fit(
        fund_values, point_type='peak', base_point='trough', end_point='phase', make_plot=False)
    print(f'Phase trough2peak:\n{trough2peak}')
    detect_results.append(trough2peak)

    # NOTE: 含 point_type：【阶段】回撤点检测
    peak2trough = peak_trough_detecter.fit(
        fund_values, point_type='trough', base_point='peak', end_point='phase', make_plot=False)
    print(f'Phase peak2trough:\n{peak2trough}')
    detect_results.append(peak2trough)

    # # %%
    # NOTE: 【最近最高点】【至今】的回撤
    peak2curr = peak_trough_detecter.fit(fund_values, base_point='peak', end_point='current', make_plot=False)
    print(f'Last peak2curr:\n {peak2curr}')
    detect_results.append(peak2curr)

    # NOTE: 【最近回撤点】【至今】的收益
    trough2curr = peak_trough_detecter.fit(fund_values, base_point='trough', end_point='current', make_plot=False)
    print(f'Last trough2curr:\n {trough2curr}')
    detect_results.append(trough2curr)

    # NOTE: 【最高收益点】【至今】的收益
    maxPeak2curr = peak_trough_detecter.fit(fund_values, base_point='max_peak', end_point='current', make_plot=False)
    print(f'Last maxPeak2curr:\n {maxPeak2curr}')
    detect_results.append(maxPeak2curr)

    # NOTE: 【最大回撤点】【至今】的收益
    maxTrough2curr = peak_trough_detecter.fit(fund_values, base_point='max_trough', end_point='current', make_plot=False)
    print(f'Last maxTrough2curr:\n {maxTrough2curr}')
    detect_results.append(maxTrough2curr)

    # # %%
    # NOTE: 返回反转点的索引点，可根据获取对应的日期（该表可以回答任意两个回撤、收益区间的收益统计）
    pprint(peak_trough_detecter.reverse_points)

    detect_table = pd.DataFrame(detect_results)
    detect_table['fundcode'] = fundcode
    detect_table['min_chg'] = min_chg
    detect_table['data_sdate'] = data_sdate

    detect_table['return_sdate'] = detect_table['start_indx'].map(lambda x: date_list[x])
    detect_table['return_edate'] = detect_table['end_indx'].map(lambda x: date_list[x])

    detect_table['etldate'] = time.strftime('%Y-%m-%d')
    print(detect_table)
    return detect_table



# %%
if __name__ == '__main__':
    peak_trough_detect_table('008798', 1/100, drange=720)