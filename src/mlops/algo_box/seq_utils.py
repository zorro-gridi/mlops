# %%
import numpy as np
from sklearn.cluster import KMeans
import logging
import pandas as pd
from typing import Union
import time

import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np

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
            logging.warning(f'---------> Invalid args value! Args "base_point" Params Options: {base_point_options}')
            raise Exception

        end_point_options = ["phase", "current"]
        if end_point not in end_point_options:
            logging.warning(f'---------> Invalid args value! Args "end_point" Params Options: {end_point_options}')
            raise Exception

        # start_idx 迭代获取每一阶段的回撤、收益点
        while True:
            try:
                # NOTE: 回测、收益点需要从数据的其实点开始，一个个点的向右边扫描，每一轮都得到一个 gain_down_list，然后计算相应的点类型
                gain_down_list = self.cal_sequence_peak_and_trough_point(sequence, start_idx=self.last_top_point['indx'])
                self.gain_down_list = gain_down_list
                # NOTE: get_batch_top_point 方法会更新 peak_trough 类的属性
                top_point = self.get_batch_top_point(gain_down_list)
                logging.warning(f'---------> loop top_point: {top_point}')
            except:
                logging.warning(f'---------> gain_down_list 循环完成..., 准备作图')
                break

        detecter_result = None
        if end_point == 'phase':
            if base_point is None or point_type is None:
                logging.warning(f'---------> end_point 为 "phase" 的模式下, point_type 和 base_point 参数不能为空！')
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
        ax.scatter(len(self.Ei), self.Ei[-1], c='blue', s=15)

        for i, point in enumerate(self.reverse_points):
        # for unit-test
        # for i, point in enumerate(self.gain_down_list):
            if point['type'] == 'peak':
                color = 'r'
                # NOTE: <= 1 是因为 2 条记录，刚好是一个 peak/trough 周期。不限定的话，会导致 lengend 很多重复的 label
                label = 'peak' if i <= 1 else None
            else:
                color = 'g'
                label = 'trough' if i <= 1 else None

            ax.scatter(point['indx'], point['sum_chg'], c=color, label=label, s=15)

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
        last_point_sum_chg = None

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
                    last_point_sum_chg = Ei_peak

                    # 始终保持阶段的收益点为局部最大值，而Ei_peak需要迭代更新，目的是判断趋势是否持续上涨，从而判断 peak 反转点
                    if Ei_peak > phase_peak:
                        phase_peak = Ei_peak

                    # NOTE: 上一个点位的类型
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
                    last_point_sum_chg = Ei_trough

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
            将peak、trough 点 batch 批次化, 用于后续计算每个批次的最大收益、回撤点
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
        # 因为后面要分段，分别取每一段的最高，最低点，所以必须添加首尾的索引号
        reverse_indxs.insert(0, 0)
        reverse_indxs.append(len(gain_down_list))
        return reverse_indxs


    def find_the_top_point(self, point_list):
        '''
        Desc:
            寻找每个阶段最大回撤、收益批次里面的 top 点
        Args:
            point_list: 最大回撤、或最高收益批次点
        Return:
            top_point: 批次中最大的一个回撤、或收益点
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
            获取每一个批次的最大收益点、与回撤点
        Args:
            gain_down_list: 序列的初始检测的收益点和回撤点(未去重)
        Return:
            top_point: 过滤后的有效最大回撤、最大收益点记录列表
        '''
        if len(gain_down_list) == 1:
            logging.warning(f'-------> gain_down_list 只有 1 条回撤、或收益点记录: {gain_down_list[0]}, 直接返回')
            top_point = gain_down_list[0]
            self.reverse_indxs = top_point['indx']
        else:
            reverse_indxs = self.find_batch_point_indx(gain_down_list)
            self.reverse_indxs = reverse_indxs

            # version2: 为保证准确性，一次提取一个
            s_indx = reverse_indxs[0]
            e_indx = reverse_indxs[1]
            patch_phase = gain_down_list[s_indx:e_indx]
            top_point = self.find_the_top_point(patch_phase)

        # 定位当前 point_type 的索引号
        top_point['indx'] += self.last_top_point['indx']
        # 获取当前 point_type 从起始点的累计收益
        top_point['sum_chg'] = self.Ei[top_point['indx']]

        # 2025-04-16 新增：修复在迭代阶段的回撤、收益点时，如果前后两个点相同，清除相同点的情况。特别是在初始化开始时，因为只有一种 point 点，容易重复
        curr_point_type = top_point['type']
        curr_point_schg = top_point['sum_chg']

        if self.reverse_points:
            if all([
                # 记录的 reverse_points 的最后一个 point_type 与当前区间获取的 curr_point_type 相同
                self.reverse_points[-1]['type'] == curr_point_type,
                # 如果当前的 top_point 记录的累计收益率点位更低，则删除之前的记录
                abs(curr_point_schg) >= abs(self.reverse_points[-1]['sum_chg'])
                ]):
                self.reverse_points.pop()

        self.reverse_points.append(top_point)
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
        reverse_points_data_copy['trade_days'] = reverse_points_data_copy['next_indx'] - reverse_points_data_copy['indx']

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
        trade_days = max_diff_indx - start_indx

        detecter_result = {
            'max_return': round(max_diff, 4),
            'trade_days': trade_days,
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
            trade_days = detecter_result['trade_days']
            logging.warning(f'-------> {stat_name}: {phase_return}, 持续 {trade_days} 个交易日\n')
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
                    'max_return': round(s2peak_return, 2),
                    'trade_days': top_peak_indx,
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
                        'trade_days': 0,
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
                trade_days = top_peak_indx_after_trough - top_trough_indx

                logging.warning(f'-------> 从【最大回撤】开始计算【最高收益】')
                logging.warning(f'-------> 最大收益: {trough2peak_return}, 持续 {trade_days} 个交易日')
                detecter_result = {
                    'max_return': round(trough2peak_return, 2),
                    'trade_days': trade_days,
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
                    return self.get_phase_max_return_or_loss(point_type=point_type)
                else:
                    logging.warning(f'-------> 从【起始点】开始计算【最大回撤】')
                    logging.warning(f'-------> 最大回撤: {s2trough_loss}, 持续 {top_trough_indx} 个交易日')
                    detecter_result = {
                        'max_return': round(s2trough_loss, 2),
                        'trade_days': top_trough_indx,
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
                        'trade_days': 0,
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
                    'max_return': round(peak2trough_loss, 2),
                    'trade_days': loss_days,
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
            'max_return': round(point2curr_return, 2),
            'trade_days': end_indx - start_indx,
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
            'max_return': round(self.Ei[-1] - base_return, 2),
            'trade_days': end_indx - start_indx,
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
            'max_return': round(self.Ei[-1] - base_return, 2),
            'trade_days': end_indx - start_indx,
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

    def format_reverse_point(self):
        '''
        Desc:
            规整序列的 reverse_point, 同于统计各阶段回撤点到收益点、或收益点到回撤点的区间收益
        # TODO: Usage:
            1. 利用该回撤与收益点序列，可以分析序列的波动幅度情况，用于选取定投佳品！
            2. 利用回撤、与收益之间的连续关系，分析在收益一波后的潜在回撤，与回撤后的潜在收益分析，用于定投抄底、与逃顶!
        '''
        reverse_points = pd.DataFrame(self.reverse_points)
        reverse_points = reverse_points.astype({
            'sum_chg': float,
            })
        reverse_points = reverse_points.sort_values(by='indx', ascending=1)
        reverse_points['sum_chg'] = reverse_points['sum_chg'].round(3)
        # shift -1 表示取当前记录的下一个数
        reverse_points['to_point_indx'] = reverse_points['indx'].shift(-1)
        reverse_points['to_point_type'] = reverse_points['type'].shift(-1)
        reverse_points['to_point_schg'] = reverse_points['sum_chg'].shift(-1).round(3)
        # 计算回撤点 -> 收益点、或收益点 -> 回撤点的区间收益
        reverse_points['phase_return'] = (reverse_points['to_point_schg'] - reverse_points['sum_chg']).round(3)
        # 计算回撤后修复、回收益点回撤后的收益统计
        reverse_points['phase_return_repair'] = reverse_points['phase_return'].shift(-1).round(3)
        # NOTE: return_rho 验证回撤损失、与上涨收益之间的比例关系：即是否跌的多，反弹的也高
        # NOTE: 分析是否是涨的多，跌的就多；反之，类似
        reverse_points['return_rho'] = (reverse_points['phase_return_repair'] / reverse_points['phase_return']).abs().round(2)
        return reverse_points

    def get_current_stage(self):
        '''
        Desc:
            返回序列最新的点状态: 1. 回撤修复(上涨阶段); 2. 收益回撤(下跌阶段）
        Return:
            current_state: 当前的点位状态
            current_state_return_name: 当前状态的累计收益
        '''
        reverse_points = self.format_reverse_point()
        current_state = reverse_points['type'].iloc[-1]
        # 达到顶点，意味着下跌；达到低点，意味着上涨
        current_state_return_map = {
            'peak': 'lastTrough2curr',
            'trough': 'lastPeak2curr',
            }
        current_state_return_name = current_state_return_map[current_state]
        return current_state, current_state_return_name

    def peak2trough_analysis(self):
        '''
        Desc:
            1. 分析"从收益点、到回撤点"的【区间回撤】统计分布
            2. 分析"从收益点回撤后"的”【潜在收益】区间“
        '''
        reverse_points = self.format_reverse_point()
        # 回撤点到收益点的统计分析
        peak2trough_cond = reverse_points['type'] == 'peak'
        peak2trough_returns = reverse_points.loc[peak2trough_cond]
        return peak2trough_returns

    def trough2peak_analysis(self):
        '''
        Desc:
            1. 分析"回撤点、到收益点“的【区间收益】统计分布
            2. 分析”从回撤点修复后“的”【潜在回撤】区间“
        '''
        reverse_points = self.format_reverse_point()
        # 回撤点到收益点的统计分析
        trough2peak_cond = reverse_points['type'] == 'trough'
        trough2peak_returns = reverse_points.loc[trough2peak_cond]
        return trough2peak_returns

    def volatility_analysis(self, exclude_num=3):
        '''
        Desc:
            基于“回撤、收益”上下震荡的波动性分析，用于选择定投标的；同时，也可以作为定投组合分析的指标参考
        Args:
            exclude_num: 剔除序列中，首、尾极值的个数，用于避免极值对平均数据的影响
        Method:
            1. 统计相同序列区间内，基于最小波动 "min_chg" 的回撤点、收益点个数统计
            2. 使用收益上下波动的区间范围
        Return:
            1. trou_peak_pnum: 序列的“收益点、与回撤点”的总数
            2. phase_return_avg: 收益点、回撤点之间的上下波动的平均收益率; 注意，这个收益率加了绝对值
        '''
        reverse_points = self.format_reverse_point()
        trou_peak_pnum = len(reverse_points)

        # TODO: 这里可以考虑回撤修复收益、与顶点回撤损失分开计算
        phase_return_req = sorted(reverse_points['phase_return'].tolist())[exclude_num:-exclude_num]
        phase_return_avg = round(np.abs(phase_return_req).mean(), 3)
        return trou_peak_pnum, phase_return_avg


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
    # NOTE: 含 point_type：【阶段】盈利点检测
    # NOTE: 返回反转点的索引点，可根据获取对应的日期（该表可以回答任意两个回撤、收益区间的收益统计）
    reverse_points = peak_trough_detecter.format_reverse_point()
    reverse_points['fundcode'] = fundcode
    reverse_points['min_chg'] = min_chg
    reverse_points['drange'] = drange
    logging.warning(f'---------> 基金 {fundcode} 的净值回撤点、与收益点记录:')
    print(reverse_points)

    db_session.run_ddl(f'delete from fund.fund_trough_peak_point_record where fundcode = "{fundcode}"')
    db_session.data_load(
        df=reverse_points,
        schema='fund',
        table_name='fund_trough_peak_point_record',
        primary_key='indx, fundcode',
        operation='append',
        verbose=1,
        )

    # NOTE: 回撤点、到收益点的区间收益统计分析
    trough2peak_returns = peak_trough_detecter.trough2peak_analysis()
    print('回撤修复的阶段收益统计：')
    print(trough2peak_returns)

    # NOTE: 收益点、到回撤点的区间收益统计分析
    peak2trough_returns = peak_trough_detecter.peak2trough_analysis()
    print('收益回撤的阶段回撤统计：')
    print(peak2trough_returns)

    # NOTE: 基于“回撤、收益”点的序列波动性分析
    trou_peak_pnum, phase_return_avg = peak_trough_detecter.volatility_analysis(exclude_num=1)
    fundcode_vibration_and_phase_return_info = pd.DataFrame([{
        'trou_peak_pnum': trou_peak_pnum,
        'phase_return_avg': phase_return_avg,
        'drange': drange,
        'fundcode': fundcode,
        'etldate': time.strftime('%Y-%m-%d'),
        }])

    for conn in ['mysql_centos', 'pg_tencent']:
        db_client = DB_Client(conn)
        db_client.data_load(
            df=fundcode_vibration_and_phase_return_info,
            schema='report',
            table_name='fundcode_vibration_and_phase_return_info',
            operation='append',
            primary_key='fundcode',
            )
    logging.warning(f'序列的回撤点、与收益点个数（即收益上下振荡次数）: {trou_peak_pnum}, 区间平均波动: {phase_return_avg}')

    # 整合“回撤点、收益点波动转换”的收益统计表
    detect_table = pd.DataFrame(detect_results)
    detect_table['fundcode'] = fundcode
    detect_table['min_chg'] = min_chg
    detect_table['data_sdate'] = data_sdate

    detect_table['return_sdate'] = detect_table['start_indx'].map(lambda x: date_list[x])
    detect_table['return_edate'] = detect_table['end_indx'].map(lambda x: date_list[x])
    detect_table['calender_days'] = (pd.to_datetime(detect_table['return_edate']) - pd.to_datetime(detect_table['return_sdate'])).dt.days

    detect_table['phase_er'] = [
        round(np.power(1+y, 1 / (d/365)) - 1, 4)
        if d > 0 else y
        for y, d in zip(detect_table['max_return'], detect_table['calender_days'])
        ]

    # 通过分析得出结论：牛短熊长！！！
    detect_table['etldate'] = time.strftime('%Y-%m-%d')
    logging.warning(f'基金 {fundcode} 的净值回撤 Trough 点、与最高 Peak 点的区间收益统计:')
    print(detect_table)

    db_session.run_ddl(f'delete from fund.fund_trough_peak_phase_return_stats where fundcode = "{fundcode}"')
    db_session.data_load(
        df=detect_table,
        schema='fund',
        table_name='fund_trough_peak_phase_return_stats',
        primary_key='fundcode, return_type',
        operation='append',
        verbose=1,
        )

    # NOTE: 分析当前【回撤修复收益】的百分位数
    lastTough2curr_yield = detect_table.loc[detect_table['return_type'] == 'lastTrough2curr', 'max_return'].max()
    # 统计（trough2peak_returns）当前【累计收益情况】下，后续的【潜在回撤范围】，取当前累计收益的 0.75 ～ 1.25 的范围
    potential_trough_scope = (
        (trough2peak_returns['phase_return'] >= lastTough2curr_yield * 0.75) &
        (trough2peak_returns['phase_return'] <= lastTough2curr_yield * 1.25)
        )
    trough2peak_returns_scope = trough2peak_returns.loc[potential_trough_scope, 'phase_return_repair'].copy()
    # NOTE: 最小、最大潜在回撤 (因为回撤是负值)（当极值为 nan 时，表示收益可能才刚刚开始）
    min_p_trough = trough2peak_returns_scope.max()
    max_p_trough = trough2peak_returns_scope.min()
    potential_trough_range = (min_p_trough, max_p_trough)
    # 统计当前累计收益的历史分位数
    # trough2curr_percentile = utils.calculate_quantile(lastTough2curr_yield, trough2peak_returns['phase_return'].tolist())
    trough2curr_percentile = round(lastTough2curr_yield / phase_return_avg * 100, 2)

    # NOTE: 分析当前【高点回撤损失】的百分位数
    lastPeak2curr_yield = detect_table.loc[detect_table['return_type'] == 'lastPeak2curr', 'max_return'].max()
    # 统计（peak2trough_returns）当前【累计回撤情况】下，后续的【潜在收益范围】，取当前累计回撤的 0.75 ～ 1.25 的范围
    potential_peak_scope = (
        (peak2trough_returns['phase_return'] >= lastPeak2curr_yield * 0.75) &
        (peak2trough_returns['phase_return'] <= lastPeak2curr_yield * 1.25)
        )
    peak2trough_returns_scope = peak2trough_returns.loc[potential_peak_scope, 'phase_return_repair'].copy()
    # 最小、最大潜在收益 （当极值为 nan 时，表示回撤可能才刚刚开始！！！）
    min_p_peak = peak2trough_returns_scope.min()
    max_p_peak = peak2trough_returns_scope.max()
    potential_peak_range = (min_p_peak, max_p_peak)

    # 因为回撤是负值，所以分位数位为： 1 - percentile
    # peak2curr_percentile = round(1 - utils.calculate_quantile(lastPeak2curr_yield, peak2trough_returns['phase_return'].tolist()), 4)
    peak2curr_percentile = round(1 - abs(lastPeak2curr_yield) / phase_return_avg * 100, 2)

    # 判断当前是处于“回撤修复、或收益回撤”阶段
    current_stage, curr_return_name = peak_trough_detecter.get_current_stage()
    logging.warning(f'基金 {fundcode} 当前处于: 【{current_stage}】阶段!')

    # NOTE: 对于债券基金来说，因为走势整体基本是向上的，因此，阶段回撤修复收益占历史阶段收益的百分位数可能经常出现 0% 的情况
    # 因此，阶段收益百分位数指标可能不适合分析债券基金
    if current_stage == 'peak':
        logging.warning(f'当前【回撤修复】的 {curr_return_name} 累计收益: {lastTough2curr_yield:0.4f}，收益百分位数: {trough2curr_percentile}%; 后续潜在[回撤范围]: {potential_trough_range}')
    else:
        logging.warning(f'当前【最高收益】的 {curr_return_name} 累计回撤: {lastPeak2curr_yield:0.4f}，历史百分位数: {peak2curr_percentile}%; 后续潜在[收益范围]: {potential_peak_range}')





# %%
if __name__ == '__main__':
    min_chg_map = {
        '008798': 1/100,
        '005176': 3/100,
        '013074': 3/100,
        '010573': 3/100,
        '001230': 3/100,
        }
    fundcode = '001230'
    peak_trough_detect_table(fundcode, min_chg_map[fundcode], drange=720, make_plot=True)