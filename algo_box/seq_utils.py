import numpy as np
from sklearn.cluster import KMeans
import logging
import pandas as pd


def phase_series_point(data, start_point, n_clusters=3):
    '''
    Desc:
        本函数实现将一段序列聚类为几类关键点位。
    Args:
        data: 需要分段的序列
        start_point: 序列中需要聚类的点的起始位置
        n_clusters: phase point 聚类的个数
    Return:
        phase point: 序列中每个点对应的聚类类别
        label_centers: 每个点聚类的中心点集合
        sorted_label_centers: 经过排序的族类的中心点
    '''
    # start_point = len(p_data) - len(data)
    # 用于聚类分组的序列
    global point_kmeans_data
    point_kmeans_data = [
        np.array(data[i:i+start_point]).reshape(-1, 1)
        for i in range(len(data)-start_point)]

    # 只能在循环中实例化
    eatimator_list = [
        KMeans(n_clusters=n_clusters, n_init=2).fit(point_datas)
        for point_datas in point_kmeans_data]

    # 取每组的最后一点标签作为结果，因为参照的是近期的整体数据
    k_labels = np.array([eatimator.labels_[-1] for eatimator in eatimator_list])
    assert len(data) - start_point == len(k_labels)

    # 取每个聚类序列的中心点
    global label_centers
    label_centers = np.array([eatimator.cluster_centers_.squeeze() for eatimator in eatimator_list])
    # 取目标点的聚类标签
    point_label_center = [center[label] for center, label in zip(label_centers, k_labels)]

    # 通过当前点位的中心点，再中心点排序后，重新统一按照顺利重新分配标签
    point_phase = np.array([
        list(np.sort(center)).index(p) for center, p in zip(label_centers, point_label_center)
        ])
    # 返回排序后的标签中心点
    sorted_label_centers = np.array([np.sort(center) for center in label_centers])
    return point_phase, label_centers, sorted_label_centers



def chunk_series(datas: pd.DataFrame, checkpoint=0.1):
    '''
    Desc:
        将序列按照涨跌分组切块, data 中必须包含 "markup" 涨跌幅指标
    Args:
        data: 需要切分的序列
        checkpoint: 合并的阈值。abs(x) <= checkpoint 则合并到上一个分块序列
    '''
    data = datas.copy()
    markup_list = list(data['markup'])
    for i in range(1, len(markup_list)):
        last_sign = np.sign(markup_list[i-1])
        if abs(markup_list[i]) <= checkpoint:
            markup_list[i] = abs(markup_list[i]) * last_sign
    # markup 字段重新赋值
    data['markup'] = markup_list
    data['last_markup'] = data.sort_values(by=['trade_date'])['markup'].shift(1)

    data['is_split_point'] = [
        i if np.sign(x) != np.sign(y) else -1
        for i, (x, y) in enumerate(zip(data['markup'], data['last_markup']))]

    markup_list = list(data['markup'])
    split_idx_list = [i for i in data['is_split_point'] if i != -1]
    split_idx_list.append(len(data))

    # 涨跌分快序列
    global markup_chunk_list
    markup_chunk_list = [markup_list[split_idx_list[i]:split_idx_list[i+1]] for i in range(len(split_idx_list)-1)]

    # 将涨跌分快序列 markup_chunk_list 统计成连续涨跌天数
    markkup_days = [len(d) if sum(d) > 0 else -len(d) for d in markup_chunk_list]
    # markup_vol: 涨跌幅切分序列
    # markkup_days：涨跌天数统计序列
    markup_vol = [sum(d) for d in markup_chunk_list]
    return markup_vol, markkup_days
