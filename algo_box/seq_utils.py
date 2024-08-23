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
