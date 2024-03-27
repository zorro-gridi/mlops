import numpy as np
from sklearn.cluster import KMeans
import logging
import pandas as pd
from typing import Union


def phase_series_point(data: Union[pd.Series, list, np.array], start_point, n_clusters=3):
    '''
    Desc:
        本函数实现将一段序列聚类为几类关键点位。
    Args:
        data: 需要分段的序列
        start_point: 序列中需要聚类的点的起始位置
        n_clusters: phase point 聚类的个数
    Return:
        phase point: 序列中每个点对应的聚类类别
        label_centers: 每个聚类的中心点集合
        sorted_label_centers: 经过排序的族类的中心点
    '''
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
