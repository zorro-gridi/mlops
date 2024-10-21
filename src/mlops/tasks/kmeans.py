import numpy as np
import logging
# import random
# import time

from sklearn.cluster import KMeans
from sklearn import metrics
from torch import randint
from mlops.tasks.base import AbstractModelFactory

# import mlflow
# from mlflow.models import infer_signature
# import ray
# from ray.air.integrations.mlflow import setup_mlflow


"""
@Desc:
    TODO: kmeans 的架构需要重新设计 !!!
"""



class kmeans_task(AbstractModelFactory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_arch = 'kmeans'


    def train_job(self, train_data, test_data, n_clusters=15):
        '''
        Desc:
            核心程序。启动 KMeans 聚类
        Args:
            n_clusters: 最大的聚类数量
        '''
        # 模型实例化的默认初始化参数
        default_init_params = dict(
            n_init=min(3, n_clusters),
            init='k-means++',
            )

        if self.model_init_params:
            default_init_params.update(self.model_init_params)

        km_estimator = KMeans(n_clusters=n_clusters, **default_init_params)
        km_estimator.fit(train_data)

        silhouette_score = self.eval_job(km_estimator, test_X=test_data, metric_name=self.model_eval_metric)
        return km_estimator, silhouette_score


    def tune_job(self, n_clusters: int, train_data=None, test_data=None, init_clusters=4):
        '''
        Desc:
            启动 train_func，开始聚类
        Args:
            init_clusters: kmeans 迭代的最大族数
            n_clusters: 搜索空间的目标分类簇数
        '''
        silhouette_score_list = []
        estimators = []

        logging.warning(f'start kmeans clusters......')
        # init_clusters 区别于 KMeans 自身的 init
        for n in range(init_clusters, n_clusters+1):
            kmeans, silhouette_score = self.train_job(train_data, test_data, n_clusters=n)
            silhouette_score_list.append(silhouette_score)
            estimators.append(kmeans)

            cluster_distance = kmeans.inertia_
            # cluster_centers = kmeans.cluster_centers_
            # mlflow.log_metric('cluster_distance', cluster_distance, step=n)
            # mlflow.log_metric('silhouette_score', silhouette_score, step=n)
            logging.warning(f'n_cluster: {n}, distance: {cluster_distance:,.0f}, {self.model_eval_metric}: {silhouette_score:,.6f}')

            # ===================================================================
            # 理论上 score 是要一直下降的，找到一个突然上升的点的前一个聚类数，则是最优聚类数
            # ===================================================================
            score_diff = [
                silhouette_score_list[i+1] - silhouette_score_list[i]
                for i in range(len(silhouette_score_list)-1)
                ]

        # 自动筛选最优聚类数的 estimator
        best_idx = np.argmax(score_diff)
        best_estimator = estimators[best_idx]
        cluster_loss = silhouette_score_list[best_idx]
        return {
            'training_loss': cluster_distance,
            self.model_eval_metric: cluster_loss,
            'best_model': best_estimator,
        }


    def eval_job(self, model_inst, test_X=None, test_y=None, metric_name=None):
        '''
        Desc:
            该函数返回模型的按指定评估指标计算的得分
        '''
        # 常见的聚类评估指标
        clustering_metrics = {
            'homogeneity_score': metrics.homogeneity_score,
            'completeness_score': metrics.completeness_score,
            'v_measure_score': metrics.v_measure_score,
            'adjusted_rand_score': metrics.adjusted_rand_score,
            'adjusted_mutual_info_score': metrics.adjusted_mutual_info_score,
        }

        if metric_name == 'silhouette_score' or metric_name is None:
            silhouette_score = metrics.silhouette_score(
                test_X,
                model_inst.labels_,
                metric='euclidean',
                sample_size=int(len(test_X) * 0.5),
                )
            return silhouette_score


        eval_score = clustering_metrics[metric_name](test_y, model_inst.labels_)
        return eval_score



    def test_job(self, model_inst, test_X=None, test_y=None):
        '''
        Desc:
            评估聚类的效果
        Return:
            eval_result: 统计各个聚类标签中正例的比例
        '''
        pred_labels = model_inst.predict(test_X)
        cluster_labels = np.unique(pred_labels)

        eval_result = {}

        for c_label in cluster_labels:
            cluster_y_label = test_y[np.where(pred_labels == c_label, True, False)]
            cluster_lens = len(cluster_y_label)
            truth_ratio = sum(cluster_y_label) / cluster_lens

            eval_result[c_label] = round(truth_ratio, 6)
            logging.warning(f'cluster label: {c_label}, truth ratio: {truth_ratio:,.5f}, samples: {cluster_lens}')

        return pred_labels, eval_result
