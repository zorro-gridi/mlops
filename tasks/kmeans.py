import numpy as np
import logging
import random
import time

from sklearn.cluster import KMeans
from sklearn import metrics
from torch import randint
from mlops.tasks.base import AbstractModelFactory

import mlflow
from mlflow.models import infer_signature



class kmeans_task(AbstractModelFactory):
    def __init__(self, **kwargs):
        super(kmeans_task, self).__init__(**kwargs)
        self.model_arch = 'kmeans'


    def train_job(self, data, max_clusters=30):
        silhouette_score_list = []
        estimators = []

        with mlflow.start_run(
            run_name=f"kmeans_train_job_{time.strftime('%Y-%m-%d %H:%M')}_{random.randint(1e3, 9e3)}"):
            logging.warning(f'start kmeans clusters......')
            for n in range(15, max_clusters):

                kmeans = KMeans(n_clusters=n, **self.model_init_params)
                kmeans.fit(data)

                cluster_distance = kmeans.inertia_
                # cluster_centers = kmeans.cluster_centers_
                silhouette_score = self.eval_job(kmeans, test_X=data, metric_name=self.model_eval_metric)
                silhouette_score_list.append(silhouette_score)

                mlflow.log_metric('cluster_distance', cluster_distance, step=n)
                mlflow.log_metric('silhouette_score', silhouette_score, step=n)

                estimators.append(kmeans)
                logging.warning(f'n_cluster: {n}, distance: {cluster_distance:,.0f}, {self.model_eval_metric}: {silhouette_score:,.6f}')

            # 理论上 score 是要一直下降的，找到一个突然上升的点的前一个聚类数，则是最优聚类数
            score_diff = [
                silhouette_score_list[i+1] - silhouette_score_list[i]
                for i in range(len(silhouette_score_list)-1)
                ]

            # 自动筛选最优聚类数的 estimator
            best_estimator_idx = np.argmax(score_diff)
            best_estimator = estimators[best_estimator_idx]

            return best_estimator


    def tune_job(self, *args, **kwargs):
        pass


    def eval_job(self, model_inst, test_X=None, test_y=None, metric_name=None):
        '''
        该函数返回模型的按指定评估指标计算的得分
        '''
        clustering_metrics = {
            'homogeneity_score': metrics.homogeneity_score,
            'completeness_score': metrics.completeness_score,
            'v_measure_score': metrics.v_measure_score,
            'adjusted_rand_score': metrics.adjusted_rand_score,
            'adjusted_mutual_info_score': metrics.adjusted_mutual_info_score,
        }

        if metric_name == 'silhouette_score':
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
