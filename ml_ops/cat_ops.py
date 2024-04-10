from mlops.ml_ops.base import AbstractMLOps
from mlops.tasks.cat import CatboostTask

import mlflow
from mlflow.models import infer_signature
from mlops.utils import mlflow_utils
from mlflow.client import MlflowClient


import logging
import shutil
from copy import copy


class CatBoostOps(AbstractMLOps):
    def __init__(self, **kwargs):
        super(CatBoostOps, self).__init__(**kwargs)


    def find_best_model_args(self, params_space, **kwargs):
        '''
        Desc:
            改写了父类方法，因为 catboost 未使用 ray[tune] 调参框架
            但是, find_best_model_args 的主要任务不能改变, 例如更新 best_data_args 和 best_model_args
        '''
        best_checkpoint = self.model_task.tune_job(
            params_space,
            train_data=self.train_data,
            test_data=self.test_data,
            **kwargs
            )
        # best_model_args 和 best_data_args 都要更新
        self.best_model_args = copy(self.model_task.model_init_params)
        if self.dataset_inst:
            self.best_data_args.update(copy(self.dataset_inst.__dict__))
        return best_checkpoint


    def save_checkpoint(self, *args, **kwargs):
        return super().save_checkpoint(*args, model_frame=mlflow.catboost, **kwargs)