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
        best_checkpoint = self.model_task.tune_job(
            params_space,
            train_data=self.train_data,
            test_data=self.test_data,
            **kwargs
            )

        self.best_model_args = copy(self.model_task.model_init_params)
        return best_checkpoint

    def save_checkpoint(self, *args, **kwargs):
        return super().save_checkpoint(*args, model_frame=mlflow.catboost, **kwargs)