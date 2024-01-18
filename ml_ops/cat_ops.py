from mlops.ml_ops.base import AbstractMLOps
from mlops.tasks.cat import CatboostTask

import mlflow
from mlflow.models import infer_signature
from mlops.utils import mlflow_utils
from mlflow.client import MlflowClient


import logging
import shutil
from copy import copy


class XgboostOps(AbstractMLOps):
    def __init__(self, **kwargs):
        super(XgboostOps, self).__init__(**kwargs)


    def find_best_data_args(self, params_space):
        best_checkpoint = self.model_task.tune_job(
            params_space,
            train_data=self.train_data,
            test_data=self.test_data,
            )

        self.best_model_args = copy(self.model_task.model_init_params)
        return best_checkpoint
