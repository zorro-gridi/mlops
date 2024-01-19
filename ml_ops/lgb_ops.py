from sklearn import metrics
from mlops.ml_ops.base import AbstractMLOps
from mlops.tasks.cat import CatboostTask

import mlflow
from mlflow.models import infer_signature
from mlops.utils import mlflow_utils
from mlflow.client import MlflowClient


import logging
import shutil
from copy import copy
from pathlib import Path
import lightgbm as lgb


class LightGBM_Ops(AbstractMLOps):
    def __init__(self, **kwargs):
        super(LightGBM_Ops, self).__init__(**kwargs)


    def find_best_model_args(self, *args, **kwargs):
        return super().find_best_model_args(*args, **kwargs)



    def save_checkpoint(self, *args, **kwargs):
        return super().save_checkpoint(*args, **kwargs)