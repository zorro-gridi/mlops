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


    def data_util_map(self, test_data, params_config=Union[None, dict]):
        '''
        Args:
            test_data: 模型输入的的的 test_data
            params_config: mlflow signature 的 params 参数
        return:
            test_loader: 供 self.test_job 评估模型
            signature: 供 mlflow 注册模型
        '''
        if params_config:
            params_config = self.exclude_non_mlflow_param_type(params_config)

        test_loader = test_data
        X, y = test_data
        signature = infer_signature(X[:5], y[:5], params_config)
        return test_loader, signature


    def find_best_model_args(self, *args, **kwargs):
        return super().find_best_model_args(*args, **kwargs)



    def save_checkpoint(self, *args, **kwargs):
        return super().save_checkpoint(*args, model_frame=mlflow.lightgbm, **kwargs)
