# import random
# import logging
# import pickle
import xgboost as xgb
# from pathlib import Path
# import numpy as np

from mlops.ml_ops.base import AbstractMLOps
# from mlops.tasks.xgb import xgboost_task


import mlflow
from mlflow.models import infer_signature
# from mlops.utils import mlflow_utils
# from mlflow.client import MlflowClient
# import ray
from typing import Union



class XgboostOps(AbstractMLOps):
    def __init__(self, **kwargs):
        super(XgboostOps, self).__init__(**kwargs)

    def run_data_args(self, *data_args, **kwargs):
        pass

    def find_best_data_args(self, *args, **kwargs):
        pass


    # 这一步确实没有必要
    def run_model_args(self, model_args, xgb_model=None):
        if xgb_model:
            dtest = xgb.DMatrix(*(self.test_data))
            return self.model_task.test_job(xgb_model, dtest)

        best_checkpoint = self.model_task.train_job(model_args, self.train_data, self.test_data, checkpoint=True)
        return best_checkpoint


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

        test_loader = xgb.DMatrix(*test_data)
        X, y = test_data
        signature = infer_signature(X[:5], y[:5], params_config)
        return test_loader, signature


    def find_best_model_args(self, params_space, **kwargs):
        return super().find_best_model_args(params_space, **kwargs)


    def save_checkpoint(self, *args, **kwargs):
        return super(XgboostOps, self).save_checkpoint(*args, model_frame=mlflow.xgboost, **kwargs)
