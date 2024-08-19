from mlops.ml_ops.base import AbstractMLOps
from mlops.tasks.cat import CatboostTask

import mlflow
from mlflow.models import infer_signature
from mlops.utils import mlflow_utils
from mlflow.client import MlflowClient

import logging
import shutil
from copy import copy
import ray
from typing import Union
import numpy as np
from catboost import Pool
import pandas as pd



class CatBoostOps(AbstractMLOps):
    def __init__(self, **kwargs):
        super(CatBoostOps, self).__init__(**kwargs)

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

        if type(test_data).__name__ == 'Pool':
            test_loader = test_data
            # 返回的是 input_size
            shape = test_data.shape
            input_examples = np.random.randint(0, 10, size=shape)
            label = np.random.randint(0, 10, size=shape[0])
            signature = infer_signature(
                input_examples[:5], label[:5], params_config)

        elif isinstance(test_data[0], np.ndarray):
            # 因为 np.ndarray 中不能设置文本分类变量
            test_loader = Pool(
                pd.DataFrame(test_data[0]), label=test_data[1], cat_features=params_config.get('categoric_features', None))
            X, y = test_data
            signature = infer_signature(X[:5], y[:5], params_config)

        else:
            test_loader = Pool(
            *test_data, cat_features=params_config.get('categoric_features', None))
            X, y = test_data
            signature = infer_signature(X[:5], y[:5], params_config)

        return test_loader, signature


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
