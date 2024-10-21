import random
import logging
import pickle
from pathlib import Path

from mlops.tasks.lstm_ts import lstmTsTask
from mlops.ml_ops.base import AbstractMLOps


import mlflow
from mlflow.models import infer_signature
from mlflow.client import MlflowClient

from ray import train, tune
import ray
import os
import shutil
import numpy as np

import torch
from torch import nn
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    )

from mlops.nn import train_utils
from mlops.utils import mlflow_utils
from typing import Union



class LstmOps(AbstractMLOps):
    def __init__(self, **kwargs):
        super(LstmOps, self).__init__(**kwargs)


    def run_data_args(self):
        pass


    def find_best_data_args(self):
        pass


    def run_model_args(self, ):
        pass


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

        test_loader = DataLoader(test_data, batch_size=1)
        test_data_sample = next(iter(test_loader))
        X, y = test_data_sample
        signature = infer_signature(X.numpy(), y.numpy(), params_config)
        return test_loader, signature


    def find_best_model_args(self, search_space, checkpoint_dir, **kwargs):
        # 删除已有文件夹
        if Path(checkpoint_dir).exists():
            shutil.rmtree(Path(checkpoint_dir).as_posix(), ignore_errors=True)
        os.makedirs(Path(checkpoint_dir), exist_ok=True)

        # 开始 ray tune
        trial_results = self.model_task.tune_job(
            search_space=search_space,
            train_data=self.train_data,
            test_data=self.test_data,
            checkpoint_dir=checkpoint_dir,
            **kwargs
            )

        eval_metric = kwargs.pop('custom_metric', None)
        report_metric_name = f'test_{self.model_task.model_eval_metric}' if not eval_metric else eval_metric
        # 获取最优训练模型参数
        best_result = trial_results.get_best_result(
            report_metric_name, mode=self.model_task.optimize_mode)

        best_config = best_result.config
        logging.warning(f'加载的最优训练结果: {best_result}')
        logging.warning(f'最优 config: {best_config}')

        if best_config.get('data_args_space', None):
            self.best_data_args.update(best_config['data_args_space'])
        if self.dataset_inst is not None:
            # 获取类的属性配置字典
            # mlflow 不支持保存的类型 torch.Datasets
            # 直接pop原始的mldel class，会剔除该属性
            # dataset_config.pop('dt_class', None)
            self.best_data_args.update(self.dataset_inst.__dict__)

        self.best_model_args = best_config.get('model_args_space', None)
        model_args = best_config['model_args_space']
        self.model_task.model_init_params.update(model_args)

        best_model_init_args = self.model_task.model_init_params
        logging.warning(f'model_init_params: {best_model_init_args}')
        dnn_model = self.model_task.nn_arch(**best_model_init_args)

        logdir = best_result.checkpoint.to_directory()
        checkpoint_path = Path(logdir) / self.model_task.model_checkpoint_name
        logging.warning(f'best model checkpoint: {checkpoint_path}')
        # 加载 checkpoint 中的 model
        model_state, optimizer_state = torch.load(checkpoint_path)
        dnn_model.load_state_dict(model_state)

        # 获取最佳实验的训练&测试指标
        test_loss = best_result.metrics[f'test_{self.model_task.model_eval_metric}']
        training_loss = best_result.metrics['training_loss']

        dnn_model.eval()
        checkpoint_inst = {
            'best_model': dnn_model,
            self.model_task.model_eval_metric: test_loss,
            'training_loss': training_loss,
            }

        logging.warning(f'mlops best_data_args result: {self.best_data_args}')
        logging.warning(f'mlops best_model_args result: {self.best_model_args}')
        return checkpoint_inst


    def save_checkpoint(self, *args, **kwargs):
        return super().save_checkpoint(*args, model_frame=mlflow.pytorch, **kwargs)
