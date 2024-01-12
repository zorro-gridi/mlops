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



class LstmOps(AbstractMLOps):
    def __init__(self, **kwargs):
        super(LstmOps, self).__init__(**kwargs)


    def run_data_args(self):
        pass


    def find_best_data_args(self):
        pass


    def run_model_args(self, ):
        pass


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

        # 获取最优训练模型参数
        best_result = trial_results.get_best_result(
            self.model_task.model_eval_metric, mode=self.model_task.optimize_mode)
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
        test_loss = best_result.metrics[self.model_task.model_eval_metric]

        dnn_model.eval()
        checkpoint_inst = {
            'best_model': dnn_model,
            self.model_task.model_eval_metric: test_loss,
            }

        logging.warning(f'mlops best_data_args result: {self.best_data_args}')
        logging.warning(f'mlops best_model_args result: {self.best_model_args}')
        return checkpoint_inst


    def save_checkpoint(self, *args, **kwargs):
        return super().save_checkpoint(*args, **kwargs)


    # def save_checkpoint(self, checkpoint_inst, reg_model_name, model_version='1', model_alias=None):
    #     best_model = checkpoint_inst['best_model']
    #     best_model.eval()
    #     test_loss = checkpoint_inst[self.model_task.model_eval_metric]

    #     mlflow_client = MlflowClient(mlflow.get_tracking_uri())
    #     # load by version: f'models:/{model_name}/{model_version}'
    #     # load by alias: f'models:/{model_name}@{alias}'
    #     hist_test_loss = np.inf
    #     # 首先判断 registerd model 是否存在
    #     if mlflow_utils.check_model_existence(reg_model_name):
    #         # 使用 pytorch 的 load_model 方法
    #         # 如果使用 mlflow.pyfunc.load_model，则需要统一使用 predict 接口方法
    #         # =============================================================
    #         loaded_model = mlflow.pytorch.load_model(f'models:/{reg_model_name}/{model_version}')
    #         hist_model_config = mlflow_utils.load_register_model_args(reg_model_name, model_version)

    #         if self.dataset_inst is None:
    #             logging.warning(f'无数据参数调参模式, 使用当前测试数据测试历史模型...')
    #             test_datasets = self.test_data
    #         else:
    #             test_datasets = self.dataset_inst.load_test_data(self.raw_data, inst_config=hist_model_config)

    #         hist_test_loader = DataLoader(test_datasets, batch_size=1)
    #         hist_test_loss = train_utils.test_func(loaded_model, hist_test_loader)

    #         if self.model_task.optimize_mode == 'max':
    #             compare_bools_result = -test_loss >= -hist_test_loss
    #         else:
    #             # 否则: 最小模式
    #             compare_bools_result = test_loss >= hist_test_loss

    #         # 默认使用最小化模式
    #         if compare_bools_result:
    #             logging.warning(f'''
    #                 hist model loss: {hist_test_loss},
    #                 new tuned model losss: {test_loss},
    #                 ''')
    #             logging.warning(f'Hist model is better, will do nothing !')
    #             return 'hist', loaded_model
    #         else:
    #             # delete existed registred model
    #             mlflow_client.delete_registered_model(reg_model_name)

    #     if self.dataset_inst is not None:
    #         self.dataset_inst.set_attr(self.best_data_args)

    #     logging.warning(f'没有注册的历史模型, 或者历史模型评分低......')
    #     test_loader = DataLoader(self.test_data, batch_size=1)
    #     test_data_sample = next(iter(test_loader))
    #     X, y = test_data_sample

    #     logging.warning(f'test data of X shape: {X.shape}')
    #     logging.warning(f'model best args: {self.best_model_args}')
    #     # best_model_args 是 mlops 实例的属性
    #     params_config = self.best_model_args
    #     # 合并模型和数据集的参数
    #     params_config.update(self.best_data_args)
    #     params_config = {k: v for k, v in params_config.items() if v is not None}

    #     # 以下数据类型 mlflow 不支持保存
    #     params_config.pop('preprocess_func')
    #     signature = infer_signature(X.numpy(), best_model(X).detach().numpy(), params_config)

    #     # 定义 mlflow run_name
    #     run_name = f'{self.best_data_args["model_arch"]}_best_model_and_config'
    #     with mlflow.start_run(run_name=run_name):
    #         # mlflow.log_params(best_config)
    #         logging.warning(f'''
    #             hist model loss: {hist_test_loss},
    #             new tuned model losss: {test_loss},
    #             ''')
    #         logging.warning(f'New model is better, update the mlflow registred model......')
    #         # 建议 log_model 不启动 mlflow.start_run()
    #         model_info = mlflow.pytorch.log_model(
    #             best_model,
    #             'models',
    #             signature=signature,
    #             registered_model_name=reg_model_name,
    #             )
    #         mlflow.log_params(self.best_model_args)
    #         mlflow.log_params(self.best_data_args)

    #         mlflow.log_metric(f'test_{self.model_task.model_eval_metric}', test_loss)
    #         mlflow_client.set_registered_model_alias(reg_model_name, model_alias, model_version)
    #         mlflow_client.set_registered_model_alias(reg_model_name, self.best_data_args["model_arch"], model_version)
    #         mlflow_client.set_registered_model_tag(reg_model_name, f'test_{self.model_task.model_eval_metric}', str(test_loss))
    #         # mlflow_client.set_registered_model_tag(reg_model_name, f'model_arch', self.best_data_args['model_arch'])
    #         return 'new', best_model
