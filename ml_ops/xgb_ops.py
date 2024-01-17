import random
import logging
import pickle
import xgboost as xgb
from pathlib import Path
import numpy as np

from mlops.ml_ops.base import AbstractMLOps
from mlops.tasks.xgb import xgboost_task


import mlflow
from mlflow.models import infer_signature
from mlops.utils import mlflow_utils
from mlflow.client import MlflowClient


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


    def find_best_model_args(self, model_args_space, checkpoint_dir=None, **kwargs):
        tune_results = self.model_task.tune_job(
            model_args_space,
            self.train_data,
            self.test_data,
            checkpoint_dir=checkpoint_dir,
            **kwargs
            )

        report_metric_name = f'test_{self.model_task.model_eval_metric}'
        best_result = tune_results.get_best_result(
            metric=report_metric_name, mode=self.model_task.optimize_mode)
        logging.warning(best_result)

        logdir = best_result.checkpoint.to_directory()
        # 最优模型文件
        checkpoint_path = Path(logdir) / self.model_task.checkpoint_model_name
        bset_xgb_model = xgb.Booster(model_file=checkpoint_path)

        eval_metric_name = self.model_task.model_eval_metric
        best_checkpoint = {
            'best_model': bset_xgb_model,
            eval_metric_name: best_result.metrics[f'test_{eval_metric_name}'],
            'best_iteration': best_result.metrics['best_iteration'],
            }

        # 更新 xgb 模型实例的最优调参结果
        self.best_model_args.update(best_result.config)
        if self.dataset_inst is not None:
            self.best_data_args.update(self.dataset_inst.__dict__)

        return best_checkpoint


    def save_checkpoint(self, *args, **kwargs):
        return super(XgboostOps, self).save_checkpoint(*args, **kwargs)