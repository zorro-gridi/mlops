
import lightgbm as lgb
import logging
import numpy as np
from functools import partial

import random
import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

from sklearn.metrics import (
    mean_squared_error,
    mean_squared_log_error,
    accuracy_score,
    roc_auc_score,
    f1_score,
    )

import sklearn.datasets
from sklearn.model_selection import train_test_split


import os
import sys
sys.path.append(os.getcwd())

from mlops.tasks.base import AbstractModelFactory



class LigthGBM_Task(AbstractModelFactory):
    '''
    # LightGBM 的问题：
        1. 好像无法在 train 中获取测试集的损失指标结果。 因此，使用 sklearn.metrics 接口计算指标
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_init_params['objective'] = self.model_loss_func
        self.model_init_params['metric'] = self.model_eval_metric
        self.model_init_params['verbosity'] = -1

        self.model_arch = 'lgb'

        self.metircs_fn = {
            'rmse': mean_squared_error,
            'auc': roc_auc_score,
            'acc': accuracy_score,
            'binary_logloss': roc_auc_score,
            }
        self.checkpoint_model_name = 'lightgbm.txt'


    def train_job(self, config, train_data, test_data, checkpoint=False):
        self.model_init_params.update(config)

        train_set = lgb.Dataset(*train_data)
        test_set = lgb.Dataset(*test_data)

        init_model = None
        gbm = lgb.train(
            self.model_init_params,
            train_set=train_set,
            num_boost_round=1000,
            valid_sets=[test_set],
            valid_names=['test'],
            # 如果数据集特别大，可以迭代训练
            init_model=init_model,
            # Training until validation scores don't improve for [stopping_rounds] rounds will stop
            callbacks=[lgb.early_stopping(stopping_rounds=10)],
            **self.model_train_params,
            )

        test_loss = self.test_job(gbm, test_data)
        logging.warning(f'LightGBM model test {self.model_eval_metric} loss: {test_loss:,.6f}')

        if checkpoint:
            return {
                'best_model': gbm,
                self.model_eval_metric: test_loss
                }

        # 单独建立一个模型的隔离文件夹
        os.makedirs('models', exist_ok=True)
        gbm.save_model(f'models/{self.checkpoint_model_name}', num_iteration=gbm.best_iteration)

        report_checkpoint = train.Checkpoint.from_directory('models')
        train.report(
            metrics={f'test_{self.model_eval_metric}': test_loss},
            checkpoint=report_checkpoint
            )


    def tune_job(self, *args, **kwargs):
        return super().tune_job(*args, **kwargs)


    def test_job(self, model: lgb.Booster, test_data):
        '''
        # test_data: can't use lgb.Dataset object
        # 注意：有些指标需要使用 y_pred_label 而不是 y_pred_proba
        '''
        x_test, y_test = test_data
        y_pred = model.predict(x_test, num_iteration=model.best_iteration)
        # 使用 sklearn metric 接口计算损失指标
        test_loss = self.metircs_fn[self.model_eval_metric](y_test, y_pred)
        return test_loss



if __name__ == '__main__':
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)

    train_data = (train_x, train_y)
    test_data = (test_x, test_y)


    config = {
        # "metric": ["binary_logloss"],
        "num_leaves": tune.randint(10, 1000),
        "learning_rate": tune.loguniform(1e-8, 1e-1),
    }

    task_config = {
        'model_eval_metric': 'auc',
        'model_loss_func': 'binary'
        }

    lgb_task = LigthGBM_Task(**task_config)
    lgb_task.tune_job(config, train_data=train_data, test_data=test_data)