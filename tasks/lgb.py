import os
import sys
sys.path.append(os.getcwd())

from mlops.tasks.base import AbstractModelFactory
from mlops.baseConfig.raytuneConfig import scaling_config

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


class LigthGBM_Task(AbstractModelFactory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_init_params['objective'] = self.model_loss_func
        self.model_init_params['metric'] = self.model_eval_metric

        self.model_arch = 'lgb'

        self.metircs_fn = {
            'rmse': mean_squared_error,
            'auc': roc_auc_score,
            }
        self.checkpoint_model_name = 'lightgbm.txt'


    def train_job(self, config, train__data, test_data, checkpoint=False):
        self.model_init_params.update(config)

        train_set = lgb.Dataset(*train__data)
        test_set = lgb.Dataset(*test_data)

        gbm = lgb.train(
            config,
            train_set=train_set,
            num_boost_round=100,
            valid_sets=[test_set],
            valid_names=['test'],
            # Training until validation scores don't improve for [stopping_rounds] rounds will stop
            callbacks=[lgb.early_stopping(stopping_rounds=10)],
            **self.model_train_params,
            )

        test_loss = self.test_job(gbm, test_data)
        logging.warning(f'LightGBM model test loss: {test_loss}')

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


    # def tune_job(self, *args, **kwargs):
    #     return super().tune_job(*args, **kwargs)


    def tune_job(self, config, train_data, test_data):
        tuner = tune.Tuner(
            partial(self.train_job, train_data=train_data, test_data=test_data),
            tune_config=tune.TuneConfig(
                metric="rmse",
                mode="min",
                scheduler=ASHAScheduler(),
                num_samples=2,
            ),
            param_space=config,
        )
        results = tuner.fit()


    def test_job(self, model: lgb.Booster, test_data):
        '''
        # test_data: can't use lgb.Dataset object
        '''
        x_test, y_test = test_data
        y_pred = model.predict(x_test, num_iteration=model.best_iteration)
        # 使用 sklearn metric 接口计算损失指标
        test_loss = self.metircs_fn[self.model_eval_metric](y_test, y_pred)
        return test_loss



if __name__ == '__main__':
    x_train = np.random.randint(0, 20, size=(100, 20))
    y_train = np.random.randint(0, 100, size=(100,))

    x_test = np.random.randint(0, 20, size=(20, 20))
    y_test = np.random.randint(0, 100, size=(20,))

    train_data = (x_train, y_train)
    test_data = (x_test, y_test)

    config = {
        "num_leaves": 10,
        "learning_rate": 0.2,
        }

    task_config = {
        'model_eval_metric': 'rmse',
        'model_loss_func': 'regression'
        }

    lgb_task = LigthGBM_Task(**task_config)
    lgb_task.tune_job(config, train_data, test_data)