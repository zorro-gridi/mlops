from mlops.tasks.base import AbstractModelFactory
from mlops.baseConfig.raytuneConfig import scaling_config
from base import AbstractModelFactory


import lightgbm as lgb
import logging
import numpy as np
from functools import partial
import os

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


    def train_job(self, config, train__data, test_data):

        self.model_init_params.update(config)
        train_set = lgb.Dataset(*train__data)
        test_set = lgb.Dataset(*test_data)

        gbm = lgb.train(
            config,
            train_set,
            num_boost_round=100,
            valid_sets=[test_set],
            valid_names=['test'],
            **self.model_train_params
            )

        test_loss = self.test_job(gbm, test_set)

        # 单独建立一个模型的隔离文件夹
        os.makedirs('models', exist_ok=True)
        gbm.save_model(f'models/{self.checkpoint_model_name}')

        report_checkpoint = train.Checkpoint.from_directory('models')
        train.report(
            metrics={f'test_{self.model_eval_metric}': test_loss},
            checkpoint=report_checkpoint
            )


    def tune_job(self, params_space, train_data, test_data, num_samples=10, checkpoint_dir=None):
        scheduler = ASHAScheduler(
            metric=f"test_{self.model_eval_metric}", mode=self.optimize_mode, max_t=10)

        checkpoint_strategy = train.CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute=f"test_{self.model_eval_metric}",
            checkpoint_score_order=self.optimize_mode,
            )
        run_config = train.RunConfig(
            stop={"training_iteration": 10},
            checkpoint_config=checkpoint_strategy,
            storage_path=checkpoint_dir,
            )

        train_partial = partial(
            self.train_job,
            train_data=train_data,
            test_data=test_data,
            checkpoint_dir=checkpoint_dir,
            )
        train_with_resources = tune.with_resources(train_partial, resources=scaling_config)

        tune_config = tune.TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler,
            )

        tuner = tune.Tuner(
            train_with_resources,
            param_space=params_space,
            tune_config=tune_config,
            run_config=run_config,
            )

        results = tuner.fit()
        return results


    def test_job(self, model: lgb.Booster, test_data: lgb.Dataset):
        y_pred = model.predict(test_data, num_iteration=model.best_iteration)
        test_loss = self.metircs_fn[self.model_eval_metric](y_test, y_pred)
        return test_loss



if __name__ == '__main__':
    x_train = np.random.randint(0, 100, size=(100, 20))
    y_train = np.random.randint(0, 2, size=(100,))

    x_test = np.random.randint(0, 100, size=(20, 20))
    y_test = np.random.randint(0, 2, size=(20,))

    train_data = (x_train, y_train)
    test_data = (x_test, y_test)


    search_space = {
        "boosting_type": tune.grid_search(["gbdt", "dart"]),
        "num_leaves": tune.randint(10, 1000),
        "learning_rate": tune.loguniform(1e-8, 1e-1),
        }

    task_config = {
        'model_eval_metric': 'auc',
        'model_loss_func': 'binary'
        }

    lgb_task = LigthGBM_Task(**task_config)

    checkpoint_dir = '/home/zorro/project/pycharm'
    lgb_task.tune_job(search_space, train_data, test_data, checkpoint_dir=checkpoint_dir)