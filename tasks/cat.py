# from mlops.tasks.base import AbstractModelFactory
from base import AbstractModelFactory

from catboost import (
    CatBoost,
    Pool,
    # metrics,
    cv,
    )
import hyperopt

import random
import numpy as np
from functools import partial
import logging
from copy import copy


class CatboostTask(AbstractModelFactory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    # def train_job(self, config, train_datat, test_data):
    #     trial_model = CatBoost(
    #         l2_leaf_reg=int(config['l2_leaf_reg']),
    #         learning_rate=config['learning_rate'],
    #         loss_function=self.model_eval_metric,
    #         custom_metric=self.custom_loss_func,
    #         **self.model_train_params,
    #         )

    #     cv_pool = Pool(*train_datat)
    #     cv_data = cv(cv_pool, trial_model.get_params(), logging_level='Silent')
    #     # cv 返回交叉验证的评估指标
    #     best_loss = np.min(cv_data[f'test-{self.model_eval_metric}-mean'])
    #     return best_loss


    def train_job(self, config, train_datat, test_data):
        config['loss_function'] = self.model_eval_metric
        config['custom_metric'] = self.custom_loss_func

        model_params = copy(config)
        model_params.update(self.model_init_params)
        model_params.update(self.model_train_params)
        # CatBoost 接受字典类型参数
        trial_model = CatBoost(model_params)

        train_pool = Pool(*train_datat)
        test_pool = Pool(*test_data)
        trial_model.fit(train_pool, eval_set=test_pool)
        best_iteration = trial_model.get_best_iteration()
        best_score = trial_model.get_best_score()
        test_loss = best_score["validation"][self.model_eval_metric]

        logging.warning(f'CatBoost model test loss: {test_loss}')
        return test_loss


    def tune_job(self, params_space, train_data, test_data):
        trials = hyperopt.Trials()
        trial_func = partial(self.train_job, train_data=train_data, test_data=test_data)

        best_params = hyperopt.fmin(
            trial_func,
            space=params_space,
            algo=hyperopt.tpe.suggest,
            max_evals=30,
            trials=trials,
            rstate=np.random.default_rng(random.seed(42))
            )
        return best_params



if __name__ == '__main__':
    train_data = np.random.randint(0, 100, size=(100, 10))
    train_labels = np.random.randint(0, 2, size=(100))

    test_data = np.random.randint(0, 100, size=(50, 10))
    test_labels = np.random.randint(0, 2, size=(50))

    cat_task = CatboostTask(model_eval_metric='Logloss')

    config = {
        'l2_leaf_reg': 3.0,
        'learning_rate': 0.2,
        }

    train_datas = (train_data, train_labels)
    test_datas = (test_data, test_labels)
    cat_task.train_job(config, train_datas, test_datas)
