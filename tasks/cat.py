from mlops.tasks.base import AbstractModelFactory


from catboost import (
    CatBoostRegressor,
    Pool,
    # metrics,
    cv,
    )
import hyperopt

import random
import numpy as np
from functools import partial



class CatboostRegTask(AbstractModelFactory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def train_job(self, config, train_datat, test_data):
        trial_model = CatBoostRegressor(
            l2_leaf_reg=int(config['l2_leaf_reg']),
            learning_rate=config['learning_rate'],
            **self.model_train_params,
            )

        cv_pool = Pool(*train_datat)
        cv_data = cv(cv_pool, trial_model.get_params(), logging_level='Silent')
        # cv 返回交叉验证的评估指标
        best_loss = np.min(cv_data[f'test-{self.model_eval_metric}-mean'])
        return best_loss


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
