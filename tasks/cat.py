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
        self.model_init_params['loss_function'] = self.model_eval_metric
        self.model_init_params['custom_metric'] = self.custom_loss_func

        baseline_params = {
            'random_seed': 42,
            'logging_level': 'Silent',
            # 返回 eval_set 上的最优模型
            'use_best_model': True,
            }
        self.model_init_params.update(baseline_params)

        earlystop_params = {
            'od_type': 'Iter',
            'od_wait': 40
            }
        self.model_init_params.update(earlystop_params)

        self.model_arch = 'cat'


    def train_job(self, config, train_data, test_data):
        model_params = copy(config)
        model_params.update(self.model_init_params)
        # CatBoost 接受字典类型参数
        trial_model = CatBoost(model_params)

        train_pool = Pool(*train_data)
        test_pool = Pool(*test_data)

        x_train, y_train = train_data
        x_test, y_test = test_data
        X = np.concatenate([x_train, x_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        cv_pool = Pool(X, y)

        # cat_features, ... 都可以写入 self.model_train_params
        trial_model.fit(train_pool, eval_set=test_pool, **self.model_train_params)
        # best_iteration = trial_model.get_best_iteration()
        best_score = trial_model.get_best_score()
        test_loss = best_score["validation"][self.model_eval_metric]
        cv_data = cv(cv_pool, trial_model.get_params(), logging_level='Silent')

        # cv 返回交叉验证的评估指标
        cv_loss = np.min(cv_data[f'test-{self.model_eval_metric}-mean'])
        if self.optimize_mode == 'max':
            cv_loss = 1 / cv_loss
        logging.warning(f'CatBoost model test loss: {test_loss:,.6f}, cv loss: {cv_loss:,.6f}')
        return cv_loss


    def tune_job(self, params_space, train_data, test_data, max_evals=50):
        trials = hyperopt.Trials()

        trial_func = partial(self.train_job, train_data=train_data, test_data=test_data)
        best_params = hyperopt.fmin(
            trial_func,
            space=params_space,
            algo=hyperopt.tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(random.seed(42))
            )

        # best_params = hyperopt.space_eval(params_space, best_params)
        logging.warning(f'CatBoost best tune model params: {best_params}')

        self.model_init_params.update(best_params)
        best_model = CatBoost(self.model_init_params)

        train_pool = Pool(*train_data)
        test_pool = Pool(*test_data)

        best_model.fit(train_pool, eval_set=test_pool, **self.model_train_params)
        test_loss = self.test_job(best_model, test_pool)

        return {
            'best_model': best_model,
            self.model_eval_metric: test_loss,
            }


    def test_job(self, model: CatBoost, test_pool):
        loss_result = model.eval_metrics(test_pool, [self.model_eval_metric])
        test_loss = loss_result[self.model_eval_metric][0]
        return test_loss



if __name__ == '__main__':
    train_data = np.random.randint(0, 100, size=(100, 10))
    train_labels = np.random.randint(0, 2, size=(100))

    test_data = np.random.randint(0, 100, size=(50, 10))
    test_labels = np.random.randint(0, 2, size=(50))

    cat_task = CatboostTask(model_eval_metric='Logloss')

    params_space = {
        'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 0, 2, 1),
        'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
        }

    train_datas = (train_data, train_labels)
    test_datas = (test_data, test_labels)
    cat_task.tune_job(params_space, train_datas, test_datas)
