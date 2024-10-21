import os
import sys
sys.path.append(os.getcwd())

from mlops.tasks.base import AbstractModelFactory
from catboost import (
    CatBoost,
    Pool,
    # metrics,
    cv,
    )
import ray
import hyperopt
# 调参早停技术
from hyperopt.early_stop import no_progress_loss

import random
import numpy as np
from functools import partial
import logging
from copy import copy
import torch



class CatboostTask(AbstractModelFactory):
    def __init__(self, **kwargs):
        '''
        Args:
            model_init_params: 实例化 catbooost 的参数字典
            model_train_params: 传入 fit 函数的参数字典
            model_loss_func: for catboost "loss_function"
            custom_loss_func: 自定义的 loss_functions
            model_eval_metric: for catboost "eval_metric"
        '''
        super().__init__(**kwargs)
        self.model_init_params['loss_function'] = self.model_loss_func
        self.model_init_params['custom_metric'] = self.custom_loss_func
        self.model_init_params['eval_metric'] = self.model_eval_metric

        baseline_params = {
            'random_seed': 42,
            'logging_level': 'Silent',
            # 返回 eval_set 上的最优模型
            'use_best_model': True,
            }
        self.model_init_params.update(baseline_params)

        # catboost built-in 早停技术
        earlystop_params = {
            'od_type': 'Iter',
            'od_wait': 40
            }
        self.model_init_params.update(earlystop_params)
        self.model_arch = 'cat'


    def convert_data(self, train_data, test_data):
        '''
        Desc:
            判断输入数据的类型，并自动转换为模型需要的类型
        '''
        # 当有分类特征时，最好提前构造好 Pool 格式的数据集
        if type(train_data).__name__ == 'Pool':
            # logging.warning(f'train data alread be cat Pool type...')
            train_pool = train_data
            test_pool = test_data
        else:
            # pool 对象本身会分批次加载数据集进行模型训练
            # 此情况下不能在 fit 中指定 cat_features
            train_pool = Pool(*train_data)
            test_pool = Pool(*test_data)
        return train_pool, test_pool


    def train_job(self, config, train_data, test_data):
        '''
        Desc:
            启动训练 job
        Return:
            training_loss
        '''
        config_device = config.pop('device', 'CPU')
        model_params = copy(config)

        if not config_device:
            device = 'GPU' if torch.cuda.is_available() else 'CPU'
            model_params['task_type'] = device
            model_params['devices'] = [0]
        else:
            model_params['task_type'] = config_device

        self.model_init_params.update(model_params)
        # CatBoost 接受字典类型参数。实例化过程类似于网络模型: 模型结构参数和模型训练参数分开
        trial_model = CatBoost(self.model_init_params)

        train_pool, test_pool = self.convert_data(train_data, test_data)
        # 使用训练集作为交叉验证集
        cv_pool = train_pool

        # cat_features, ... 都可以写入 self.model_train_params
        trial_model.fit(train_pool, eval_set=test_pool, **self.model_train_params)
        # best_iteration = trial_model.get_best_iteration()
        cv_data = cv(cv_pool, trial_model.get_params(), logging_level='Silent')

        # cv 返回交叉验证的评估指标
        cv_loss = np.min(cv_data[f'test-{self.model_eval_metric}-mean'])
        if self.optimize_mode == 'max':
            cv_loss = np.max(cv_data[f'test-{self.model_eval_metric}-mean'])
        else:
            cv_loss = np.min(cv_data[f'test-{self.model_eval_metric}-mean'])
        # 精度控制
        return round(cv_loss, 6)


    def tune_job(self, params_space, train_data, test_data, max_evals=50, early_stop_round=10):
        '''
        Args:
            params_space: 搜索空间
            train_data:
            test_data:
            max_evals: 实验的数量, 类似 ray[tune] 的 num_samples
            early_stop_round: 早停技术。如果 n 轮后损失没有提升，则停止实验
        Return:
            training checkpoint info dict
        '''
        trials = hyperopt.Trials()

        trial_func = partial(self.train_job, train_data=train_data, test_data=test_data)
        best_params = hyperopt.fmin(
            trial_func,
            space=params_space,
            algo=hyperopt.tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            # 也可以自定义 stop fn
            early_stop_fn=no_progress_loss(early_stop_round),
            rstate=np.random.default_rng(random.seed(42))
            )

        # best_params = hyperopt.space_eval(params_space, best_params)
        logging.warning(f'CatBoost best tune model params: {best_params}')

        self.model_init_params.update(best_params)
        best_model = CatBoost(self.model_init_params)

        # catboost 类似于神经网络，fit 完之后，得到的就是最后的模型
        # 所以，应该可以使用数据集迭代训练
        train_pool, test_pool = self.convert_data(train_data, test_data)
        best_model.fit(train_pool, eval_set=test_pool, **self.model_train_params)

        # 获取测试集的损失
        best_score = best_model.get_best_score()
        logging.warning(f'Best training & testing score: {best_score}')
        test_loss = best_score["validation"][self.model_eval_metric]
        training_loss = best_score["learn"][self.model_eval_metric]

        return {
            'best_model': best_model,
            self.model_eval_metric: test_loss,
            'training_loss': training_loss,
            }


    def test_job(self, model: CatBoost, test_pool: Pool):
        '''
        Desc:
            返回指定模型在测试集上的测试指标
        Return:
            test_loss
        '''
        loss_result = model.eval_metrics(test_pool, [self.model_eval_metric])
        # loss_result 是一个字典或者 NamedTuple
        test_loss = loss_result[self.model_eval_metric][0]
        return test_loss


    def predict(self, *args, **kwargs):
        '''
        @Url: https://catboost.ai/en/docs/concepts/python-reference_catboost_predict#prediction_type
        '''
        pass



if __name__ == '__main__':
    # train_data = np.random.randint(0, 100, size=(100, 10))
    # train_labels = np.random.randint(0, 2, size=(100))

    # test_data = np.random.randint(0, 100, size=(50, 10))
    # test_labels = np.random.randint(0, 2, size=(50))

    # cat_task = CatboostTask(model_eval_metric='Logloss')

    # params_space = {
    #     'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 0, 2, 1),
    #     'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
    #     }

    # train_datas = (train_data, train_labels)
    # test_datas = (test_data, test_labels)
    # cat_task.tune_job(params_space, train_datas, test_datas)
    pass
