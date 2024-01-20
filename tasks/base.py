from abc import ABCMeta, abstractclassmethod
from sklearn import metrics
import numpy as np
from pathlib import Path
from functools import partial
import shutil
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

from ray.air.integrations.mlflow import setup_mlflow, MLflowLoggerCallback
from ray.tune.logger.aim import AimLoggerCallback
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hebo import HEBOSearch
from ray.tune.search.hyperopt import HyperOptSearch

import os
import sys
sys.path.append(os.getcwd())

from mlops.baseConfig.raytuneConfig import scaling_config


class AbstractModelFactory(metaclass=ABCMeta):

    def __init__(self,
            model_loss_func=None,
            model_eval_metric=None,
            model_init_params={},
            model_train_params={},
            optimize_mode='min',
            custom_loss_func=None,
        ):
        '''
        # model_loss_func: 模型损失函数
        # model_eval_metric: 模型评估指标名称. metric(s) to be evaluated on the evaluation set(s)
        # model_init_params: 模型初始化参数
        # model_train_params: 提供给 train 方法的参数
        # custom_loss_func: 非标准库的自定义损失函数类。必须定义 loss_name 属性 和 caculate 方法
        '''
        self.model_loss_func = model_loss_func
        self.model_eval_metric = model_eval_metric
        self.model_init_params = model_init_params
        self.model_train_params = model_train_params
        self.optimize_mode = optimize_mode
        self.custom_loss_func = custom_loss_func

        self.model_arch = None


    @abstractclassmethod
    def train_job(self):
        pass


    def tune_job(self, search_space, train_data, test_data, checkpoint_dir=None, **kwargs):
        '''
        基于 raytune 调参框架的通用方法
        '''
        num_samples=kwargs.pop('num_samples', 20)
        callbacks = kwargs.pop('callbacks', None)

        if checkpoint_dir is not None:
            if Path(checkpoint_dir).exists():
                shutil.rmtree(checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)

        report_metric_name = f'test_{self.model_eval_metric}'

        train_cifar = partial(self.train_job, train_data=train_data, test_data=test_data)
        # 神经网络模型，增加一个 max_epochs 参数
        if self.model_arch == 'nn':
            max_epochs = kwargs.pop('max_epochs', 10)
            train_cifar = partial(train_cifar, max_epochs=max_epochs)

        train_with_resources = tune.with_resources(train_cifar, resources=scaling_config)

        checkpoint_strategy = train.CheckpointConfig(
            # 只保存一个最优的checkpoint，节约存储空间
            num_to_keep=1,
            # *Best* checkpoints are determined by these params:
            checkpoint_score_attribute=report_metric_name,
            checkpoint_score_order=self.optimize_mode,
            # 不支持函数调用，只支持类调用
            # checkpoint_frequency=2,
            # checkpoint_at_end=True,
            )

        run_config = train.RunConfig(
            # 最大迭代训练的次数, report 表格中 iter 数字
            stop={"training_iteration": 10},
            # checkpoint 是 ray.train 的方法
            checkpoint_config=checkpoint_strategy,
            # checkpoint 的保存路径
            storage_path=checkpoint_dir,
            name=f'{type(self).__name__}_model',
            callbacks=callbacks,
            # callbacks=[
            #     MLflowLoggerCallback(
            #         tracking_uri=tracking_uri,
            #         experiment_name=experiment_name,
            #         save_artifact=True,
            #         ),
            #     # AimLoggerCallback(
            #     #     metrics=[self.model_eval_metric],
            #     #     ),
            #     ],
            )

        # 定义超参数搜索算法
        # 贝叶斯搜索不支持离散参数
        # bayesopt = BayesOptSearch(metric=self.model_eval_metric, mode=self.optimize_mode)
        hyperot_search = HyperOptSearch(metric=self.model_eval_metric, mode=self.optimize_mode)
        # hebo = HEBOSearch(metric=self.model_eval_metric, mode=self.optimize_mode)

        scheduler = ASHAScheduler(max_t=10, metric=report_metric_name, mode=self.optimize_mode)
        tune_config = tune.TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler,
            # 方案一：利用搜索算法自动暂停
            search_alg=hyperot_search,
            # search_alg=bayesopt,
            # search_alg=hebo,
            )

        tuner = tune.Tuner(
            train_with_resources,
            param_space=search_space,
            tune_config=tune_config,
            run_config=run_config,
            )

        trail_results = tuner.fit()
        return trail_results



    def eval_job(self, y_true, y_pred, metric_name, tasktype='binary', **kwargs):
        metric_config = {
            'auc': metrics.roc_auc_score,
            'recall': metrics.recall_score,
            'precision': metrics.precision_score,
            'accuracy': metrics.accuracy_score,
            'rmse': metrics.mean_squared_error,
            }

        if tasktype in ['binary']:
            if metric_name == 'auc':
                test_score = metric_config[metric_name](y_true, y_pred, **kwargs)
            else:
                y_label = np.where(y_pred > 0.5, 1, 0)
                test_score = metric_config[metric_name](y_true, y_label, **kwargs)
        else:
            test_score = metric_config[metric_name](y_true, y_pred, **kwargs)
        return test_score


    def test_job(self, *args, **kwargs):
        pass
