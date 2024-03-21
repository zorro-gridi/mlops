from abc import ABCMeta, abstractclassmethod
from sklearn import metrics
import numpy as np
from pathlib import Path
from functools import partial
import shutil

import ray
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


    @abstractclassmethod
    def test_job(self):
        pass


    def tune_job(self, search_space, train_data, test_data, checkpoint_dir=None, **kwargs):
        '''
        Args:
            search_space:
            train_data:
            test_data:
            checkpoint_dir:
        Kwargs:
            num_samples:
            callbacks:
            num_gpus:
            num_cpus:
            custom_metric: default "train_test_mean_loss" 处理有时候test loss小, 但是train loss 大的问题
        Return:
            tuen.ResultGrid class
        Desc:
            基于 ray[tune] 调参框架的通用方法
        '''
        num_samples=kwargs.pop('num_samples', 20)
        callbacks = kwargs.pop('callbacks', None)

        if checkpoint_dir is not None:
            if Path(checkpoint_dir).exists():
                shutil.rmtree(checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)

        # 定义选取最佳实验的评估指标
        eval_metric = kwargs.pop('custom_metric', None)
        report_metric_name = f'test_{self.model_eval_metric}' if not eval_metric else eval_metric

        # https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html#how-to-leverage-gpus-in-tune
        num_gpus = kwargs.pop('num_gpus', round(1/4, 1))
        # 这个是表示每个 trial 使用的 cpu 数量
        num_cpus = kwargs.pop('num_cpus', 1)

        # TODO: 直接把偏函数传入 ray.tune 导致 large object warning, 因为，参数太大了
        # 解决方案：使用 ray.put() ray.get() 好像能解决问题

        # Warning: The actor ImplicitFunc is very large (20 MiB).
        # Check that its definition is not implicitly capturing a large array or other object in scope.
        # Tip: use ray.put() to put large objects in the Ray object store.
        train_cifar = partial(self.train_job, train_data=train_data, test_data=test_data)

        # 神经网络模型，增加一个 max_epochs 参数
        if self.model_arch == 'nn':
            max_epochs = kwargs.pop('max_epochs', 10)
            train_cifar = partial(train_cifar, max_epochs=max_epochs)

        # scaling_config: 可能是 ray.train 的写法
        # train_with_resources = tune.with_resources(train_cifar, resources=scaling_config)
        # tune 写法案例：https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html
        train_with_resources = tune.with_resources(train_cifar, resources={'gpu': num_gpus, 'cpu': num_cpus})
        checkpoint_strategy = train.CheckpointConfig(
            # 只保存 1 个最优的checkpoint,节 约存储空间, 但是会导致训练一直在同步 checkpoint, 训练速度变慢
            num_to_keep=5,
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
            #     #     metrics=[report_metric_name],
            #     #     ),
            #     ],
            )

        # 定义超参数搜索算法
        # 贝叶斯搜索不支持离散参数
        # bayesopt = BayesOptSearch(metric=report_metric_name, mode=self.optimize_mode)
        hyperot_search = HyperOptSearch(metric=report_metric_name, mode=self.optimize_mode)
        # hebo = HEBOSearch(metric=report_metric_name, mode=self.optimize_mode)

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
                # 二分类问题的 准确率 & 精确率需要使用预测标签计算，而不是 proba 概率
                y_label = np.where(y_pred > 0.5, 1, 0)
                test_score = metric_config[metric_name](y_true, y_label, **kwargs)
        else:
            test_score = metric_config[metric_name](y_true, y_pred, **kwargs)
        return test_score
