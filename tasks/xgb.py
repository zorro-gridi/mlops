import xgboost as xgb
import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from xgboost.callback import EarlyStopping
import numpy as np
import logging
import random
import time
import os
from pathlib import Path
import shutil

from functools import partial
from mlops.tasks.base import AbstractModelFactory

import mlflow
from mlflow.models import infer_signature
from ray.air.integrations.mlflow import setup_mlflow, MLflowLoggerCallback



class xgboost_task(AbstractModelFactory):
    def __init__(self, **kwargs):
        '''
        # model_eval_metric: 原生 xgb 接口支持 metric str 列表，本 task 类只支持 str
        '''
        super(xgboost_task, self).__init__(**kwargs)
        self.checkpoint_model_name = 'xgb_model_checkpoint.json'
        self.model_arch = 'xgb'


    def train_job(self, config, train_data, test_data, checkpoint=False):
        '''
        # train_data: tuple, (X_train, y_train)
        # test_data: tuple, (X_test, y_test)
        '''
        bst_params = {
            'max_depth': config['max_depth'],
            'eta': config['eta'],
            'objective': self.model_loss_func,
            'eval_metric': self.model_eval_metric,
            'min_child_weight': config['min_child_weight'],
            # 是否是因为 subsample 参数影响 ???
            'subsample': config['subsample'],
            }
        # 将初始化参数更新到 config 中
        bst_params.update(self.model_init_params)

        early_stopping = EarlyStopping(
            rounds=100,
            metric_name=self.model_eval_metric,
            data_name='test',
            save_best=True,
            min_delta=1e-3 * 0.2,
            maximize=True if self.optimize_mode == 'max' else False,
            )

        dtrain = xgb.DMatrix(*train_data)
        dtest = xgb.DMatrix(*test_data)

        evals_result = {}
        # random_state 在哪里设置???
        bst = xgb.train(
            bst_params,
            dtrain,
            num_boost_round=10000,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            evals_result=evals_result,
            callbacks=[early_stopping],
            **self.model_train_params,
            )

        # 使用 auc 作为评估函数
        logging.warning(f'logging mlflow......')
        test_eval_metric_list = evals_result['test'][self.model_eval_metric]
        train_eval_metric_list = evals_result['train'][self.model_eval_metric]

        best_bst_round = np.argmax(np.array(evals_result['test'][self.model_eval_metric]))
        test_eval_metric = test_eval_metric_list[best_bst_round]
        train_eval_metric = train_eval_metric_list[best_bst_round]

        # 返回 xgb 模型实例
        if checkpoint:
            best_checkpoint = {
                'best_model': bst,
                self.model_eval_metric: test_eval_metric,
                'best_iteration': bst.best_iteration,
                }
            return best_checkpoint

        # 创建 checkpoint 子文件夹
        os.makedirs("models", exist_ok=True)
        bst.save_model(f"models/{self.checkpoint_model_name}")
        report_checkpoint = train.Checkpoint.from_directory("models")

        # logging.warning(f'best_bst_round: {best_bst_round}, test_auc: {test_auc}')
        train.report(metrics={
            f'test_{self.model_eval_metric}': test_eval_metric,
            f'train_{self.model_eval_metric}': train_eval_metric,
            'best_iteration': bst.best_iteration,
            }, checkpoint=report_checkpoint)


    def tune_job(self, search_space, train_data, test_data, num_samples=20, checkpoint_dir='xgb_checkpoint'):
        if Path(checkpoint_dir).exists():
            shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)

        scale_config = train.ScalingConfig(
            num_workers=4,
            resources_per_worker={
                'CPU': 4,
                'GPU': 0,
                },
            use_gpu=False,
            )

        report_metric_name = f'test_{self.model_eval_metric}'
        train_cifar = partial(self.train_job, train_data=train_data, test_data=test_data)

        train_with_resources = tune.with_resources(train_cifar, resources=scale_config)
        tune_config = tune.TuneConfig(
            num_samples=num_samples,
            scheduler=ASHAScheduler(max_t=10, metric=report_metric_name, mode=self.optimize_mode),
            )

        run_config = train.RunConfig(
            # 最大迭代训练的次数, report 表格中 iter 数字
            stop={"training_iteration": 10},
            # checkpoint 是 ray.train 的方法
            checkpoint_config=train.CheckpointConfig(
                # 只保存一个最优的checkpoint，节约存储空间
                num_to_keep=1,
                # *Best* checkpoints are determined by these params:
                checkpoint_score_attribute=self.model_eval_metric,
                checkpoint_score_order=self.optimize_mode,
                # 不支持函数调用，只支持类调用
                # checkpoint_frequency=2,
                # checkpoint_at_end=True,
                ),
            # checkpoint 的保存路径
            storage_path=checkpoint_dir,
            name='xgb_model',
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

        tuner = tune.Tuner(
            train_with_resources,
            tune_config=tune_config,
            run_config=run_config,
            param_space=search_space,
            )

        results = tuner.fit()
        return results


    def eval_job(self, model, dtest, metric_name, **kwargs):
        y_true = dtest.get_label()
        y_pred = model.predict(dtest)

        test_score = super(xgboost_task, self).eval_job(y_true, y_pred, metric_name, **kwargs)
        return test_score


    # test job 与 eval job 的区别是，固定使用 self.model_eval_metric
    def test_job(self, model, dtest, **kwargs):
        y_true = dtest.get_label()
        y_pred = model.predict(dtest)

        test_score = super(xgboost_task, self).eval_job(y_true, y_pred, metric_name=self.model_eval_metric, **kwargs)
        return test_score
