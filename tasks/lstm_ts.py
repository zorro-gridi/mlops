import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
import numpy as np
import logging
import random
import time

from functools import partial
from mlops.tasks.base import AbstractModelFactory
from mlops.baseConfig.raytuneConfig import (
    scaling_config,
    )

import mlflow
from mlflow.models import infer_signature
from ray.air.integrations.mlflow import setup_mlflow, MLflowLoggerCallback
from ray.tune.logger.aim import AimLoggerCallback
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hebo import HEBOSearch
from ray.tune.search.hyperopt import HyperOptSearch


import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    )
from pathlib import Path
import shutil


# from mlops.datas.BaseDt import SeqToTsDt
from mlops.nn import train_utils
import os

# will deprecated in fueture version
# ==================================
from mlops.nn.lstm import LstmModel
LstmModel = LstmModel
# ==================================


class lstmTsTask(AbstractModelFactory):
    def __init__(self, nn_arch, **kwargs):
        super(lstmTsTask, self).__init__(**kwargs)
        self.model_arch = 'nn'
        self.nn_arch = nn_arch
        self.model_checkpoint_name = 'lstm_checkpoint.pt'

    # train_job ray tune 调参不支持传入 **kwargs 关键字参数
    def train_job(self, config, train_data, test_data, max_epochs=10):
        '''
        # config: 输入的模型和数据参数。
        '''
        logging.warning(f'trial config: {config}')
        model_args = config['model_args_space']
        self.model_init_params.update(model_args)

        # 除去 model_args 剩余的为 data_args
        data_args = config['data_args_space']
        batch_size = data_args['batch_size']

        logging.warning(f'model_init_params: {self.model_init_params}')
        dnn_model = self.nn_arch(**self.model_init_params)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        lr = config.pop('lr', None)
        # AdamW 优化器
        if lr:
            optimizer = torch.optim.AdamW(params=dnn_model.parameters(), lr=config['lr'])
        else:
            optimizer = torch.optim.AdamW(params=dnn_model.parameters())
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        logging.warning(f'train_loader len: {len(train_loader)}')
        logging.warning(f'train loopping......')

        # iterate epoch
        metric_loss_init = 1e10
        for i in range(max_epochs):
            training_loss = train_utils.train_func(dnn_model, optimizer, train_loader)
            # 使用学习率衰减策略
            # 一次 epoch 更新一次 lr scheduler
            scheduler.step()
            metric_loss = train_utils.test_func(dnn_model, test_loader)

            # 创建 checkpoint 子文件夹
            os.makedirs("models", exist_ok=True)
            # 同时保存模型和优化器
            improved_rate = (metric_loss_init - metric_loss) / metric_loss_init
            if improved_rate >= 0.01:
            # if train.get_context().get_world_rank() == 0:
                torch.save(
                    (dnn_model.state_dict(), optimizer.state_dict()), f'models/{self.model_checkpoint_name}')
                metric_loss_init = metric_loss

                # checkpoint (实际上 checkpoint 的意思就是每一轮训练都保存一个文件夹)
                report_checkpoint = train.Checkpoint.from_directory('models')
                # 将训练指标报告给 tuner
                if not self.model_eval_metric:
                    logging.warning(f'Please set a eval metric name when model initialized.....')
                    self.model_eval_metric = 'unkown_test_metric'

                logging.warning(f'''
                    Epoch: {i}:
                        train {self.model_eval_metric} loss: {training_loss:,.6f}
                        test {self.model_eval_metric} loss: {metric_loss:,.6f}
                    ''')

                report_metric_name = f'test_{self.model_eval_metric}'
                train.report(metrics={report_metric_name: metric_loss, 'training_loss': training_loss}, checkpoint=report_checkpoint)
            else:
                train.report(metrics={report_metric_name: metric_loss, 'training_loss': training_loss})


    def test_job(self, model, test_loader):
        return train_utils.test_func(model, test_loader)


    def tune_job(self, *args, **kwargs):
        return super().tune_job(*args, **kwargs)


    # def tune_job(self, search_space, train_data, test_data, checkpoint_dir, **kwargs):
    #     max_epochs = kwargs.pop('max_epochs', 10)
    #     num_samples=kwargs.pop('num_samples', 20)

    #     # tracking_uri = search_space.pop('tracking_uri', None)
    #     # experiment_name = search_space.pop('experiment_name', None)

    #     logging.warning(f'search_space: {search_space}')
    #     logging.warning(f'max_epochs: {max_epochs}')
    #     run_config = train.RunConfig(
    #         # 最大迭代训练的次数, report 表格中 iter 数字
    #         stop={"training_iteration": 10},
    #         # checkpoint 是 ray.train 的方法
    #         checkpoint_config=train.CheckpointConfig(
    #             # 只保存一个最优的checkpoint，节约存储空间
    #             num_to_keep=1,
    #             # *Best* checkpoints are determined by these params:
    #             checkpoint_score_attribute=self.model_eval_metric,
    #             checkpoint_score_order=self.optimize_mode,
    #             # 不支持函数调用，只支持类调用
    #             # checkpoint_frequency=2,
    #             # checkpoint_at_end=True,
    #             ),
    #         # checkpoint 的保存路径
    #         storage_path=checkpoint_dir,
    #         name='lstm_model',
    #         # callbacks=[
    #         #     MLflowLoggerCallback(
    #         #         tracking_uri=tracking_uri,
    #         #         experiment_name=experiment_name,
    #         #         save_artifact=True,
    #         #         ),
    #         #     # AimLoggerCallback(
    #         #     #     metrics=[self.model_eval_metric],
    #         #     #     ),
    #         #     ],
    #         )

    #     # 定义 ray tune 运行的资源
    #     train_cifar_partial = partial(
    #         self.train_job, train_data=train_data, test_data=test_data, max_epochs=max_epochs)
    #     # 资源配置
    #     train_with_resources = tune.with_resources(train_cifar_partial, resources=scaling_config)
    #     scheduler = ASHAScheduler(metric=self.model_eval_metric, mode=self.optimize_mode)
    #     # 定义超参数搜索算法
    #     # bayesopt = BayesOptSearch(metric=self.model_eval_metric, mode=self.optimize_mode)
    #     hyperot_search = HyperOptSearch(metric=self.model_eval_metric, mode=self.optimize_mode)
    #     # hebo = HEBOSearch(metric=self.model_eval_metric, mode=self.optimize_mode)

    #     tuner = tune.Tuner(
    #         train_with_resources,
    #         tune_config=tune.TuneConfig(
    #             num_samples=num_samples,
    #             scheduler=scheduler,
    #             # 方案一：利用搜索算法自动暂停
    #             search_alg=hyperot_search,
    #             # search_alg=bayesopt,
    #             # search_alg=hebo,
    #             ),
    #         # 方案二：使用 early stop 技术
    #         param_space=search_space,
    #         run_config=run_config,
    #         **kwargs,
    #         )
    #     trail_results = tuner.fit()
    #     return trail_results


    def eval_job(self, ):
        pass
