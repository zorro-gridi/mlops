from mlops.ml_ops.base import AbstractMLOps

import logging
import numpy as np
import pickle
from pathlib import Path

import mlflow
from mlflow.models import infer_signature
from mlflow.client import MlflowClient
from mlops.utils import mlflow_utils
import ray
from typing import Union
from copy import copy




class KmeansOps(AbstractMLOps):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def data_util_map(self, test_data, params_config=Union[None, dict]):
        '''
        Args:
            test_data: 模型输入的的的 test_data
            params_config: mlflow signature 的 params 参数
        return:
            test_loader: 供 self.test_job 评估模型
            signature: 供 mlflow 注册模型
        '''
        if params_config:
            params_config = self.exclude_non_mlflow_param_type(params_config)

        test_loader = test_data
        if isinstance(test_data, tuple):
            X, y = test_data.iloc[:5]
        else:
            X, y = test_data[:5, :], None
        signature = infer_signature(X, y, params_config)
        return test_loader, signature



    def find_best_model_args(self, params_space, **kwargs):
        '''
        Desc:
            kmeans 算法寻找最优聚类数量
        '''
        logging.warning(f'------------> 启动 Kmeans 聚类搜索...')
        best_checkpoint = self.model_task.tune_job(
            params_space,
            train_data=self.train_data,
            test_data=self.test_data,
            **kwargs
            )
        # best_model_args 和 best_data_args 都要更新
        self.best_model_args = copy(self.model_task.model_init_params)
        if self.dataset_inst:
            self.best_data_args.update(copy(self.dataset_inst.__dict__))
        return best_checkpoint



    def find_best_data_args(self, data_args_space):
        '''
        Desc:
            凡是通过 for 循环寻找最优解的变量, 都必须初始化
        Return:
            best_checkpoint
        Rmeark:
            TODO: 该方法专门设计用于聚类分析后的二分类数据研究。如果需要通用模式，请覆盖继承该方法
        '''
        # 最优 loss
        best_test_loss = 0
        # 最优模型
        final_best_estimator = None

        built_in_metric = self.model_task.model_eval_metric
        # 自定义的损失函数，必须具有 caculate (见下方代码) 方法和 loss_name 属性
        loss_fn = self.model_task.custom_loss_func
        loss_name = loss_fn.loss_name if loss_fn else None

        for data_args in data_args_space:
            logging.warning(f'''
                data args: {data_args}
                ''')

            self.dataset_inst.__dict__.update(data_args)
            X, y = self.dataset_inst.feature_engineering(self.raw_data)
            # 获取所有的正例进行模式聚类，并获取最佳聚类数量
            X_positive = np.array([x for x, label in zip(X, y) if label == 1])

            for _ in range(2):
                best_estimator = self.model_task.train_job(X_positive, **self.mlflow_config)
                pred_labels = best_estimator.predict(X)

                # 返回结果 post-processing
                best_labels_ratio = self.postprocess_func(pred_labels, y)
                custom_test_loss = loss_fn.caculate(best_labels_ratio)
                logging.warning(f'data args: {data_args} - {loss_name}: {custom_test_loss}')

                if custom_test_loss:
                    break
                else:
                    continue

            if self.model_task.optimize_mode == 'max':
                compare_bool = custom_test_loss > best_test_loss
            else:
                compare_bool = -custom_test_loss > -best_test_loss

            # 如果指标有提升, 则更新相关参数
            if compare_bool:
                # 更新最优 model 和 loss
                # 如果使用了自定以的损失函数，但是 checkpoint 返回的函数名还是 model_eval_metric，
                # 和自定义名字不一样，不要误会。不影响代码数据结果
                best_test_loss = custom_test_loss
                final_best_estimator = best_estimator
                # 必须写在里面更新
                self.train_data = X, y
                best_eatimator_params = best_estimator.get_params()
                # 更新模型参数
                # self.best_model_args.update(best_eatimator_params)
                self.best_model_args['n_clusters'] = best_eatimator_params['n_clusters']
                # 更新数据参数
                self.best_data_args.update(data_args)
                self.best_data_args.update(self.dataset_inst.__dict__)
                self.output_model = best_estimator

        if best_test_loss == 0:
            logging.warning(f'新模型训练失败，没有满足的目标数据......')

        #  在此处自定义更新了模型默认的 loss_name 名称，导致多进程的时候互相影响
        return_metirc_name = loss_name if loss_name else built_in_metric
        best_checkpoint = {
            'best_model': final_best_estimator,
            # 测试集指标是自定义的损失函数
            return_metirc_name: best_test_loss,
            # TODO: 其实可以返回聚类的评估指标
            'training_loss': best_test_loss
            }
        return best_checkpoint


    def test_hist_model(self, reg_model_name, model_version='1', model_frame=None):
        '''
        Desc:
            此处重写父类的 test_hist_model 方法。 聚类算法与常规的分类/回归算法不同
        Remark:
            TODO: 该方法也是针对聚类模式识别的特殊方法。通用方法待继承覆盖定义
        '''
        hist_model = model_frame.load_model(f"models:/{reg_model_name}/{model_version}")
        hist_model_config = self.load_hist_model_config(reg_model_name, model_version)
        trianing_loss = hist_model_config['training_loss']

        # 更新数据参数属性为历史参数
        self.dataset_inst.set_attr(hist_model_config)
        X, y = self.dataset_inst.feature_engineering(self.raw_data)

        pred_labels = hist_model.predict(X)
        best_labels_ratio = self.postprocess_func(pred_labels, y)
        hist_eval_metric = self.model_task.custom_loss_func.caculate(best_labels_ratio)
        return trianing_loss, hist_eval_metric


    def save_checkpoint(self, *args, **kwargs):
        return super().save_checkpoint(*args, model_frame=mlflow.sklearn, **kwargs)
