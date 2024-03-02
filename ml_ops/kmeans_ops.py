from mlops.ml_ops.base import AbstractMLOps

import logging
import numpy as np
import pickle
from pathlib import Path

import mlflow
from mlflow.models import infer_signature
from mlflow.client import MlflowClient
from mlops.utils import mlflow_utils



class KmeansOps(AbstractMLOps):
    def __init__(self, **kwargs):
        super(KmeansOps, self).__init__(**kwargs)


    def run_data_args(self,):
        pass


    def find_best_data_args(self, data_args_space):
        '''
        凡是通过 for 循环寻找最优解的变量, 都必须初始化
        # best_test_loss: 最优 loss
        # final_best_estimator: 最优模型
        '''
        best_test_loss = 0
        final_best_estimator = None

        for data_args in data_args_space:
            logging.warning(f'''
                data args:
                    {data_args}
                ''')

            self.dataset_inst.__dict__.update(data_args)
            X, y = self.dataset_inst.feature_engineering(self.raw_data)
            # 获取所有的正例进行模式聚类，并获取最佳聚类数量
            X_positive = np.array([x for x, label in zip(X, y) if label == 1])

            for _ in range(2):
                best_estimator = self.model_task.train_job(X_positive)
                pred_labels = best_estimator.predict(X)

                best_labels_ratio = self.postprocess_func(pred_labels, y)
                loss_fn = self.model_task.custom_loss_func
                loss_name = self.model_task.custom_loss_func.loss_name
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

        self.model_task.model_eval_metric = loss_name
        best_checkpoint = {
            'best_model': final_best_estimator,
            self.model_task.model_eval_metric: best_test_loss,
            }
        return best_checkpoint


    def test_hist_model(self, reg_model_name, model_version='1'):
        '''
        此处重写父类的 test_hist_model 方法
        '''
        hist_model = self.mlflow_model_flavor[self.model_task.model_arch].load_model(
            f"models:/{reg_model_name}/{model_version}")

        hist_model_config = mlflow_utils.load_register_model_args(reg_model_name, model_version)
        # 更新数据参数属性为历史参数
        self.dataset_inst.set_attr(hist_model_config)
        X, y = self.dataset_inst.feature_engineering(self.raw_data)

        pred_labels = hist_model.predict(X)
        best_labels_ratio = self.postprocess_func(pred_labels, y)
        hist_eval_metric = self.model_task.custom_loss_func.caculate(best_labels_ratio)
        return hist_eval_metric


    def save_checkpoint(self, *args, **kwargs):
        return super().save_checkpoint(*args, **kwargs)


    def run_model_args(self, *data_args_space, **kwargs):
        pass


    def find_best_model_args(self, *model_params_search_space, **kwargs):
        pass
