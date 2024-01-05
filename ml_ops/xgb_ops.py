import random
import logging
import pickle
import xgboost as xgb
from pathlib import Path
import numpy as np

from mlops.ml_ops.base import AbstractMLOps
from mlops.tasks.xgb import xgboost_task


import mlflow
from mlflow.models import infer_signature
from mlops.utils import mlflow_utils
from mlflow.client import MlflowClient


class XgboostOps(AbstractMLOps):
    def __init__(self, **kwargs):
        super(XgboostOps, self).__init__(**kwargs)
        self.best_model_args['model_arch'] = 'xgb'

    def run_data_args(self, *data_args, **kwargs):
        pass

    def find_best_data_args(self, *args, **kwargs):
        pass


    def run_model_args(self, model_args, xgb_model=None):
        if xgb_model:
            dtest = xgb.DMatrix(*(self.test_data))
            return self.model_task.test_job(xgb_model, dtest)

        best_checkpoint = self.model_task.train_job(model_args, self.train_data, self.test_data, checkpoint=True)
        return best_checkpoint


    def find_best_model_args(self, model_args_space, **kwargs):
        best_result = self.model_task.tune_job(model_args_space, self.train_data, self.test_data, **kwargs)
        # 更新 xgb 模型实例的最优调参结果
        self.best_model_args.update(best_result.config)
        if self.dataset_inst is not None:
            self.best_data_args.update(
                {k: v for k, v in self.dataset_inst.__dict__.items()
                 if type(v) in [str, list, dict, np.ndarray, np.array]})
        return best_result


    # def save_checkpoint(self, checkpoint, reg_model_name, model_version='1', model_alias=None):
    #     tune_model_metric = checkpoint['model_eval_metric']
    #     mlflow_client = MlflowClient(mlflow.get_tracking_uri())

    #     if mlflow_utils.check_model_existence(reg_model_name):
    #         hist_xgb_model = mlflow.xgboost.load_model(f"models:/{reg_model_name}/{model_version}")
    #         hist_model_config = mlflow_utils.load_register_model_args(reg_model_name, model_version)

    #         # 如果历史没有配置数据参数，就直接加载传入的数据
    #         if self.dataset_inst is None:
    #             # 这里还是有问题，因为 kmeans 提供的数据结构会变化，所以还是要重新生成 测试数据集
    #             logging.warning(f'无数据参数调参模式, 使用当前测试数据测试历史模型...')
    #             x_test, y_test = self.test_data
    #         else:
    #             x_test, y_test = self.dataset_inst.load_test_data(self.raw_data, inst_config=hist_model_config)

    #         dtest = xgb.DMatrix(x_test, y_test)
    #         hist_eval_metric = self.model_task.test_job(hist_xgb_model, dtest)

    #         if self.model_task.optimize_mode == 'min':
    #             compare_bools_result = -tune_model_metric <= -hist_eval_metric
    #         else:
    #             compare_bools_result = tune_model_metric <= hist_eval_metric

    #         # 默认使用最大化模式
    #         if compare_bools_result:
    #             # 更新输出的模型
    #             self.output_model = hist_xgb_model

    #             logging.warning(f'''
    #                 tune xgb model {self.model_task.model_eval_metric}: {tune_model_metric:,.3f}
    #                 hist xgb model {self.model_task.model_eval_metric}: {hist_eval_metric:,.3f}
    #                 --> 使用历史最优模型推理......
    #                 ''')
    #             return 'hist', hist_xgb_model
    #         else:
    #             # 将针对数据实例的更改撤回
    #             mlflow_client.delete_registered_model(reg_model_name)

    #     if self.dataset_inst is not None:
    #         self.dataset_inst.set_attr(self.best_data_args)

    #     logging.warning(f'没有注册的历史模型, 或者历史模型评分低......')
    #     params_config = self.best_model_args
    #     params_config.update(self.best_data_args)
    #     best_model = checkpoint['best_model']
    #     self.output_model = best_model

    #     X, y = self.test_data
    #     signature = infer_signature(X[:5], y[:5], params_config)

    #     dtest = xgb.DMatrix(*self.test_data)
    #     run_name = f'{self.best_model_args["model_arch"]}_best_model_and_config'
    #     with mlflow.start_run(run_name=run_name):
    #         model_info = mlflow.xgboost.log_model(
    #             best_model,
    #             'models',
    #             signature=signature,
    #             registered_model_name=reg_model_name,
    #             )

    #         mlflow.log_params(params_config)
    #         mlflow.log_metric(f'test_{self.model_task.model_eval_metric}', tune_model_metric)
    #         mlflow_client.set_registered_model_alias(reg_model_name, model_alias, model_version)
    #         mlflow_client.set_registered_model_alias(reg_model_name, self.best_model_args['model_arch'], model_version)
    #         mlflow_client.set_registered_model_tag(reg_model_name, f'test_{self.model_task.model_eval_metric}', str(tune_model_metric))

    #         logging.warning(f'''
    #             xgb model test {self.model_task.model_eval_metric}: {tune_model_metric:,.3f}
    #             --> 使用当前模型推理......
    #             ''')
    #         logging.warning(f'当前训练的 xgb 模型已保存.')

    #         return 'new', best_model


    def save_checkpoint(self, *args, **kwargs):
        return super(XgboostOps, self).save_checkpoint(*args, **kwargs)