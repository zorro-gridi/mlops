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
        self.best_model_args['model_arch'] = 'kmeans'


    def run_data_args(self, *data_args, estimator=None):
        pass


    def find_best_data_args(self, data_args_space):
        best_test_loss = 0
        for data_args in data_args_space:
            logging.warning(f'''
                data args:
                    {data_args}
                ''')

            self.dataset_inst.__dict__.update(data_args)
            X, y = self.dataset_inst.feature_engineering(self.raw_data)
            # X, y = self.train_data
            # 获取所有的正例进行模式聚类，并获取最佳聚类数量
            X_positive = np.array([x for x, label in zip(X, y) if label == 1])

            for _ in range(2):
                best_estimator = self.model_task.train_job(X_positive)
                pred_labels = best_estimator.predict(X)

                best_labels_ratio = self.postprocess_func(pred_labels, y)
                loss_fn = self.model_task.custom_loss_func
                loss_name = self.model_task.custom_loss_func.loss_name
                # loss_fn = self.model_task.model_loss_func
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

            if compare_bool:
                best_test_loss = custom_test_loss
                # 必须写在里面更新
                self.train_data = X, y
                best_eatimator_params = best_estimator.get_params()
                # 更新模型参数
                self.best_model_args.update(best_eatimator_params)
                # 更新数据参数
                self.best_data_args.update(data_args)
                self.best_data_args.update(
                    {k: v for k, v in self.dataset_inst.__dict__.items()
                     if type(v) in [str, list, dict, np.ndarray, np.array]})
                self.output_model = best_estimator

        if best_test_loss == 0:
            logging.warning(f'新模型训练失败，没有满足的目标数据......')

        self.model_task.model_eval_metric = loss_name
        best_checkpoint = {
            'best_model': best_estimator,
            self.model_task.model_eval_metric: best_test_loss,
            }
        return best_checkpoint


    def test_hist_model(self, reg_model_name, model_version='1'):
        hist_model = self.mlflow_model_flavor[self.best_model_args['model_arch']].load_model(f"models:/{reg_model_name}/{model_version}")

        hist_model_config = mlflow_utils.load_register_model_args(reg_model_name, model_version)
        # 更新数据参数属性为历史参数
        self.dataset_inst.set_attr(hist_model_config)
        X, y = self.dataset_inst.feature_engineering(self.raw_data)

        pred_labels = hist_model.predict(X)
        best_labels_ratio = self.postprocess_func(pred_labels, y)
        hist_eval_metric = self.model_task.custom_loss_func.caculate(best_labels_ratio)
        return hist_eval_metric



    # def save_checkpoint(self, checkpoint, reg_model_name, model_version='1', model_alias='custom_alias'):
    #     tune_model_metric = checkpoint[self.model_task.model_eval_metric]
    #     mlflow_client = MlflowClient(mlflow.get_tracking_uri())
    #     model_arch = self.best_model_args['model_arch']

    #     if mlflow_utils.check_model_existence(reg_model_name):
    #         hist_model = self.mlflow_model_flavor[self.best_model_args['model_arch']].load_model(f"models:/{reg_model_name}/{model_version}")

    #         hist_model_config = mlflow_utils.load_register_model_args(reg_model_name, model_version)
    #         # 更新数据参数属性为历史参数
    #         self.dataset_inst.set_attr(hist_model_config)
    #         X, y = self.dataset_inst.feature_engineering(self.raw_data)

    #         pred_labels = hist_model.predict(X)
    #         best_labels_ratio = self.postprocess_func(pred_labels, y)
    #         hist_eval_metric = self.model_task.custom_loss_func.caculate(best_labels_ratio)

    #         if self.model_task.optimize_mode == 'min':
    #             compare_bools_result = -tune_model_metric <= -hist_eval_metric
    #         else:
    #             compare_bools_result = tune_model_metric <= hist_eval_metric

    #         # 默认使用最大化模式
    #         if compare_bools_result:
    #             logging.warning(f'''
    #                 tune model {self.model_task.model_eval_metric}: {tune_model_metric:,.3f}
    #                 hist model {self.model_task.model_eval_metric}: {hist_eval_metric:,.3f}
    #                 --> 使用历史最优模型推理......
    #                 ''')
    #             self.train_data = (X, y)
    #             self.output_model = hist_model
    #             return
    #         else:
    #             mlflow_client.delete_registered_model(reg_model_name)

    #     if self.dataset_inst is not None:
    #         # 将针对数据实例的更改撤回
    #         self.dataset_inst.set_attr(self.best_data_args)

    #     if tune_model_metric == 0:
    #         logging.warning(f'未注册历史模型，新模型同时训练失败，请调整数据，重新训练......')
    #         raise Exception

    #     logging.warning(f'没有注册的历史模型, 或者历史模型评分低......')
    #     params_config = self.best_model_args
    #     params_config.update(self.best_data_args)
    #     best_model = checkpoint['best_model']
    #     self.output_model = best_model

    #     X, y = self.train_data
    #     signature = infer_signature(X[:5], y[:5], params_config)

    #     run_name = f'{self.best_model_args["model_arch"]}_best_model_and_config'
    #     with mlflow.start_run(run_name=run_name):
    #         # 保存模型
    #         model_info = self.mlflow_model_flavor[model_arch].log_model(
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
    #             new model {self.model_task.model_eval_metric}: {tune_model_metric:,.3f}
    #             --> 使用当前模型推理......
    #             ''')
    #         logging.warning(f'当前训练的 xgb 模型已保存.')


    def save_checkpoint(self, *args, **kwargs):
        return super().save_checkpoint(*args, **kwargs)


    def run_model_args(self, *data_args_space, **kwargs):
        pass


    def find_best_model_args(self, *model_params_search_space, **kwargs):
        pass
