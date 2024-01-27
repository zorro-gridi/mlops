from abc import ABCMeta, abstractclassmethod
import xgboost as xgb
from catboost import Pool
from torch.utils.data import DataLoader
import numpy as np

from mlops.utils import mlflow_utils
import mlflow
from mlflow.models import infer_signature
from mlflow.client import MlflowClient
import logging
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoost as cat
import ray


class AbstractMLOps(metaclass=ABCMeta):
    def __init__(self,
            model_task=None,
            dataset_inst=None,
            best_model_args={},
            best_data_args={},
            raw_data=None,
            train_data=None,
            test_data=None,
            output_model=None,
            preprocess_func=None,
            postprocess_func=None,
            ):
        '''
        # preprocess_func: 对输入的raw_data进行预处理加工
        # postprocess_func: 对模型的输出进行加工,使满足最终输出要求。需要在主程序中定义
        '''
        self.model_task = model_task
        self.dataset_inst = dataset_inst
        self.best_model_args = best_model_args
        self.best_data_args = best_data_args
        self.raw_data = raw_data
        self.train_data = train_data
        self.test_data = test_data
        self.output_model = output_model

        self.preprocess_func = preprocess_func
        self.postprocess_func = postprocess_func

        # 框架目前已实现的模型训练流程类型
        self.mlflow_model_flavor = {
            'xgb': mlflow.xgboost,
            'cat': mlflow.catboost,
            'nn': mlflow.pytorch,
            'kmeans': mlflow.sklearn,
            'lgb': mlflow.lightgbm,
            }

    def run_data_args(self, *args, **kwargs):
        pass


    def run_model_args(self, *args, **kwargs):
        pass


    def find_best_data_args(self, *args, **kwargs):
        pass


    def find_best_model_args(self, params_space, **kwargs):
        '''
        # 该函数实现了 raytune 自动调参，并返回最优模型和参数 checkpoint
        '''

        tune_results = self.model_task.tune_job(
            params_space,
            train_data=self.train_data,
            test_data=self.test_data,
            **kwargs
            )

        report_metric_name = f'test_{self.model_task.model_eval_metric}'
        best_result = tune_results.get_best_result(
            metric=report_metric_name, mode=self.model_task.optimize_mode)

        logging.warning(f'Ray Tune Best Result: {best_result}')

        logdir = best_result.checkpoint.to_directory()
        # 最优模型文件
        checkpoint_path = Path(logdir) / self.model_task.checkpoint_model_name

        best_model = None
        if self.model_task.model_arch in ['xgb', 'lgb']:
            model_arch = self.model_task.model_arch
            best_model = eval(model_arch).Booster(model_file=checkpoint_path)

        self.output_model = best_model

        eval_metric_name = self.model_task.model_eval_metric
        best_checkpoint = {
            'best_model': best_model,
            eval_metric_name: best_result.metrics[f'test_{eval_metric_name}'],
            }

        # 更新模型实例的最优调参结果
        self.best_model_args.update(self.model_task.model_init_params)
        self.best_model_args.update(best_result.config)

        if self.dataset_inst is not None:
            self.best_data_args.update(self.dataset_inst.__dict__)

        return best_checkpoint



    def test_hist_model(self, reg_model_name, model_version='1'):
        '''
        该方法实现以下功能：
            1. 加载历史注册模型；
            2. 加载历史模型的最新数据输入；
            3. 计算历史模型损失函数。
        加载模型方面,mlflow已经实现了统一接口。加载测试数据集方面,如果模型需要定制方法,可以通过子类改写此方法
        '''
        model_arch = self.model_task.model_arch
        histt_regis_model = self.mlflow_model_flavor[model_arch].load_model(
            f"models:/{reg_model_name}/{model_version}")
        hist_model_config = mlflow_utils.load_register_model_args(reg_model_name, model_version)

        # 如果历史没有配置数据参数,就直接加载传入的数据
        if self.dataset_inst is None:
            logging.warning(f'无数据参数调参模式, 使用当前测试数据测试历史模型...')
            test_data = self.test_data
        else:
            # 加载 dataset hist cnofig, 更新当前的 dataset_inst 为历史模式
            test_data = self.dataset_inst.load_test_data(self.raw_data, inst_config=hist_model_config)

        # 历史模型不用更新参数，不用返回 model signature
        test_loader, _ = data_util_map(test_data, params_config=None)
        # 此处很容易出 bug, 根源还是没有正确加载数据
        # ====================================
        hist_eval_metric = self.model_task.test_job(histt_regis_model, test_loader)
        return hist_eval_metric


    def save_checkpoint(self, checkpoint, reg_model_name, model_version='1', model_alias=None):
        '''
        该方法实现了如下统一接口功能:
            1. 对比new model 和 hist model 的测试评分
            2. 保存并注册最优模型到 mlflow
            3. 更新 mloops 类的 output_model 输出
            4. 更新 mlops 类的 best_model_args, best_data_args
            5. 更新 mlops 类的 dataset_inst 类的属性
        '''
        tune_model_metric = checkpoint[self.model_task.model_eval_metric]
        mlflow_client = MlflowClient(mlflow.get_tracking_uri())
        model_arch = self.model_task.model_arch
        best_model = checkpoint['best_model']

        global data_util_map
        def data_util_map(test_data, params_config=None):
            '''
            # test_data: 模型输入的的的 test_data
            # params_config: mlflow signature 的 params 参数
           return:
                test_loader: 供 self.test_job 评估模型
                signature: 供 mlflow 注册模型
            '''
            if model_arch == 'xgb':
                test_loader = xgb.DMatrix(*test_data)
                X, y = test_data
                signature = infer_signature(X[:5], y[:5], params_config)

            elif model_arch == 'cat':
                test_loader = Pool(*test_data)
                X, y = test_data
                signature = infer_signature(X[:5], y[:5], params_config)

            elif model_arch == 'nn':
                test_loader = DataLoader(test_data, batch_size=1)
                test_data_sample = next(iter(test_loader))
                X, y = test_data_sample
                signature = infer_signature(X.numpy(), y.numpy(), params_config)

            # 聚类模型没有测试集
            elif model_arch == 'kmeans':
                test_loader = test_data
                X, y = test_data
                signature = infer_signature(X[:5], best_model.predict(X[:5]), params_config)

            else:
                test_loader = test_data
                X, y = test_data
                signature = infer_signature(X[:5], y[:5], params_config)
            return test_loader, signature

        if mlflow_utils.check_model_existence(reg_model_name):
            histt_regis_model = self.mlflow_model_flavor[model_arch].load_model(f"models:/{reg_model_name}/{model_version}")

            # 测试历史模型
            hist_eval_metric = self.test_hist_model(reg_model_name, model_version=model_version)
            if self.model_task.optimize_mode == 'min':
                compare_bools_result = -tune_model_metric <= -hist_eval_metric
            else:
                compare_bools_result = tune_model_metric <= hist_eval_metric

            # 默认使用最大化模式
            if compare_bools_result:
                # 更新输出的模型
                self.output_model = histt_regis_model

                logging.warning(f'''
                    tune {model_arch} model {self.model_task.model_eval_metric}: {tune_model_metric:,.3f}
                    hist {model_arch} model {self.model_task.model_eval_metric}: {hist_eval_metric:,.3f}
                    --> 使用历史最优模型推理......
                    ''')
                return 'hist', histt_regis_model
            else:
                # 将针对数据实例的更改撤回
                mlflow_client.delete_registered_model(reg_model_name)
                logging.warning(f'''
                    test loss vs: hist: {hist_eval_metric:,.6f}, new: {tune_model_metric:,.6f}.
                    历史模型评分低, 将保存当前的模型...
                    ''')
        else:
            logging.warning(f'没有注册的历史模型...')

        if tune_model_metric == 0:
            logging.warning(f'新模型同时训练失败,请调整数据,重新训练......')
            raise Exception

        # 在 hist 步，将 dataset_inst 调整到了 hist 模式
        # 因此, 如果tune模式最优，则将 dataset_inst 更新为当前最优配置
        if self.dataset_inst is not None:
            self.dataset_inst.set_attr(self.best_data_args)

        params_config = self.best_model_args
        params_config.update(self.best_data_args)
        params_config = {
            k: v for k, v in params_config.items()
            if v is not None
            and type(v) in [bool, str, int, float, list, dict, np.array, np.ndarray]
            }

        if model_arch == 'nn':
            best_model.eval()
        self.output_model = best_model

        test_data = self.test_data if self.test_data is not None else self.train_data
        test_loader, signature = data_util_map(test_data, params_config=params_config)

        run_name = f'{model_arch}_best_model_and_config'
        with mlflow.start_run(run_name=run_name):
            model_info = self.mlflow_model_flavor[model_arch].log_model(
                best_model,
                'models',
                signature=signature,
                registered_model_name=reg_model_name,
                )

            mlflow.log_params(params_config)
            mlflow.log_metric(f'test_{self.model_task.model_eval_metric}', tune_model_metric)
            mlflow_client.set_registered_model_alias(reg_model_name, model_alias, model_version)
            mlflow_client.set_registered_model_alias(reg_model_name, model_arch, model_version)
            mlflow_client.set_registered_model_tag(
                reg_model_name, f'test_{self.model_task.model_eval_metric}', str(round(tune_model_metric, 6)))

            logging.warning(f'''
                {model_arch} model test {self.model_task.model_eval_metric}: {tune_model_metric:,.3f}
                --> 使用当前模型推理......
                ''')
            logging.warning(f'当前训练的 {model_arch} 模型已保存.')
            return 'new', best_model