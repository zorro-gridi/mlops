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
import time

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoost as cat
import ray
from munch import DefaultMunch
from mlops.utils.wrappers import DictWrapper
import ray
from ray.air.integrations.mlflow import setup_mlflow


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
            mlflow_config=None,
            ):
        '''
        Args:
            preprocess_func: 对输入的raw_data进行预处理加工
            postprocess_func: 对模型的输出进行加工,使满足最终输出要求。需要在主程序中定义
            mlflow_config: mlflow 任务的配置名称与后端链接等信息 dict
                demo:
                mlflow_config = {
                    'experiment_name': experiment_name,
                    'tracking_uri': f'http://127.0.0.1:9001/',
                    }
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
        self.mlflow_config = mlflow_config

        # 框架目前已实现的模型训练流程类型
        # 这个 lazyloader 对象无法序列化, 无法存入 ray remote object store
        # 为了兼容 ray 框架，将以下参数该写入 save_checkpoint 的 model_frame 参数中
        # self.mlflow_model_flavor = {
        #     'xgb': mlflow.xgboost,
        #     'cat': mlflow.catboost,
        #     'nn': mlflow.pytorch,
        #     'kmeans': mlflow.sklearn,
        #     'lgb': mlflow.lightgbm,
        #     }

    def _get_attr_sets(self):
        kwarg = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        return DefaultMunch.fromDict(kwarg)


    def _get_attr(self, name):
        attr_ref = getattr(self, name)
        # 对于属性为 class 的属性，需要再次调用 ray.get()
        attr_value = ray.get(attr_ref)
        return attr_value


    def run_data_args(self, *args, **kwargs):
        pass


    def run_model_args(self, *args, **kwargs):
        pass


    def find_best_data_args(self, *args, **kwargs):
        pass


    def find_best_model_args(self, params_space, **kwargs):
        '''
        Args:
            params_space:
        Desc:
            该函数实现了 raytune 自动调参，并返回最优模型和参数 checkpoint
        Return:
            best_checkpoint: Dict
        '''
        if self.model_task.model_arch in ['xgb']:
            # TODO: 有些案例直接调用 task train_job 接口，因此可能引发异常
            try:
                train_data = ray.put(self.train_data)
                test_data = ray.put(self.test_data)
            except:
                logging.warning(f'train & test data already on ray remote data store! pass...')
                train_data = self.train_data
                test_data = self.test_data

        tune_results = self.model_task.tune_job(
            params_space,
            train_data=train_data,
            test_data=test_data,
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
            'training_loss': best_result.metrics['training_loss'],
            }

        # 更新模型实例的最优调参结果
        self.best_model_args.update(self.model_task.model_init_params)
        self.best_model_args.update(best_result.config)

        if self.dataset_inst is not None:
            self.best_data_args.update(self.dataset_inst.__dict__)

        return best_checkpoint


    # 一般情况
    def test_hist_model(self, reg_model_name, model_version='1', model_frame=None):
        '''
        Desc:
            该方法实现以下功能：
            1. 加载历史注册模型
            2. 加载历史模型的最新数据输入
            3. 计算历史模型损失函数
            设计模式
            =======================================
            ps1. 加载模型方面, mlflow已经实现了统一接口。
            ps2. 加载测试数据集方面, 如果模型需要定制方法, 可以通过子类继承改写此方法！！
        '''
        model_arch = self.model_task.model_arch
        hist_regis_model = model_frame.load_model(
            f"models:/{reg_model_name}/{model_version}")
        # 下载历史模型的参数
        hist_model_config = mlflow_utils.load_register_model_args(reg_model_name, model_version)
        training_loss = hist_model_config['training_loss']

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
        hist_eval_metric = self.model_task.test_job(hist_regis_model, test_loader)
        return training_loss, hist_eval_metric


    def save_checkpoint(
            self, checkpoint, reg_model_name, model_version='1', model_alias=None, model_frame=None, loss_strategy='UNIT'):
        '''
        Desc:
            该方法实现了如下统一接口功能:
            1. 对比new model 和 hist model 的测试评分
            2. 保存并注册最优模型到 mlflow
            3. 更新 mloops 类的 output_model 输出
            4. 更新 mlops 类的 best_model_args, best_data_args
            5. 更新 mlops 类的 dataset_inst 类的属性
        Args:
            model_frame: 模型框架, 需要在 ops 实现的 save_checkpoint 方法中指定默认值
                例如 lstmops 中, model_frame=mlflow.pytorch
            loss_strategy:
                'SUM': 使用 train + test 的损失综合比较, 可避免 train loss 过大的问题
                'UNIT': 仅使用 test 损失比较
        '''
        # find_best_model_args 的 checkpoint metric 指标不带 test 前缀
        if self.model_task.custom_loss_func:
            metric_name = self.model_task.custom_loss_func.loss_name
        else:
            metric_name = self.model_task.model_eval_metric
        tune_model_metric = checkpoint[metric_name]
        training_loss = checkpoint['training_loss']
        tune_sum_loss = tune_model_metric + training_loss

        model_arch = self.model_task.model_arch
        best_model = checkpoint['best_model']

        # 首先启动 mlflow 的 session
        # =============================================
        run_name = f'{model_arch}_best_model_and_config'
        setup_mlflow(run_name=run_name, **self.mlflow_config,)
        mlflow_client = MlflowClient(mlflow.get_tracking_uri())

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
            hist_regis_model = model_frame.load_model(f"models:/{reg_model_name}/{model_version}")
            # 测试历史模型
            # =================================================================
            hist_training_loss, hist_eval_metric = self.test_hist_model(
                reg_model_name, model_version=model_version, model_frame=model_frame)
            hist_sum_loss = hist_training_loss + hist_eval_metric

            if loss_strategy == 'UNIT':
                if self.model_task.optimize_mode == 'min':
                    compare_bools_result = -tune_model_metric <= -hist_eval_metric
                else:
                    compare_bools_result = tune_model_metric <= hist_eval_metric
            else:
                if self.model_task.optimize_mode == 'min':
                    compare_bools_result = -tune_sum_loss <= -hist_sum_loss
                else:
                    compare_bools_result = tune_sum_loss <= hist_sum_loss

            # 默认使用最大化模式
            if compare_bools_result:
                # 更新输出的模型
                self.output_model = hist_regis_model

                logging.warning(f'''
                    tune {model_arch} model {metric_name}: {tune_model_metric:,.3f}, sum loss: {tune_sum_loss:,.3f}
                    hist {model_arch} model {metric_name}: {hist_eval_metric:,.3f}, sum loss: {hist_sum_loss:,.3f}
                    --> 使用历史最优模型推理......
                    ''')
                return {
                    'training_loss': hist_training_loss,
                    'test_loss': hist_eval_metric,
                    'best_model': hist_regis_model
                    }
            else:
                # 将针对数据实例的更改撤回
                mlflow_client.delete_registered_model(reg_model_name)
                logging.warning(f'''
                    test loss vs ------> hist: {hist_eval_metric:,.6f}, new: {tune_model_metric:,.6f}.
                    sum loss vs  ------> hist: {hist_sum_loss:,.6f}, new: {tune_sum_loss:,.6f}.
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

        # 记录模型的关键参数
        params_config = self.best_model_args
        params_config.update(self.best_data_args)
        params_config = {
            k: v for k, v in params_config.items()
            if v is not None
            and type(v) in [bool, str, int, float, list, dict, np.array, np.ndarray]
            }
        params_config['training_loss'] = training_loss
        params_config[f'test_{metric_name}'] = tune_model_metric

        if model_arch == 'nn':
            best_model.eval()
        self.output_model = best_model

        test_data = self.test_data if self.test_data else self.train_data
        if model_arch in ['xgb']:
            test_data = ray.get(test_data)
        test_loader, signature = data_util_map(test_data, params_config=params_config)

        # with mlflow.start_run(run_name=run_name):
        # TODO: current path: file:///home/zorro/project/pycharm/mlruns/0/01de69589f3b45df8c6111899175b97c/artifacts
        logging.warning(f'register model uri: {mlflow.get_registry_uri()}')
        logging.warning(f'tracking model uri: {mlflow.get_tracking_uri()}')
        # model_info = self.mlflow_model_flavor[model_arch].log_model(
        model_info = model_frame.log_model(
            best_model,
            artifact_path='models',
            signature=signature,
            registered_model_name=reg_model_name,
            )

        mlflow.log_params(params_config)
        mlflow.log_metric(f'test_{metric_name}', tune_model_metric)
        mlflow_client.set_registered_model_alias(reg_model_name, model_alias, model_version)
        mlflow_client.set_registered_model_alias(reg_model_name, model_arch, model_version)

        mlflow_client.set_registered_model_tag(
            reg_model_name, f'test_{metric_name}', str(round(tune_model_metric, 6)))
        mlflow_client.set_registered_model_tag(
            reg_model_name, f'training_loss', str(round(training_loss, 6)))

        mlflow.end_run()
        logging.warning(f'''
            {model_arch} model test {metric_name}: {tune_model_metric:,.3f}
            --> 使用当前模型推理......
            ''')
        logging.warning(f'当前训练的 {model_arch} 模型已保存.')
        return {
            'training_loss': training_loss,
            'test_loss': tune_model_metric,
            'best_model': best_model
            }