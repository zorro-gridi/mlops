import functools
from abc import ABCMeta, abstractclassmethod
import datetime
import xgboost as xgb
from catboost import Pool
from torch.utils.data import DataLoader
import numpy as np
from copy import copy
from pprint import pprint
from typing import Union

from mlops.utils import mlflow_utils
from mlops.datas.exceptions import (
    No_SeqDataException,
    No_MLflow_Model_Found_Exception,
    )


# import mlflow
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
import pandas as pd
import mlflow

from mlflow.exceptions import RestException

# from threading import Lock
# lock = Lock()

mlflow.set_tracking_uri(f'http://192.168.5.7:9001/')


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
                config demo:
                mlflow_config = {
                    'experiment_name': experiment_name,
                    'tracking_uri': f'http://192.168.5.7:9001/',
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

        # 这个 lazyloader 对象无法序列化, 无法存入 ray remote object store
        # 为了兼容 ray 框架，将以下参数该写入 save_checkpoint 的 model_frame 参数中
        # self.mlflow_model_flavor = {
        #     'xgb': mlflow.xgboost,
        #     'cat': mlflow.catboost,
        #     ...
        #     }

    def _get_attr_sets(self):
        '''
        Desc:
            获取类的所有的属性
        '''
        kwarg = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        return DefaultMunch.fromDict(kwarg)


    def _get_attr(self, name):
        '''
        Desc:
            获取类的指定属性
        '''
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
        Desc:
            该函数实现了 raytune 自动调参，并返回最优模型和参数 checkpoint.
            主要操作如下：
            1. 获取最优checkpoint
            2. 更新 mlops 的 best_data_args
            3. 更新 mlops 的 best_model_args
        Args:
            params_space:
        Return:
            best_checkpoint: Dict
        '''
        if self.model_task.model_arch in ['xgb']:
            # TODO: 有些案例直接调用 task train_job 接口，因此可能引发异常
            try:
                train_data = ray.put(self.train_data)
                test_data = ray.put(self.test_data)

                # 更新 self 的 data 类型为 ray ObjectRef
                self.train_data = train_data
                self.test_data = test_data
                logging.warning(f'------> 将数据 put 到 ray remote')
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

        # 获取测试指标对比情况
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

        # 更新模型的最有数据参数
        if self.dataset_inst is None:
            logging.warning(f'---------> MLOps 无 dataset inst 模式')
            return best_checkpoint

        # 更新模型的数据参数
        self.best_data_args.update(copy(self.dataset_inst.__dict__))
        logging.warning(f'更新后的 MLOps 的 best_data_args: {self.best_data_args}')
        return best_checkpoint


    def load_hist_model_config(self, reg_model_name, model_version):
        '''
        Desc:
            加载注册模型的参数
        Special Usage:
            可通过自定义该方法，传入外部参数到数据的参数字典
        Remark:
            如果需要对模型的config进行特殊处理, 可以通过继承重写该方法
        '''
        # 下载历史模型的参数
        hist_model_config = mlflow_utils.load_register_model_args(reg_model_name, model_version)
        logging.warning(f'----------> load_hist_model_config 加载历史模型配置:')
        pprint(hist_model_config)
        return hist_model_config


    def data_util_map(self, test_data, params_config=Union[None, dict]):
        '''
        Desc:
            返回历史模型测试使用的 test data loader & 注册 mlflow 的 signature
        Args:
            test_data: 模型输入的的的 test_data
            params_config: mlflow signature 的 params 参数
        return:
            test_loader: 供 self.test_job 评估模型
            signature: 供 mlflow 注册模型
        '''
        if params_config:
            params_config = self.exclude_non_mlflow_param_type(params_config)

        test_loader = None
        signature = None
        return test_loader, signature


    def _eval_hist_model(self, model, config):
        '''
        Desc:
            评估历史模型的评分, 主要两个操作
            1. self.dataset_inst 实例方法 load_test_data() 加载测试数据
            2. self.model_task.test_job() 方法评估模型
        Args:
            model: 历史模型实例
            config: 历史模型的配置
        Return:
            hist_eval_metric: 历史模型在新数据的测试损失
        '''
        # 如果历史没有配置数据参数,就直接加载传入的数据
        if self.dataset_inst is None:
            logging.warning(f'无数据参数调参模式, 使用当前测试数据测试历史模型...')
            test_data = self.test_data
        else:
            # load_test_data 加载历史模型的测试集
            # 加载 dataset hist cnofig, 更新当前的 dataset_inst 为历史模式
            logging.warning(f'----------> 历史模型刚注册已过期，最新数据重新测试')
            test_data = self.dataset_inst.load_test_data(self.raw_data, inst_config=config)

        # 历史模型不用更新参数，不用返回 model signature, 所以 params_config 可为 None
        test_loader, _ = self.data_util_map(test_data, params_config=config)
        # 此处很容易出 bug, 根源还是没有正确加载数据
        # ====================================
        hist_eval_metric = self.model_task.test_job(model, test_loader)
        return hist_eval_metric


    def test_hist_model(self, reg_model_name, model_version='1', model_frame=None, update_interval=5):
        '''
        Desc:
            测试历史模型的一般方法
        Usage:
            可优先改写 load_hist_model_config 方法, 该方法返回 datainst 的 config 配置，mlops内部会自动更新实例配置
            该方法提供 BaseOps 测试历史注册模型的一般化方法，如果需要在应用中定制化，可以通过继承重写灵活满足实际需要。
            该方法实现以下功能：
                1. 加载历史注册模型
                2. 加载历史模型的最新数据输入
                3. 计算历史模型损失函数
            设计模式
            =======================================
                ps1. 加载模型方面: mlflow已经实现了统一接口。
                ps2. 加载测试数据集方面: 如果模型需要定制方法, 可以通过子类继承改写此方法！！!
        Remark:
            1. log to mlflow number type option: ["float", "int", ...]
        Args:
            update_interval: default 5. 模型测试更新的天数，超过该时间，使用最新测试数据测试一次
        Return:
            training_loss: 训练损失
            hist_eval_metric: 测试损失
        '''
        try:
            hist_regis_model = model_frame.load_model(f"models:/{reg_model_name}/{model_version}")
        except No_MLflow_Model_Found_Exception:
            return np.inf, np.inf if self.model_task.optimize_mode == 'max' else -np.inf, -np.inf

        # 下载历史模型的参数
        hist_model_config = self.load_hist_model_config(reg_model_name, model_version)
        if hist_model_config:
            pass
        else:
            logging.warning(f'-------> hist_model_config: {type(hist_model_config)}')
            raise Exception(f'-------> 没有找到历史的模型配置')

        # 历史模型的 training_loss 注册信息中提取
        training_loss = hist_model_config['training_loss']
        current_date = datetime.datetime.today()
        regist_date = hist_model_config['regist_date']
        days_diff = (current_date - datetime.datetime.strptime(regist_date, '%Y-%m-%d')).days

        if days_diff <= update_interval:
            logging.warning(f'----------> 历史模型刚注册未超过 {update_interval} 天，不使用最新数据测试')
            if self.model_task.custom_loss_func:
                # 自定义损失函数名
                metric_name = self.model_task.custom_loss_func.loss_name
            else:
                metric_name = self.model_task.model_eval_metric
            test_loss = hist_model_config[f'test_{metric_name}']
            return training_loss, test_loss

        # 如果超过注册有效期，就使用新数据，重新评估历史模型
        hist_eval_metric = self._eval_hist_model(hist_regis_model, hist_model_config)
        return training_loss, hist_eval_metric


    def exclude_non_mlflow_param_type(self, params_config):
        '''
        Desc:
            排除mlflow不支持注册的参数类型
        '''
        params_config_copy = copy(params_config)
        new_params_config = {
            k: v for k, v in params_config_copy.items()
            if v is not None
                # 筛选 data args 的类型, 因为 mlflow 限制log params的数据类型
                and type(v) in [
                    bool, str, int, float, list, dict, np.array, np.ndarray, np.float64
                ]
                and type(v) not in [
                    functools.partial,
               ]
            }
        return new_params_config

    def save_checkpoint(
        self,
        checkpoint,
        reg_model_name,
        model_alias=None,
        model_frame=None,
        loss_strategy='UNIT',
        ):
        '''
        Desc:
            该方法实现了如下统一接口功能:
            1. 对比new model 和 hist model 的测试评分
            2. 保存并注册最优模型到 mlflow
            3. 更新 mloops 类的 output_model 输出
            4. 更新 mlops 类的 best_model_args, best_data_args
            5. 更新 mlops 类的 dataset_inst 类的属性
        Args:
            model_frame: 默认为None, 不需要传递。模型框架, 在 BaseOps 中为参数占位符。
                在对应框架的 ops 实现的 save_checkpoint 方法中已经指定默认值
                例如 LstmOps 中, model_frame=mlflow.pytorch
            loss_strategy:
                'SUM': 使用 train + test 的损失综合比较, 可避免 train loss 过大的问题
                'UNIT': 仅使用 test 损失比较
                'WEIGHT': 加权损失
        Return:
            checkpoint
        '''
        # find_best_model_args 的 checkpoint metric 指标不带 test 前缀
        if self.model_task.custom_loss_func:
            metric_name = self.model_task.custom_loss_func.loss_name
        else:
            metric_name = self.model_task.model_eval_metric

        optimize_mode = self.model_task.optimize_mode
        tune_model_metric = checkpoint[metric_name]
        training_loss = checkpoint['training_loss']
        tune_sum_loss = tune_model_metric + training_loss
        tune_weight_loss = tune_model_metric * 0.8 + training_loss * 0.2

        model_arch = self.model_task.model_arch
        best_model = checkpoint['best_model']

        # 首先启动 mlflow 的 session
        # =============================================
        run_name = f'{model_arch}_best_model_and_config'
        mlflow = setup_mlflow(run_name=run_name, **self.mlflow_config,)
        mlflow_client = MlflowClient(mlflow.get_tracking_uri())

        if mlflow_utils.check_model_existence(reg_model_name):
            # 使用 get_best_model_version 加载最优模型的版本信息
            # 模型训练阶段不删除历史模型, 推理的时候过滤, 避免训练任务失败
            model_info = mlflow_utils.get_best_model_version(
                reg_model_name, metric_name, optimize_mode, delete=False)

            if model_info is not None:
                best_model_version = model_info['version']
                logging.warning(f'✅ save checkpoint pipeline 历史最佳注册模型版本: {best_model_version}')

                try:
                    hist_regis_model = model_frame.load_model(f"models:/{reg_model_name}/{best_model_version}")
                    # 测试历史模型。当测试的序列数据特征工程切分异常时，表明最新数据已经变化，需要抛弃历史模型
                    # =========================================================================
                    hist_training_loss, hist_eval_metric = self.test_hist_model(
                        reg_model_name, model_version=best_model_version, model_frame=model_frame)
                except any([
                    No_SeqDataException,
                    No_MLflow_Model_Found_Exception,
                    RestException,
                    ]):
                        logging.warning(f'❌ 数据源切分错误、或者找不到 mlflow 注册模型 ...')
                        exception_value = {
                            'min': np.inf,
                            'max': -np.inf,
                            }
                        hist_training_loss = hist_eval_metric = exception_value[optimize_mode]

                # 更新评估指标的权重
                hist_sum_loss = hist_training_loss + hist_eval_metric
                hist_weight_loss = hist_training_loss * 0.2 + hist_eval_metric * 0.8

                # NOTE: 比较测试指标
                if loss_strategy == 'UNIT':
                    if optimize_mode == 'min':
                        compare_bools_result = -tune_model_metric <= -hist_eval_metric
                    else:
                        compare_bools_result = tune_model_metric <= hist_eval_metric
                elif loss_strategy == 'WEIGHT':
                    if optimize_mode == 'min':
                        compare_bools_result = -tune_weight_loss <= -hist_weight_loss
                    else:
                        compare_bools_result = tune_weight_loss <= hist_weight_loss
                else:
                    if optimize_mode == 'min':
                        compare_bools_result = -tune_sum_loss <= -hist_sum_loss
                    else:
                        compare_bools_result = tune_sum_loss <= hist_sum_loss

                # 默认使用最大化模式
                if compare_bools_result:
                    # 更新输出的模型
                    self.output_model = hist_regis_model

                    logging.warning(f'''
                        tune {model_arch} model {metric_name}: {tune_model_metric:,.3f}, {loss_strategy} loss: {tune_sum_loss:,.3f}
                        hist {model_arch} model {metric_name}: {hist_eval_metric:,.3f}, {loss_strategy} loss: {hist_sum_loss:,.3f}
                        --> 使用历史最优模型推理......
                        ''')
                    return {
                        'training_loss': hist_training_loss,
                        'test_loss': hist_eval_metric,
                        'best_model': hist_regis_model,
                        'save_mode': 'hist',
                        }
                else:
                    # 将针对数据实例的更改撤回。不删除历史模型
                    # mlflow_client.delete_registered_model(reg_model_name)
                    logging.warning(f'''
                        test loss vs ------> hist: {hist_eval_metric:,.6f}, new: {tune_model_metric:,.6f}.
                        sum loss vs  ------> hist: {hist_sum_loss:,.6f}, new: {tune_sum_loss:,.6f}.
                        历史模型评分低, 将保存当前的模型...
                        ''')
            else:
                logging.warning(f'历史最优模型已经被更新删除，使用当前模型注册...')
        else:
            logging.warning(f'没有注册的历史模型......')

        # TODO: 之前应该是误会了，评估指标可以为 0, 不代表训练失败
        # if tune_model_metric == 0:
        #     logging.warning(f'新模型同时训练失败,请调整数据,重新训练......')
        #     raise Exception

        # 在 hist 步，将 dataset_inst 调整到了 hist 模式
        # 因此, 如果 tune 模式最优，则将 dataset_inst 更新为当前最优配置
        if self.dataset_inst is not None:
            self.dataset_inst.set_attr(self.best_data_args)

        # logging.warning(f'best_data_args params ---------> {self.best_data_args}')

        # 记录模型的 model 和 datainst 关键参数
        params_config = copy(self.best_model_args)
        logging.warning(f'--------> best model args: {self.best_model_args}')
        logging.warning(f'--------> best data args: {self.best_data_args}')

        params_config['training_loss'] = training_loss
        params_config[f'test_{metric_name}'] = tune_model_metric
        # 登记模型的注册日期
        params_config['regist_date'] = time.strftime('%Y-%m-%d')

        params_config.update(self.best_data_args)
        logging.warning(f'---------> params_config: {params_config}')

        if model_arch == 'nn':
            best_model.eval()
        self.output_model = best_model

        # logging.warning(f'--------> test data type: {type(self.test_data)}')
        if model_arch in ['xgb']:
            test_data = ray.get(self.test_data)
        elif any([
            isinstance(self.test_data, pd.DataFrame),
            isinstance(self.test_data, np.ndarray),
            ]):
            test_data = self.test_data if len(self.test_data) > 0 else self.train_data
        else:
            test_data = self.test_data if self.test_data else self.train_data

        test_loader, signature = self.data_util_map(test_data, params_config=params_config)
        # with mlflow.start_run(run_name=run_name):
        # TODO: current path: file:///home/zorro/project/pycharm/mlruns/0/01de69589f3b45df8c6111899175b97c/artifacts
        logging.warning(f'register model uri: {mlflow.get_registry_uri()}')
        logging.warning(f'tracking model uri: {mlflow.get_tracking_uri()}')

        try:
            mlflow_client.delete_registered_model(reg_model_name)
        except:
            logging.warning(f'------> 应删除的 reg_model_name: {reg_model_name} 版本已被清空, 忽略...')

        model_info = model_frame.log_model(
            best_model,
            artifact_path='models',
            signature=signature,
            registered_model_name=reg_model_name,
            )
        # log 记录 model & data best args
        mlflow.log_params(params_config)

        mlflow.log_metric(f'test_{metric_name}', tune_model_metric)
        mlflow_client.set_registered_model_alias(reg_model_name, model_alias, '1')
        mlflow_client.set_registered_model_alias(reg_model_name, model_arch, '1')

        mlflow_client.set_registered_model_tag(
            reg_model_name, f'test_{metric_name}', str(round(tune_model_metric, 6)))
        mlflow_client.set_registered_model_tag(
            reg_model_name, f'training_loss', str(round(training_loss, 6)))

        # 结束 mlflow session
        mlflow.end_run()

        logging.warning(f'''
            ----> 使用当前模型推理
            {model_arch} model test {metric_name}: {tune_model_metric:,.3f}
            ''')
        logging.warning(f'当前训练的 {model_arch} 模型已保存.')
        return {
            'training_loss': round(training_loss, 6),
            'test_loss': round(tune_model_metric, 6),
            'best_model': best_model,
            'save_mode': 'new',
            }
