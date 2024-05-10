import mlflow
from mlflow.client import MlflowClient
import logging
import numpy as np


tracking_uri = 'http://192.168.0.106:9001/'


def check_model_existence(model_name, tracking_uri=tracking_uri):
    '''
    Desc:
        判断模型是否存在注册表中
    Retrun:
        bool
    '''
    # 获取注册的模型的所有版本
    mlflow_client = MlflowClient(tracking_uri)
    registered_models = [
        dict(rm)['name'] for rm in mlflow_client.search_registered_models()]
    return model_name in registered_models


def load_register_model_args(reg_model_name, model_version, tracking_uri=tracking_uri):
    '''
    Desc:
        加载注册模型的参数
    Return:
        model 的参数字典, 包括model 参数，数据集参数...
    '''
    mlflow_client = MlflowClient(tracking_uri)
    hist_model_uri = mlflow_client.get_model_version_download_uri(reg_model_name, model_version)
    hist_model_info = mlflow.models.get_model_info(hist_model_uri)
    hist_model_signature_dict = hist_model_info._signature_dict
    params_list = eval(
        hist_model_signature_dict['params'].replace('null', 'None').replace('true', 'True').replace('false', 'False'))
    hist_model_args = {param['name']: param['default'] for param in params_list}

    logging.warning(f'hist model params details: {hist_model_args}')
    return hist_model_args


def get_best_model_version(reg_model_name, eval_metric, optimize_mode):
    '''
    Desc:
        获取最优模型的版本号
    Args:
        reg_model_name: 模型名称
        eval_metric: 模型的比较指标
    '''
    mlflow_client = MlflowClient(tracking_uri)
    best_loss = -np.inf if optimize_mode == 'max' else np.inf
    best_version = 1
    best_model_config = None

    for mv in mlflow_client.search_model_versions(f"name='{reg_model_name}'"):
        mv = dict(mv)
        cur_version = mv['version']
        model_config = load_register_model_args(reg_model_name, cur_version)
        eval_loss = model_config[f'test_{eval_metric}']

        if optimize_mode == 'max':
            if eval_loss > best_loss:
                best_loss = eval_loss
                best_version = cur_version
                best_model_config = model_config
            else:
                mlflow_client.delete_model_version(reg_model_name, cur_version)
        else:
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_version = cur_version
                best_model_config = model_config
            else:
                mlflow_client.delete_model_version(reg_model_name, cur_version)

    logging.warning(f'最优模型的版本号: {best_version}')
    return {
        'config': best_model_config,
        'version': best_version,
        }