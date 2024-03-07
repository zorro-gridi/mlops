from mlflow.client import MlflowClient
import mlflow
import logging

tracking_uri = 'http://192.168.1.105:9001/'

def check_model_existence(model_name, tracking_uri=tracking_uri):
    # 获取注册的模型的所有版本
    mlflow_client = MlflowClient(tracking_uri)
    registered_models = [
        dict(rm)['name'] for rm in mlflow_client.search_registered_models()]

    return model_name in registered_models



def load_register_model_args(reg_model_name, model_version, tracking_uri=tracking_uri):
    mlflow_client = MlflowClient(tracking_uri)
    hist_model_uri = mlflow_client.get_model_version_download_uri(reg_model_name, model_version)
    hist_model_info = mlflow.models.get_model_info(hist_model_uri)
    hist_model_signature_dict = hist_model_info._signature_dict
    params_list = eval(
        hist_model_signature_dict['params'].replace('null', 'None').replace('true', 'True').replace('false', 'False'))
    hist_model_args = {param['name']: param['default'] for param in params_list}

    logging.warning(f'hist model params details: {hist_model_args}')
    return hist_model_args