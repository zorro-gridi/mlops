import mlflow
from mlflow.client import MlflowClient
import logging
import numpy as np


tracking_uri = 'http://192.168.5.7:9001/'
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)


def load_model(model_frame, reg_model_name, model_version):
    hist_regis_model = model_frame.load_model(f"models:/{reg_model_name}/{model_version}")
    return hist_regis_model


def check_model_existence(model_name, tracking_uri=tracking_uri):
    '''
    Desc:
        判断模型是否存在注册表中
    Retrun:
        bool
    '''
    # 获取注册的模型的所有版本
    registered_models = [
        dict(rm)['name'] for rm in mlflow.search_registered_models()]
    return model_name in registered_models


def load_register_model_args(reg_model_name: str, model_version: str, tracking_uri=tracking_uri):
    '''
    Desc:
        加载注册模型的参数
    Return:
        model 的参数字典, 包括 model 参数，数据集参数...
    Remark:
        必须要要求 log_model 时传入 signature 参数
    NOTE:
        2.16.0 版本接口
    '''
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow_client = MlflowClient(tracking_uri)
    hist_model_uri = mlflow_client.get_model_version_download_uri(reg_model_name, model_version)
    hist_model_info = mlflow.models.get_model_info(hist_model_uri)
    hist_model_signature_dict = hist_model_info._signature_dict
    params_list = eval(
        hist_model_signature_dict['params'].replace('null', 'None').replace('true', 'True').replace('false', 'False'))
    hist_model_args = {param['name']: param['default'] for param in params_list}
    # logging.warning(f'hist model params details: {hist_model_args}')
    return hist_model_args


# def load_register_model_args(reg_model_name: str, model_version: str, tracking_uri=tracking_uri):
#     '''
#     Desc: 加载注册模型的参数
#     NOTE:
#         3.7.0 版本
#         有严重错误：私自将模型的保存参数全部改为字符串类型，导致模型读取任务失败
#     '''
#     mlflow.set_tracking_uri(tracking_uri)
#     mlflow.set_registry_uri(tracking_uri)

#     mlflow_client = MlflowClient(tracking_uri)

#     try:
#         # 直接通过运行信息获取参数
#         model_version_info = mlflow_client.get_model_version(reg_model_name, model_version)
#         run_id = model_version_info.run_id

#         # 获取运行的所有参数
#         run = mlflow_client.get_run(run_id)
#         params = dict(run.data.params)

#         # 同时获取指标信息
#         metrics = dict(run.data.metrics)

#         # 合并参数和指标
#         all_params = {**params, **metrics}

#         logging.warning(f'✅ 从运行 {run_id} 加载参数成功: {len(all_params)}个参数')
#         return all_params

#     except Exception as e:
#         logging.error(f"❌ 加载模型参数失败: {str(e)}")
#         return {}



def get_best_model_version(reg_model_name, eval_metric, optimize_mode, delete=True):
    '''
    Desc:
        获取最优模型的版本号
    Args:
        reg_model_name: 模型名称
        eval_metric: 模型的比较指标
        delete: 是否删除旧模型; 使用方法：
            1. 训练阶段请设为 False 保证模型库不出错 key 错误，训练最后阶段可以设为 True, 保存最优模型
            2. 预测阶段也设为 True, 保留最优
    '''
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow_client = MlflowClient(tracking_uri)
    # 初始化模型评分
    best_loss = -np.inf if optimize_mode == 'max' else np.inf

    best_version = None
    best_model_config = None
    delete_version = None

    # 搜索注册模型的所有版本号
    hist_models_info = mlflow_client.search_model_versions(f"name='{reg_model_name}'")
    if len(hist_models_info) == 0:
        return None
    logging.warning(f'✅ hist_models_info: {hist_models_info}')
    for idx, mv in enumerate(hist_models_info):
        mv = dict(mv)
        cur_version = mv['version']
        try:
            model_config = load_register_model_args(reg_model_name, cur_version)
        except:
            logging.warning(f'✅ {reg_model_name} 模型仓库没有找到版本: {cur_version}')
            continue
        # 获取该注册模型的测试评分
        eval_loss = model_config[f'test_{eval_metric}']

        if optimize_mode == 'max':
            if eval_loss > best_loss:
                best_version = cur_version
                best_model_config = model_config
            else:
                delete_version = cur_version
        else:
            if eval_loss < best_loss:
                best_version = cur_version
                best_model_config = model_config
            else:
                delete_version = cur_version

        # logging.warning(f'✅ best loss: {best_loss}, eval loss: {eval_loss}, best version: {best_version}')
        best_loss = eval_loss

        if delete and delete_version:
            try:
                mlflow_client.delete_model_version(reg_model_name, delete_version)
                logging.warning(f'❌ reg model name: {reg_model_name}, version: {delete_version} 已删除')
            except:
                logging.warning(f'❌ 历史次优模型 reg model name: {reg_model_name}, version: {delete_version} 已删除')

    logging.warning(f'最优模型的版本号: {best_version}, 评估指标: {eval_metric}: {best_loss}')
    if best_version and best_model_config:
        return {
            'model_name': reg_model_name,
            'config': best_model_config,
            'version': best_version,
            }
    else:
        return None



if __name__ == '__main__':
    get_best_model_version('Stock_Volume_Breakout_Model_Cat', 'MCC', 'max')
