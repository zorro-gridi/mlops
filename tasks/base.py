from abc import ABCMeta, abstractclassmethod
from sklearn import metrics
import numpy as np



class AbstractModelFactory(metaclass=ABCMeta):

    def __init__(self,
            model_loss_func=None,
            model_eval_metric=None,
            model_init_params={},
            model_train_params={},
            optimize_mode='min',
            custom_loss_func=None,
        ):
        '''
        # model_loss_func: 模型损失函数
        # model_eval_metric: 模型评估指标名称
        # model_init_params: 模型初始化参数
        # model_train_params: 提供给 train 方法的参数
        # custom_loss_func: 非标准库的自定义损失函数
        '''
        self.model_loss_func = model_loss_func
        self.model_eval_metric = model_eval_metric
        self.model_init_params = model_init_params
        self.model_train_params = model_train_params
        self.optimize_mode = optimize_mode
        self.custom_loss_func = custom_loss_func


    @abstractclassmethod
    def train_job(self, *args, **kwargs):
        pass

    @abstractclassmethod
    def tune_job(self, *args, **kwargs):
        pass


    @abstractclassmethod
    def eval_job(self, y_true, y_pred, metric_name, tasktype='binary_clf', **kwargs):
        metric_config = {
            'auc': metrics.roc_auc_score,
            'recall': metrics.recall_score,
            'precision': metrics.precision_score,
            'accuracy': metrics.accuracy_score,
            'rmse': metrics.mean_squared_error,
            }

        if tasktype in ['binary_clf']:
            if metric_name == 'auc':
                test_score = metric_config[metric_name](y_true, y_pred, **kwargs)
            else:
                y_label = np.where(y_pred > 0.5, 1, 0)
                test_score = metric_config[metric_name](y_true, y_label, **kwargs)
        else:
            test_score = metric_config[metric_name](y_true, y_pred, **kwargs)
        return test_score


    @abstractclassmethod
    def test_job(self, *args, **kwargs):
        pass
