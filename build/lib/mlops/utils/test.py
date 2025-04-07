import mlflow_utils
import mlflow


tracking_uri = 'http://192.168.5.7:9001/'
mlflow.set_tracking_uri(tracking_uri)

mlflow_utils.get_best_model_version('SZ50_bounch_and_reverse_pattern_recog_cat', 'MCC', 'max')