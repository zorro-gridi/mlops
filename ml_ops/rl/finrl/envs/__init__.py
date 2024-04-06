import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dir_list = [dname for dname in current_dir.as_posix().split('/')]
home_dir = '/'.join([dirname for dirname in dir_list[:dir_list.index('zorro')+1]])
env_path = '/'.join([dirname for dirname in dir_list[:dir_list.index('pycharm')+1]])
sys.path.append(env_path)

from mlops.ml_ops.rl.finrl.envs.rllib_StockTradeEnv import StockTradeEnv
from mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv import FundQuantTradeEnv
from mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V2 import FundQuantTradeEnv_V2


"""
@Desc:
    本代码实现将已经定义的 gym env class 导入到 runtime 环境
"""