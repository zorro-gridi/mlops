import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dir_list = [dname for dname in current_dir.as_posix().split('/')]
home_dir = '/'.join([dirname for dirname in dir_list[:dir_list.index('zorro')+1]])
env_path = '/'.join([dirname for dirname in dir_list[:dir_list.index('pycharm')+1]])
sys.path.append(env_path)

from mlops.ml_ops.rl.finrl.envs.rllib_StockTradeEnv import StockTradeEnv
from mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V1 import FundQuantTradeEnv_V1
from mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V2 import FundQuantTradeEnv_V2
from mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V3 import FundQuantTradeEnv_V3
from mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V4 import FundQuantTradeEnv_V4
from mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V5 import FundQuantTradeEnv_V5
from mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V6 import FundQuantTradeEnv_V6


"""
@Desc:
    本代码实现将已经定义的 gym env class 导入到 runtime 环境
"""