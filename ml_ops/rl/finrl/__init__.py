from gymnasium.envs.registration import register
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dir_list = [dname for dname in current_dir.as_posix().split('/')]
home_dir = '/'.join([dirname for dirname in dir_list[:dir_list.index('zorro')+1]])
env_path = '/'.join([dirname for dirname in dir_list[:dir_list.index('pycharm')+1]])
sys.path.append(env_path)


"""
@Desc:
    本代码注册已定义的 gym env 实例到 gymnasium env register 注册表
"""


register(
    id='sb3_StockTradeEnv-v0',
    entry_point='mlops.ml_ops.rl.finrl.envs.SB3_StockTradeEnv:StockTradeEnv',
    max_episode_steps=300,
    )


register(
    id='rllib_StockTradeEnv-v0',
    entry_point='mlops.ml_ops.rl.finrl.envs.rllib_StockTradeEnv:StockTradeEnv',
    max_episode_steps=300,
    )


register(
    id='rllib_FundQuantTradeEnv-v0',
    entry_point='mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv:FundQuantTradeEnv',
    max_episode_steps=300,
    )