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
    id='sb3_StockTradeEnv-v0.2',
    entry_point='mlops.ml_ops.rl.finrl.envs.SB3_StockTradeEnv:StockTradeEnv',
    max_episode_steps=300,
    )


register(
    id='rllib_StockTradeEnv-v0.2',
    entry_point='mlops.ml_ops.rl.finrl.envs.rllib_StockTradeEnv:StockTradeEnv',
    max_episode_steps=300,
    )


register(
    id='rllib_FundQuantTradeEnv-v0.1',
    entry_point='mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V1:FundQuantTradeEnv_V1',
    max_episode_steps=300,
    )


register(
    id='rllib_FundQuantTradeEnv-v0.2',
    entry_point='mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V2:FundQuantTradeEnv_V2',
    max_episode_steps=300,
    )


register(
    id='rllib_FundQuantTradeEnv-v0.3',
    entry_point='mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V3:FundQuantTradeEnv_V3',
    max_episode_steps=300,
    )


register(
    id='rllib_FundQuantTradeEnv-v0.4',
    entry_point='mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V4:FundQuantTradeEnv_V4',
    max_episode_steps=300,
    )


register(
    id='rllib_FundQuantTradeEnv-v0.5',
    entry_point='mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V5:FundQuantTradeEnv_V5',
    max_episode_steps=300,
    )


register(
    id='rllib_FundQuantTradeEnv-v0.6',
    entry_point='mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V6:FundQuantTradeEnv_V6',
    max_episode_steps=300,
    )