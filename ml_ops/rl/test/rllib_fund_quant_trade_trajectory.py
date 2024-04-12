# %%
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.pg import PGConfig
from ray.rllib.algorithms.dqn import DQNConfig

from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.logger import pretty_print

from ray.rllib.algorithms.callbacks import MemoryTrackingCallbacks
from ray.tune.registry import register_env
import gymnasium as gym
import time
import pandas as pd

import logging
import ray
from ray.rllib.utils import torch_utils
import shutil

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dir_list = [dname for dname in current_dir.as_posix().split('/')]
home_dir = '/'.join([dirname for dirname in dir_list[:dir_list.index('zorro')+1]])
env_path = '/'.join([dirname for dirname in dir_list[:dir_list.index('pycharm')+1]])
sys.path.append(env_path)

# finrl 必须导入到 runtime
from mlops.ml_ops.rl import finrl
from mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V1 import FundQuantTradeEnv_V1
from mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V2 import FundQuantTradeEnv_V2
from tools.DB_Client import DB_Client


logger = logging.getLogger(__name__)
logging.basicConfig(filename=current_dir / 'fund_trader_bot.log', level=logging.WARNING)


"""
@Desc:
    本代码实现基于 rllib 框架的交易策略训练测试
"""


# %%
sql = '''
    select
        *
    from idx.index_markup_days_rl_observations_dataset
    where 1=1
        and indexname = '传媒'
    order by
         trade_date
        ,indexname
'''

mysql_db_client = DB_Client(con_type='mysql_centos')
stock_data = mysql_db_client.data_read(sql)
stock_data.rename(columns={
    'trade_date': 'date',
    'indexname': 'tic',
    'markup': 'close',
    }, inplace=True)
stock_data.drop(columns=['model_frame'], inplace=True)

# factorize() 将一组数据编码成整数编码
stock_data.index = stock_data.date.factorize()[0]
logger.warning(f'\n{stock_data.tail()}')
logger.warning(f'\n{stock_data.head()}')


# %%
env_kwargs = dict(
    df=stock_data,
    initial_amount=100000,
    num_stock_shares=[0],           # 初始持仓。 注意：持仓 & 交易手续费 的顺序要和 dataframe 中的股票名称顺序一致
    buy_cost_pct=[1.5/1000],
    sell_cost_pct=[None],
    reward_scaling=0.6,             # reward discount 系数
    tech_indicator_list=None,
    print_verbosity=10,
    day=0,
    initial=True,
    window_size=1,
    future_days=5,
    model_name="rl_fund_bot",
    mode="train",
    iteration=1000,
    per_unit_qty=10,                # 单笔交易最小交易量，例如股票100股
    per_unit_amount=200,            # 单笔交易的最小金额，例如基金1～10元
    output_dir=Path(env_path) / 'rl_results',
    # goal_yield=0.07,              # 训练模式，不用设置
    phase_yield=0.02,
    )


# %%
env_config = {'config': env_kwargs}
rllib_fund_env = gym.make('rllib_FundQuantTradeEnv-v0.1', **env_config)
ray.rllib.utils.check_env(rllib_fund_env)


# %%
# 连接 ray start --head 已启动的 ray server
ray.init(address='auto')


# %%
def env_creator(env_config):
    return rllib_fund_env

env_name = 'rllib_fund_env_trajectory'
register_env(env_name, env_creator)


# %%
config = ( # 1. Configure the algorithm,
    PPOConfig()
    # PGConfig()
    # 直接传参
    .environment(env=env_name)
    .rollouts(
        num_rollout_workers=12,
        )
    .resources(
        num_gpus=0,
        num_learner_workers=8,
        num_gpus_per_learner_worker=0.1,
        num_cpus_per_worker=1,
        num_gpus_per_worker=0,
        )
    # https://docs.ray.io/en/latest/rllib/rllib-torch2x.html#exploration
    .framework(
        "torch",
        torch_compile_worker=True,
        torch_compile_worker_dynamo_backend="ipex",
        torch_compile_worker_dynamo_mode="default",
        )
    # 策略网络的参数
    # 每次打印的 eposode 信息并不是在训练，应该是达到 batch_size 的大小后，gpu 才开始训练
    .training(model={"fcnet_hiddens": [64, 64]}, train_batch_size=int(512), vf_clip_param=100)
    # .callbacks(MemoryTrackingCallbacks)
    # .evaluation(evaluation_interval=100, evaluation_num_workers=1,)
    .offline_data(input_='/tmp/rllib_fund_trade_bot_out')
    # .offline_data(output='/tmp/rllib_fund_trade_bot_out')
    )

algo = config.build()
for _ in range(1000):
    algo.train()
    logging.warning(f'training loop: {_+1}')


# %%
