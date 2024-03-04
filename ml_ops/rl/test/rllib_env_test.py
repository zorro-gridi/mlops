from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import MemoryTrackingCallbacks
from ray.tune.registry import register_env
import gymnasium as gym
import time

import logging
import ray

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dir_list = [dname for dname in current_dir.as_posix().split('/')]
home_dir = '/'.join([dirname for dirname in dir_list[:dir_list.index('zorro')+1]])
env_path = '/'.join([dirname for dirname in dir_list[:dir_list.index('pycharm')+1]])
sys.path.append(env_path)

from mlops.ml_ops.rl import finrl
from mlops.ml_ops.rl.finrl.envs.rllib_StockTradeEnv import StockTradeEnv
from tools.DB_Client import DB_Client

mysql_db_client = DB_Client(con_type='mysql_centos')


sql = '''
    select
         trade_date as `date`
        ,scode      as tic
        ,open * 1   as open
        ,closed * 1     as close
        ,highest * 1    as high
    from stock.east_money_stock_trade_data
    where 1=1
        and scode in ('600036', '600900')
        and trade_date >= '2023-01-01'
    order by
         `date`
        ,tic
'''

stock_data = mysql_db_client.data_read(sql)
# factorize() 将一组数据编码成整数编码
stock_data.index = stock_data.date.factorize()[0]
logging.warning(f'\n{stock_data.head()}')

env_kwargs = dict(
    df=stock_data,
    initial_amount=100000,
    # 初始持仓。 注意：持仓 & 交易手续费 的顺序要和 dataframe 中的股票名称顺序一致
    num_stock_shares=[0, 0],
    buy_cost_pct=[3/10000, 3/10000],
    sell_cost_pct=[3/10000, 3/10000],
    reward_scaling=0.6, # reward discount 系数
    tech_indicator_list=None,
    print_verbosity=10,
    day=0,
    initial=True,
    window_size=30,
    future_days=30,
    hmax=200,
    model_name="rl_trade_bot",
    mode="train",
    iteration=1000,
    per_buy_order_max_amt=20000, # 单笔买入的最大金额
    per_unit_qty=100, # 单笔交易最小交易量，例如股票100股
    per_unit_amount=10, # 单笔交易的最小金额，例如基金1～10元
    output_dir=Path(env_path) / 'rl_results',
    )

if ray.is_initialized():
    ray.shutdown()
ray.init(num_gpus=1, num_cpus=10, dashboard_host='0.0.0.0')

env_config = {'config': env_kwargs}
rllib_trade_env = gym.make('rllib_StockTradeEnv-v0', **env_config)

def env_creator(env_config):
    return rllib_trade_env

env_name = 'rllib_env'
register_env(env_name, env_creator)


config = ( # 1. Configure the algorithm,
    PPOConfig()
    # .environment(StockTradeEnv, env_config=env_kwargs)
    .environment(env=env_name)
    .rollouts(num_rollout_workers=8, batch_mode='complete_episodes')
    # https://docs.ray.io/en/latest/rllib/rllib-torch2x.html#exploration
    .framework(
        "torch",
        torch_compile_worker=True,
        torch_compile_worker_dynamo_backend="ipex",
        torch_compile_worker_dynamo_mode="default",
        )
     # num_gpus: for learning task
     # num_learner_workers & num_gpus_per_learner_worker for collection samples task
     # avoid for compute intensive or dataset size too large
    .resources(num_gpus=1, num_cpus_per_worker=1,)
    # 策略网络的参数
    # 每次打印的 eposode 信息并不是在训练，应该是达到 batch_size 的大小后，gpu 才开始训练
    .training(model={"fcnet_hiddens": [64, 64]}, train_batch_size=1024 * 4,)
    # .callbacks(MemoryTrackingCallbacks)
    # .evaluation(evaluation_interval=100, evaluation_num_workers=1)
    )

algo = config.build()  # 2. build the algorithm,

start_time = time.time()
time_steps = 1000

for _ in range(time_steps):
    epoch_start_time = time.time()
    results = algo.train()  # 3. train it,
    time.sleep(0.5)
    epoch_end_time = time.time()
    logging.warning(f'''
        training round No. {_+1}, round time: {(epoch_end_time - epoch_start_time):.1f}s, total_time: {(epoch_end_time - start_time):.1f}s''')

# algo.evaluate()  # 4. and evaluate it.


# trade_env = StockTradeEnv(config=env_kwargs)
# trade_env.reset()
# for i in range(1000):
#     actions = trade_env.action_space.sample()
#     state, reward, teminated, truncated, acct_info = trade_env.step(actions)

#     if teminated or truncated:
#         print(reward, acct_info)
#         print(trade_env.asset_memory[-1])
#         print(sum(trade_env.rewards_memory))
#         break
