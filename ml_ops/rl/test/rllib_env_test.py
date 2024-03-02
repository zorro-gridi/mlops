from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import MemoryTrackingCallbacks
from ray.tune.registry import register_env

import logging
import ray

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dir_list = [dname for dname in current_dir.as_posix().split('/')]
home_dir = '/'.join([dirname for dirname in dir_list[:dir_list.index('zorro')+1]])
env_path = '/'.join([dirname for dirname in dir_list[:dir_list.index('pycharm')+1]])
sys.path.append(env_path)

from mlops.ml_ops.rl.finrl.env.rllib_SimpleStockTradeEnv import StockTradeEnv
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
        and scode in ('600036')
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
    make_plots=False,
    print_verbosity=10,
    day=0,
    initial=True,
    window_size=10,
    future_days=30,
    hmax=100,
    model_name="",
    mode="",
    iteration="",
    per_buy_order_max_amt=20000, # 单笔买入的最大金额
    per_unit_qty=100, # 单笔交易最小交易量，例如股票100股
    per_unit_amount=10, # 单笔交易的最小金额，例如基金1～10元
    )

# ray.init()

# def env_creator(env_config):
#     return StockTradeEnv(config=env_kwargs)

# env_name = 'stock_trade_env_0.1'
# register_env(env_name, env_creator)

config = ( # 1. Configure the algorithm,
    PPOConfig()
    .environment(StockTradeEnv, env_config=env_kwargs)
    # .environment(env=env_name)
    .rollouts(num_rollout_workers=2)
    .framework("torch")
    .training(model={"fcnet_hiddens": [64, 64]}) # 策略网络的参数(策略也是一个神经网路)
    .resources(num_gpus=0, num_cpus_per_worker=4)
    .callbacks(MemoryTrackingCallbacks)
    .evaluation(evaluation_interval=100, evaluation_num_workers=1)
    )

algo = config.build()  # 2. build the algorithm,

for _ in range(10):
    results = algo.train()  # 3. train it,

# algo.evaluate()  # 4. and evaluate it.

# ray.shutdown()