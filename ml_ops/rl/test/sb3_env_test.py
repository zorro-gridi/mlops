import logging
from stable_baselines3 import SAC, TD3 # for contunous actions
from stable_baselines3 import A2C # 离散空间 faster

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dir_list = [dname for dname in current_dir.as_posix().split('/')]
home_dir = '/'.join([dirname for dirname in dir_list[:dir_list.index('zorro')+1]])
env_path = '/'.join([dirname for dirname in dir_list[:dir_list.index('pycharm')+1]])
sys.path.append(env_path)

from mlops.ml_ops.rl.finrl.env.SB3_StockTradeEnv import StockTradeEnv
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
    hmax=100,
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
    )

trade_env = StockTradeEnv(df=stock_data, **env_kwargs)
state, acct_info = trade_env.reset()

# for i in range(1000):
#     actions = trade_env.action_space.sample()
#     state, reward, teminated, truncated, acct_info = trade_env.step(actions)

#     if teminated or truncated:
#         print(reward, acct_info)
#         print(trade_env.asset_memory[-1])
#         print(sum(trade_env.rewards_memory))
#         break

# print(state)
# trade_env.close()

# model = SAC('MlpPolicy', trade_env, verbose=1)
# model = TD3('MlpPolicy', trade_env, verbose=1)
model = A2C('MlpPolicy', trade_env, verbose=1)
model.learn(total_timesteps=100_000)