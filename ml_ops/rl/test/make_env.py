import gymnasium as gym
import logging

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dir_list = [dname for dname in current_dir.as_posix().split('/')]
home_dir = '/'.join([dirname for dirname in dir_list[:dir_list.index('zorro')+1]])
env_path = '/'.join([dirname for dirname in dir_list[:dir_list.index('pycharm')+1]])
sys.path.append(env_path)

from mlops.ml_ops.rl import finrl
from tools.DB_Client import DB_Client
mysql_db_client = DB_Client(con_type='mysql_centos')


"""
@Desc: 本代码测试构建 gym 环境模型
"""


sql = '''
    select
         trade_date     as `date`
        ,scode          as tic
        ,open * 1       as open
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

stock_data.to_csv(Path(current_dir) / 'stock.csv', index=False)
logging.warning(f'\n{stock_data.head()}')

stock_nums = len(stock_data.tic.unique())

env_config = dict(
    df=stock_data,
    num_stock_shares=[0] * stock_nums,
    buy_cost_pct=[3/10000, 3/10000],
    sell_cost_pct=[3/10000, 3/10000],
    )
# 加载已经注册的 env
gym.make('sb3_StockTradeEnv-v0', **env_config)


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

env_config = {'config': env_kwargs}
rllib_trade_env = gym.make('rllib_StockTradeEnv-v0', **env_config)