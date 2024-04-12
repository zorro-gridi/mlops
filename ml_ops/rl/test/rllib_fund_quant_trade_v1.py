# %%
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.logger import pretty_print

from ray.rllib.algorithms.callbacks import MemoryTrackingCallbacks
from ray.tune.registry import register_env
import gymnasium as gym
import time
import pandas as pd

import logging
import ray
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
# from mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V2 import FundQuantTradeEnv_V2
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
    goal_yield=0.07,              # 训练模式，不用设置
    phase_yield=0.02,
    )


# %%
env_config = {'config': env_kwargs}
rllib_fund_env = gym.make('rllib_FundQuantTradeEnv-v0.1', **env_config)
ray.rllib.utils.check_env(rllib_fund_env)


# %%
# if ray.is_initialized():
#     ray.shutdown()
# ray.init(num_gpus=1, num_cpus=18, dashboard_host='0.0.0.0')

# 连接 ray start --head 已启动的 ray server
ray.init(address='auto')


# %%
def env_creator(env_config):
    return rllib_fund_env

env_name = 'rllib_fund_env'
register_env(env_name, env_creator)


# %%
config = ( # 1. Configure the algorithm,
    PPOConfig()
    # 直接传参
    # .environment(FundQuantTradeEnv, env_config=env_kwargs)
    .environment(env=env_name)
    .rollouts(
        num_rollout_workers=12,
        # batch_mode='complete_episodes',
        # num_envs_per_worker=2,
        # remote_worker_envs=True,
        )
     # num_gpus: for learning task
     # num_learner_workers & num_gpus_per_learner_worker for collection samples task
     # avoid for compute intensive or dataset size too large
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
    .training(model={"fcnet_hiddens": [64, 64]}, train_batch_size=int(512),)
    # .callbacks(MemoryTrackingCallbacks)
    .evaluation(evaluation_interval=100, evaluation_num_workers=1,)
    .offline_data()
    )


# %%
checkpoint_dir = Path(current_dir) / 'trader_bot/Media/V1'

# if checkpoint_dir.exists() and any(checkpoint_dir.glob('*')):
#     algo = Algorithm.from_checkpoint(checkpoint_dir)
#     logger.warning(f'Resume from Checkpoint -----------> {checkpoint_dir}')
# else:
    # algo = config.build()

# algo = config.build()


# # %%
# start_time = time.time()
# time_steps = 1000

# for epoch in range(time_steps):
#     epoch_start_time = time.time()
#     results = algo.train()
#     epoch_end_time = time.time()

#     epoch_time = (epoch_end_time - epoch_start_time)
#     total_time = (epoch_end_time - start_time)
#     logger.warning(f'''training round No. {epoch+1}, round time: {epoch_time:.1f}s, total_time: {total_time:.1f}s''')


# # %%
# shutil.rmtree(checkpoint_dir, ignore_errors=True)
# save_result = algo.save(checkpoint_dir=checkpoint_dir)

# path_to_checkpoint = save_result.checkpoint.path
# print(
#     "An Algorithm checkpoint has been created inside directory: "
#     f"'{path_to_checkpoint}'."
#     )

# # %%
# # Let's terminate the algo for demonstration purposes.
# algo.stop()


# inference:
# ========================================
# %%
algo = Algorithm.from_checkpoint(checkpoint_dir)
fund_env = FundQuantTradeEnv(config=env_kwargs)
# reset 默认返回两个元素
obs, acct_info = fund_env.reset(seed=42)


# %%
while True:
    actions = algo.compute_single_action(obs)
    state, reward, teminated, truncated, acct_info = fund_env.step(actions)

    if teminated or truncated or fund_env.goal_achieved:
        logger.warning(f'finished!')
        # logger.warning(f'reward: {reward}')
        # print(acct_info)

        holding_list = []
        acct_holdings = fund_env.acct_info['pfo_shares_redeem']
        for tic, holdings in acct_holdings.items():
            tic_holdings = pd.DataFrame(holdings)
            tic_holdings['tic'] = tic
            holding_list.append(tic_holdings)

        acct_holdings_df = pd.concat(holding_list)
        acct_holdings_df['sold_shares'] = acct_holdings_df['shares'] - acct_holdings_df['hold']

        logger.warning(f'''
            pfo ratio: {fund_env._get_pfo_ratio()}
            acct cash: {sum(fund_env.acct_info['cash_asset']):0.2f}
            pfo asset: {fund_env._get_acct_pfo_shares()[1]:0.2f}
            acct asset: {fund_env.asset_memory[-1]:0.2f}
            acct debit cost yield: {fund_env._get_acct_cumsum_yield():0.4f}
            total_reward: {sum(fund_env.rewards_memory):0.2f}
            cost: {fund_env.cost:0.2f}
            trades: {fund_env.trades}
            acct holding: \n{acct_holdings_df}
            soldout: {fund_env.soldout}
            yield goal achieved: {fund_env.goal_achieved}
            ''')
        break


# %%
idx_point_df = stock_data[['date', 'tic', 'close', 'closed']]
buy_shares_df = acct_holdings_df.groupby(
    by=['buy_date', 'tic'], as_index=False)['shares'].sum()
selling_shares_df = acct_holdings_df.groupby(
    by=['selling_date', 'tic'], as_index=False)['sold_shares'].sum()


backtest_df = pd.merge(idx_point_df, buy_shares_df,
    how='left', left_on=['date', 'tic'], right_on=['buy_date', 'tic']
    ).merge(selling_shares_df,
        how='left', left_on=['date', 'tic'], right_on=['selling_date', 'tic']
    ).drop(columns=['buy_date', 'selling_date'])

backtest_df['close'] = backtest_df['close'].cumsum()
logging.warning(f'backtest df ------>\n{backtest_df}')

to_cons = ['mysql_centos', 'pg_centos', 'pg_aliyun']
for con in to_cons:
    db_client = DB_Client(con_type=con)
    db_client.data_load(
        df=backtest_df,
        schema='report',
        table_name='gridi_quant_fund_strategy_backtest',
        operation='replace',
        )

backtest_df.to_csv(current_dir / 'gridi_quant_fund_strategy_backtest.csv', index=False)