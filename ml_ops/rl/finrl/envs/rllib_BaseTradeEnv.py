
from __future__ import annotations
import logging
from pathlib import Path

# from typing import List
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from ray.rllib.env import EnvContext

matplotlib.use("Agg")


"""
@Author: Zorro
@Date: 2024-01-01
@Desc:
    本代码定义了基于 rllib 强化学习训练框架的 gym 环境模型
"""


class BaseTradeEnv(gym.Env):
    """
    Desc:
        A base trading environment for OpenAI gym
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: EnvContext):
        '''
        config key of Args:
            df: pd.DataFrame, 必须包含以下字段:
                'date': trade date
                'tic': stock or fund code
                'close': stock or fund closed price
            stock_dim: 交易的股票数量
            hmax: action 表示交易的手数，一手 = 100股。 hmax 可设置为 100
            state_dim: 仅仅表示维度 int; 本质是输入的特征数量, 在本class中定义为 self.state 列表的长度
            action_dim:  仅仅表示维度 int; 本质是股票持仓的数量，或者说交易股票池的数量
            window_size: 输入序列的长度
            future_days: 使用未来多少天的数据计算买入的预期收益率
        '''
        self.day = config.get('day', 0)
        self.df = config['df']

        self.hmax = config['hmax']
        self.num_stock_shares = config['num_stock_shares']
        self.initial_amount = config['initial_amount']  # get the initial cash
        self.buy_cost_pct = config['buy_cost_pct']
        self.sell_cost_pct = config.get('sell_cost_pct', [None])
        self.reward_scaling = config['reward_scaling']
        self.tech_indicator_list = config['tech_indicator_list']

        # 当日的交易数据特征
        self.stock_pools = self.df.tic.unique()
        self.stock_dim = len(self.stock_pools)

        self.window_size = config['window_size']
        self.future_days = config['future_days']
        self.per_batch_size = self.stock_dim * self.window_size

        # 是否完全重置环境与账户信息
        self.initial = config['initial']

        # initalize state
        self.acct_info = self._initial_acct_info()
        # 先重新初始化状态
        self.state = self._initiate_state()

        # 定义 observation_space, 因为本 class 定义的环境只有1维
        # 因为股票市场的波动是完全随机的，这个环境预设其实没有意义，我们将已有的数据集理解为一个完全观察状态
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.state),), dtype="float32"
            )
        # 定义 action_space
        # 为什么可以是 box，连续区间？因为在 step 函数中，action = (action * self.hmax).astype(int)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,), dtype="float32")
        # self.action_space = spaces.Discrete(21, start=-10)

        self.terminal = False
        self.print_verbosity = config['print_verbosity']
        self.model_name = config['model_name']
        self.mode = config['mode']
        self.iteration = config['iteration']

        # initialize reward
        self.reward = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        self.soldout = 0
        self.goal_achieved = 0
        self.pfo_ratio = config.get('pfo_ratio', 1)
        self.goal_yield = config.get('goal_yield', 0.1)
        self.phase_yield = config.get('phase_yield', 0.02)

        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self._seed()

        self.per_buy_order_max_amt = config['per_buy_order_max_amt']
        self.per_unit_qty = config['per_unit_qty']       # 每笔交易最小的股数，股票为100
        self.per_unit_amount = config['per_unit_amount'] # 每笔最小的交易额，基金一般为10元

        self.output_dir = config['output_dir']
        if self.output_dir:
            if not Path(self.output_dir).exists():
                Path(self.output_dir).mkdir(exist_ok=True)


    def _sell_stock(self, index, action):
        '''
        Desc:
            卖出 action. 这个函数交易的是一个股票
        Args:
            index 是一个索引，从 self.state 中取出对应的股票的持仓份额 或着 股价
            action 是一个标量数值，表示针对制定 index 股票进行加减仓操作；在 self.action_space 中定义
        '''
        stock_name = self.current_data['tic'].to_list()[index]
        close_price = self.current_data['close'].to_list()[index]
        stock_shares = sum(self.acct_info['pfo_holding'][stock_name])

        # holding_price: 平均持仓成本
        pfo_asset = sum(np.array(self.acct_info['pfo_holding'][stock_name]) * np.array(self.acct_info['pfo_price'][stock_name]))
        holding_price = pfo_asset / stock_shares if stock_shares > 0 else close_price # 因为初始仓位为 0

        # 卖出的条件：TODO: 这个需要放在主程序中定义, BaseModel 不需要特别指定
        # 1. 是否有赚钱的持仓
        is_profit_price = close_price - np.array(self.acct_info['pfo_price'][stock_name]) >= 0
        # 2. 是否整体持仓收益为证
        is_toatl_profit = holding_price < close_price
        # 大于 0 的持仓流水为买入操作
        is_profit_shares = np.array(self.acct_info['pfo_holding'][stock_name]) > 0

        # 3. 存在未卖出的盈利持仓。计算累计可卖出的持仓
        total_profit_shares = sum(
            np.extract(
            [any([price, is_toatl_profit]) and share for price, share in zip(is_profit_price, is_profit_shares)],
            np.array(self.acct_info['pfo_holding'][stock_name])
            )
        )
        # 计算剩余可卖出的持仓
        profit_shares_rest = max(total_profit_shares - self.acct_info['profit_shares_sold'][stock_name], 0)

        # check if the stock is able to sell, for simlicity we just add it in techical index
        # 也就是说，对应的股票是否可以交易，在技术指标中内置了。因为可能有些股票当日停牌，不可交易
        def _do_sell_normal():
            '''
            Desc:
                定义卖出交易的前提逻辑，例如：账户是否还有持仓？股票当前是否可以交易？
            '''
            sell_num_shares = 0
            sell_amount = 0

            if (
                # 需要判断股票自身是否可交易
                1==1
                ):
                # 判断当前是否有该股票的持仓 & 股价是否大于 0
                if profit_shares_rest > 0 and close_price > 0 and stock_shares > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(abs(action), stock_shares, profit_shares_rest)

                    if sell_num_shares > 0:
                        # 记录累计已卖出的盈利头寸
                        # logging.warning(f'do sell stock action: {sell_num_shares} quantities.')
                        self.acct_info['profit_shares_sold'][stock_name] += sell_num_shares
                        # 计算卖出可获得的金额，考虑交易费用
                        sell_amount = close_price * sell_num_shares * (1 - self.sell_cost_pct[index])
                        # 卖出股票，仓位减少
                        self.acct_info['pfo_holding'][stock_name].append(-sell_num_shares)
                        self.acct_info['pfo_price'][stock_name].append(close_price)

                        # 卖出股票，现金账户增加金额
                        self.acct_info['cash_asset'].append(sell_amount)
                        self.cost += close_price * sell_num_shares * self.sell_cost_pct[index]
                        self.trades += 1
            return sell_num_shares, sell_amount

        sell_num_shares, sell_amount = _do_sell_normal()
        return sell_num_shares, sell_amount


    def _buy_stock(self, index, action):
        '''
        Desc:
            买入 action. 这个函数交易的是 1 个股票
        Args:
            index 是一个索引，从 self.state 中取出对应的股票的持仓份额 或着 股价
            action 是一个标量数值，表示针对制定 index 股票进行加减仓操作; 在 self.action_space 中定义
        '''
        stock_name = self.current_data.tic.to_list()[index]
        close_price = self.current_data.close.to_list()[index]
        stock_shares = sum(self.acct_info['pfo_holding'][stock_name])

        # holding_price: 平均持仓成本
        pfo_asset = sum(np.array(self.acct_info['pfo_holding'][stock_name]) * np.array(self.acct_info['pfo_price'][stock_name]))
        holding_price = round(pfo_asset / stock_shares, 3) if stock_shares > 0 else close_price # 因为初始仓位为 0

        def _do_buy():
            buy_num_shares = 0
            buy_amount = 0
            # check if the stock is able to buy
            if (
                # 判断当前这个股票是否可以交易（定义为技术指标，且为指标列表的第一个索引）
                1==1
                ):
                # 基于单笔最大交易限制的买入策略
                cash_asset = sum(self.acct_info['cash_asset'])
                available_cash = min(cash_asset, self.per_buy_order_max_amt)
                available_shares = available_cash // close_price

                # 计算可买入的最多股票数量（基于单笔交易金额限制的）
                if available_shares > 0:
                    # update balance
                    buy_num_shares = min(available_shares, action)

                    if buy_num_shares > 0:
                        buy_amount = close_price * buy_num_shares * (1 - self.buy_cost_pct[index])
                        buy_fee = close_price * buy_num_shares * self.buy_cost_pct[index]
                        if buy_amount >= self.per_unit_amount:
                            # 更新账户的可用本金
                            # 买入股票，现金账户减少金额, 并扣除手续费
                            self.acct_info['cash_asset'].append(-close_price * buy_num_shares-buy_fee)
                            # 买入股票，增加持仓
                            self.acct_info['pfo_holding'][stock_name].append(buy_num_shares)
                            self.acct_info['pfo_price'][stock_name].append(close_price)

                            # 更新买入的手续费
                            self.cost += buy_fee
                            # 更新交易频次，不能写在 step 函数中
                            self.trades += 1

            # 返回买入的份额数量
            return buy_num_shares, buy_amount
        buy_num_shares, buy_amount = _do_buy()
        return buy_num_shares, buy_amount


    def step(self, actions):
        '''
        Desc:
            在环境中执行一个动作
        Args:
            actions: 交易的份额列表
        '''
        # logging.warning(f'sample actions {actions}')
        # 是否 TimeLimit & truncate
        # 因为 self.df 的最后 30 行用来计算 cumulative reward，非训练数据需要剔除
        self.terminal = self.day == len(self.df.iloc[:-self.future_days].index.unique()) - self.per_batch_size
        begin_total_asset = self._get_acct_asset()

        if self.terminal or self.goal_achieved == 1:
            # 交易后的累计资产
            end_total_asset = begin_total_asset

            # 以下全部为辅助信息：
            # ==========================================================================
            df_total_value = pd.DataFrame(self.asset_memory)
            # tot_reward = 当前资产 - 起始资产 （策略当前的累计奖励值）
            tot_reward = end_total_asset - self.initial_amount
            # 账户资产流水 dataframe
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory

            # 计算账户资产每日变动pct
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
            if df_total_value["daily_return"].std() != 0:
                # 计算策略的 sharpe 值
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )

            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")

                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")
                # logging.warning(f'action history: \n{self.actions_memory}')

            # # reward 流水 dataframe
            # df_rewards = pd.DataFrame(self.rewards_memory)
            # df_rewards.columns = ["account_rewards"]
            # df_rewards["date"] = self.date_memory[:-1]

            # if (self.model_name != "") and (self.mode != ""):
            #     df_total_value.to_csv(
            #         self.output_dir / "account_value_{}_{}_{}.csv".format(
            #             self.mode, self.model_name, self.iteration
            #         ),
            #         index=False,
            #     )
            #     df_rewards.to_csv(
            #         self.output_dir / "account_rewards_{}_{}_{}.csv".format(
            #             self.mode, self.model_name, self.iteration
            #         ),
            #         index=False,
            #     )
            #     plt.plot(self.asset_memory, "r")
            #     plt.savefig(
            #         self.output_dir / "account_value_{}_{}_{}.png".format(
            #             self.mode, self.model_name, self.iteration
            #         )
            #     )
            # plt.close()
            # ==========================================================================
            return self.state, self.reward, self.terminal, True, self.acct_info

        else:
            # actions initially is scaled between 0 to 1
            # self.hmax 表示每一笔交易需要买入的最低股票数量
            # 不能买入分数的份额
            actions = actions * self.hmax
            actions = actions // self.per_unit_qty * self.per_unit_qty

            # action 就是股票交易的份额，包含每一支股票对应买卖份额的数组。其中，正为买入，负为卖出，0 为持有
            argsort_actions = np.argsort(actions)
            # 获取卖出的清单
            # .shape[0] 取交易卖出的股票数量, 因为 actions 本身只有一维， 即表示持仓股票中每个股票的加减仓数量
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            # 获取买入的清单
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            # 这里交易一组股票
            for index in sell_index:
                # 对于卖出以实际账户的变动计算 reward
                sell_shares, sell_amouunt = self._sell_stock(index, actions[index])
                # 卖出为负, 所以乘以 -1
                actions[index] = sell_shares * -1
            for index in buy_index:
                actions[index], buy_amount = self._buy_stock(index, actions[index])

            # 交易的记录
            self.actions_memory.append(actions)

            # 更新 timetick & env data state
            # ==============================
            # state: s -> s+1
            self.day += 1
            # 更新环境的状态
            self.state = self._update_state()
            # 同上, 再计算一次期末的累计资产，因为, 进行了买卖交易
            end_total_asset = self._get_acct_asset()

            # 当前reward的定义: 使用资产增值的数额，可以处理多股票的组合任务
            # 这种 reward 定义的就是短期激励!!!
            self.reward = end_total_asset - begin_total_asset

            # 记录账户的累计资产记录
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            # 记录真实的账户盈亏记录
            self.rewards_memory.append(self.reward)

        # truncate = False
        return self.state, self.reward, self.terminal, False, self.acct_info


    # 每个 espisode 之后要重新收集资料。俗话说，”一个人的美酒可能是另一个人的毒药“
    def reset(self, *, seed=None, options=None):
        '''
        Desc:
            重置环境数据，从头开始一个新的 episode
        Args:
            *: 表示接受任意数量的可变参数
        '''
        super().reset(seed=seed, options=options)
        # initiate state
        # ===============
        self.day = 0
        # ==============================================================================
        # 可以在这个地方添加 self.data 的重定义逻辑, 结合 self.episode 索引，切换不同的股票股价序列
        # ==============================================================================
        # self.data 是 self.df 数据的动态切片
        # 记录股价与指标的序列信息
        self.data = self.df.iloc[self.day * self.per_batch_size : (self.day+1) * self.per_batch_size]
        # 因为有时候会设置 window_size 窗口
        # 记录股票当前最新的股价信息
        self.current_data = self.data.iloc[-self.stock_dim:]
        self.future_data = self.df.iloc[(self.day+1) * self.per_batch_size : (self.day+1) * self.per_batch_size + self.future_days]
        # 先重新初始化状态
        self.state = self._initiate_state()
        # ===================================

        # 先计算当前的账户累计金额
        # 更新 self.asset_memory
        # 需要区分是首次 reset 还是 episode 之后 reset
        # =================================================================
        if self.initial:
            self.acct_info = self._initial_acct_info()

        begin_total_asset = self._get_acct_asset()
        self.asset_memory = [begin_total_asset]
        # =================================================================

        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()] # _get_date 依赖 self.data, 上文已经更新, 因此仍然是从头开始

        # 一个 reset, 计算一次 episode
        self.episode += 1
        return self.state, self.acct_info # info


    def render(self, mode="human", close=False):
        return self.state


    def _initial_acct_info(self):
        '''
        Desc:
            初始化账户的信息
        Data example:
            acct_info = {
                'cash_asset': [1000, 1200, 900, 1200],
                'pfo_holding': {
                    '000001': [100, 200, -100, 400],
                },
                'pfo_price': {
                    '000001': [10, 60, 450, 200],
                },
                'profit_shares_sold': {
                    '000001': 100,
                },
                'pfo_shares_redeem': {
                    '000001': [{
                        'buy_date': '1900-01-01',
                        'shares': 0,
                        'hold': 0,
                        'yield': 0,
                        'soldout': 1,
                        'selling_date': '2500-01-01',
                    }]
                }
            }
        '''
        acct_info = {
            'cash_asset': [self.initial_amount],
            'pfo_holding': {}, # 持仓的变化流水
            'pfo_price': {},  # 买卖的价格流水
            'profit_shares_sold': {}, # 已卖出的盈利份额
            'pfo_shares_redeem': {},  # 记录持仓买入的时间，同时，卖出时更新对应的持仓变化；主要应用于基金统计
            }

        init_holding = [{
            'buy_date': '1900-01-01', 'shares': 0, 'hold': 0,
            'yield': 0, 'soldout': 1, 'selling_date': '2500-01-01'
            }]
        # 此处需要注意股票列表与持仓列表的mapping
        for idx, tic in enumerate(self.stock_pools):
            # 记录持仓量变化
            acct_info['pfo_holding'].setdefault(tic, [self.num_stock_shares[idx]])
            # 记录对应持仓的价格
            acct_info['pfo_price'].setdefault(tic, [0])
            acct_info['profit_shares_sold'].setdefault(tic, 0)
            acct_info['pfo_shares_redeem'].setdefault(tic, init_holding)
        return acct_info


    def _get_acct_asset(self):
        '''
        Desc:
            统计当前账户的资产价值, 包括持仓市值+现金金额
        '''
        cash_asset = sum(self.acct_info['cash_asset'])
        holding_asset = sum([
            sum(self.acct_info['pfo_holding'][stock_name]) * close_price
            for stock_name, close_price in zip(self.current_data.tic, self.current_data.close)
            ])
        total_asset = holding_asset + cash_asset
        return total_asset


    # 初始状态
    def _initiate_state(self):
        '''
        Dec:
            Env State 初始化
        '''
        # 记录股价与指标的序列信息
        self.data = self.df.iloc[0 : self.per_batch_size]
        # 记录股票当前最新的股价信息
        self.current_data = self.data.iloc[-self.stock_dim:]
        self.future_data = self.df.iloc[
            (self.day+1) * self.per_batch_size : (self.day+1) * self.per_batch_size + self.future_days]

        state = self._state_reshape()
        # logging.warning(f'env state view:\n{state}')
        # state shape lens = vars_num * window_size + 1
        # logging.warning(f'init env state shape ------------> {state.shape}')
        return state


    # 更新环境的状态，与 initial 状态不同
    def _update_state(self):
        '''
        Desc:
            更新 Env State
        '''
        self.data = self.df.iloc[self.day : (self.day + self.per_batch_size)]
        self.current_data = self.data.iloc[-self.stock_dim:]
        self.future_data = self.df.iloc[
            self.day * self.per_batch_size : self.day * self.per_batch_size + self.future_days]

        state = self._state_reshape()
        # logging.warning(f'update env state shape ------------> {state.shape}')
        return state


    def _state_reshape(self):
        '''
        Desc:
            将 State Reshape 成1维数组
        '''
        # 先去掉标签字段
        state_df = self.data.drop(columns=['date', 'tic']).copy()
        state = state_df.values
        # 此处因为是多个变量
        state = state.reshape(1, -1).tolist()[0]

        # 是否添加 持仓信息 & 账户现金 到 obs 中
        acct_cash_asset = sum(self.acct_info['cash_asset'])
        # holding_shares = [sum(shares) for _, shares in self.acct_info['pfo_holding'].items()]

        # state.extend(holding_shares)
        state.append(acct_cash_asset)
        # logging.warning(f'state cash asset ---------> {acct_cash_asset}')
        state = np.array(state, dtype='float32')
        return state


    # 获取交易的起始日期
    def _get_date(self):
        '''
        Desc:
            本函数收集一个日期列表, 作为 agent 交易日期索引
        '''
        # 多个股票的情况下
        if self.stock_dim > 1:
            # 从起始位置开始
            date = self.data.date.unique()[0]
        # 单只股票
        else:
            # 获取滚动日期
            date = self.data.date.iloc[0]

        # logging.warning(f'date ----------> {date}')
        return date


    def save_action_memory(self):
        pass


    # 随机种子
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    # 获取 SB3 环境
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
