from __future__ import annotations
import logging

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")

# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat


class StockTradeEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_amount: int = 0,
         # 初始持仓。 注意：持仓 & 交易手续费
         # 如果初始化多只股票，持仓的顺序要和 dataframe 中的股票名称顺序一致
        num_stock_shares: list[int] = [],
        buy_cost_pct: list[float] = [],
        sell_cost_pct: list[float] = [],
        hmax: int = 100, # action 对应的是买卖的单位数量，hamx 表示一单位买多少股
        reward_scaling: float = 0.8, # reward discount 系数
        state_dim: int = None,
        action_dim: int = None,
        tech_indicator_list: list[str] = None,
        make_plots: bool = False,
        print_verbosity=10,
        day=0, # 这个是索引的起始位置；配合 pandas index 属性和 factorize() 函数使用
        initial=True, # 是否是从头开始，完全重置环境 与 账户信息
        model_name="",
        mode="",
        iteration="",
        window_size=10,
        future_days=30,
        per_buy_order_max_amt=20000, # 单笔买入的最大金额
        per_unit_qty=100, # 单笔交易最小交易单位，例如股票100股
        per_unit_amount=10, # 单笔交易的最小金额，例如基金1～10元
    ):
        '''
        # stock_dim: 交易的股票数量
        # hmax: action 表示交易的手数，一手 = 100股。 hmax 可设置为 100
        # state_dim: 仅仅表示维度 int; 本质是输入的特征数量, 在本class中定义为 self.state 列表的长度
        # action_dim:  仅仅表示维度 int; 本质是股票持仓的数量，或者说交易股票池的数量
        # window_size: 输入序列的长度
        # future_days: 使用未来多少天的数据计算买入的预期收益率
        '''
        self.day = day
        self.df = df

        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list

        # 当日的交易数据特征
        self.stock_pools = self.df.tic.unique()
        self.stock_dim = len(self.stock_pools)
        self.window_size = window_size
        self.future_days = future_days
        self.per_batch_size = self.stock_dim * self.window_size

        # 是否是从头开始，完全重置环境 与 账户信息
        self.initial = initial

        # initalize state
        self.acct_info = self._initial_acct_info()
        # 先重新初始化状态
        self.state = self._initiate_state()

        # self.state_shape = (self.window_size, self.state.shape[1]) if state_dim is None else state_dim
        self.state_shape = (len(self.state),) if state_dim is None else state_dim
        # 定义 observation_space, 因为本 class 定义的环境只有1维
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.state_shape, dtype="float32"
        )

        self.action_dim = self.stock_dim if action_dim is None else action_dim
        # 定义 action_space
        # 为什么可以是 box，连续区间？因为在 step 函数中，action = (action * self.hmax).astype(int)
        # self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype="float32")
        # self.action_space = spaces.Discrete(21, start=-10)
        # MultiDiscrete: 多维离散空间
        self.action_space = spaces.MultiDiscrete(nvec=[11] * self.action_dim, start=[-5] * self.action_dim)

        # 是否作图
        self.make_plots = make_plots
        self.terminal = False
        self.print_verbosity = print_verbosity
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        # initialize reward
        self.reward = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0

        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )
        self.date_memory = [self._get_date()]
        self._seed()

        self.per_buy_order_max_amt = per_buy_order_max_amt
        self.per_unit_qty = per_unit_qty
        self.per_unit_amount = per_unit_amount


    # 卖出 action
    def _sell_stock(self, index, action):
        '''
        这个函数交易的是一个股票
        # index 是一个索引，从 self.state 中取出对应的股票的持仓份额 或着 股价
        # action 是一个标量数值，表示针对制定 index 股票进行加减仓操作；在 self.action_space 中定义
        '''
        stock_name = self.current_data.tic.to_list()[index]
        close_price = self.current_data.close.to_list()[index]
        stock_shares = sum(self.acct_info['pfo_holding'][stock_name])

        is_profit_price = np.array(self.acct_info['pfo_price'][stock_name]) <= close_price * 1.01 # 即最小盈利 1% 才允许卖出
        is_profit_shares = np.array(self.acct_info['pfo_holding'][stock_name]) > 0

        # 累计已盈利的持仓
        total_profit_shares = sum(
            np.extract(
            [a and b for a, b in zip(is_profit_price, is_profit_shares)],
            np.array(self.acct_info['pfo_holding'][stock_name])
            )
        )
        # 剩余可卖出的持仓
        profit_shares_rest = total_profit_shares - self.acct_info['profit_shares_sold'][stock_name]

        # check if the stock is able to sell, for simlicity we just add it in techical index
        # 也就是说，对应的股票是否可以交易，在技术指标中内置了。因为可能有些股票当日停牌，不可交易
        def _do_sell_normal():
            '''
            # 定义卖出交易的前提逻辑，例如：账户是否还有持仓？股票当前是否可以交易？
            '''
            sell_num_shares = 0
            sell_amount = 0

            if (
                # 需要判断股票自身是否可交易
                1==1
            ):
                # if self.state[index + 1] > 0: # if we use price < 0 to denote a stock is unable to trade in that day,
                # the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                # 判断当前是否有该股票的持仓 & 股价是否大于 0
                if profit_shares_rest > 0 and close_price > 0 and stock_shares > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(abs(action), stock_shares, profit_shares_rest)

                    if sell_num_shares % self.per_unit_qty == 0:
                        # 记录累计已卖出的盈利头寸
                        # logging.warning(f'do sell stock action: {sell_num_shares} quantities.')
                        self.acct_info['profit_shares_sold'][stock_name] += sell_num_shares
                        # 计算卖出可获得的金额，考虑交易费用
                        sell_amount = (
                            close_price
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # 卖出股票，仓位减少
                        self.acct_info['pfo_holding'][stock_name].append(-sell_num_shares)
                        self.acct_info['pfo_price'][stock_name].append(close_price)

                        # 卖出股票，现金账户增加金额
                        self.acct_info['cash_asset'].append(sell_amount)
                        # self.cost: 交易成本
                        self.cost += (
                            close_price
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1

            return sell_num_shares, sell_amount

        sell_num_shares, sell_amount = _do_sell_normal()
        return sell_num_shares, sell_amount


    def _buy_stock(self, index, action):
        '''
        这个函数交易的是一个股票
        # index 是一个索引，从 self.state 中取出对应的股票的持仓份额 或着 股价
        # action 是一个标量数值，表示针对制定 index 股票进行加减仓操作; 在 self.action_space 中定义
        '''
        stock_name = self.current_data.tic.to_list()[index]
        close_price = self.current_data.close.to_list()[index]
        pfo_asset = sum(np.array(self.acct_info['pfo_holding'][stock_name]) * np.array(self.acct_info['pfo_price'][stock_name]))
        stock_shares = sum(self.acct_info['pfo_holding'][stock_name])
        # 因为初始仓位为 0
        holding_price = round(pfo_asset / stock_shares, 3) if stock_shares > 0 else close_price

        def _do_buy():
            buy_num_shares = 0
            buy_amount = 0
            # check if the stock is able to buy
            if (
                # 判断当前这个股票是否可以交易（定义为技术指标，且为指标列表的第一个索引）
                1==1
            ):
                # 基于单笔最大交易限制
                available_cash = sum(self.acct_info['cash_asset'])
                available_cash = min(available_cash, self.per_buy_order_max_amt)

                # 计算可买入的最多股票数量（基于单笔交易金额限制的）
                # 因为买入的股数需要是 100 的倍数
                available_shares = available_cash // close_price // 100 * 100
                if available_shares > 0 and close_price > 0 and holding_price >= close_price * 0.6: # 平均亏损 40% 后不再补仓
                    # update balance
                    buy_num_shares = min(available_shares, action)
                    if buy_num_shares % self.per_unit_qty == 0:
                        buy_amount = (
                            close_price
                            * buy_num_shares
                            * (1 + self.buy_cost_pct[index])
                        )
                        # 更新账户的可用本金
                        # 买入股票，现金账户减少金额
                        self.acct_info['cash_asset'].append(-buy_amount)
                        # 买入股票，增加持仓
                        self.acct_info['pfo_holding'][stock_name].append(buy_num_shares)
                        self.acct_info['pfo_price'][stock_name].append(close_price)

                        # 更新买入的手续费
                        self.cost += (close_price * buy_num_shares * self.buy_cost_pct[index])
                        # 更新交易频次，不能写在 step 函数中
                        self.trades += 1

            # 返回买入的份额数量
            return buy_num_shares, buy_amount
        buy_num_shares, buy_amount = _do_buy()
        return buy_num_shares, buy_amount


    # 画出账户资产（现金+市值）变化的趋势图
    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()


    def step(self, actions):
        # 是否 TimeLimit & truncate
        # 因为 self.df 的最后 30 行用来计算 cumulative reward，非训练数据需要剔除
        self.terminal = self.day > len(self.df.iloc[:-self.future_days].index.unique()) - self.per_batch_size

        cash_asset = sum(self.acct_info['cash_asset'])
        holding_asset = sum([
            sum(self.acct_info['pfo_holding'][stock_name]) * close_price
            for stock_name, close_price in zip(self.current_data.tic, self.current_data.close)
            ])
        begin_total_asset = holding_asset + cash_asset

        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()

            # 交易后的累计资产
            end_total_asset = begin_total_asset

            # 以下全部为辅助信息：
            # ==========================================================================
            df_total_value = pd.DataFrame(self.asset_memory)
             # initial_amount is only cash part of our initial asset
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

            # reward 流水 dataframe
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]

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

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                plt.close()
            # ==========================================================================
            return self.state, self.reward, self.terminal, False, self.acct_info

        else:
            # actions initially is scaled between 0 to 1
            # self.hmax 表示每一笔交易需要买入的最低股票数量
            # 这个可以自定义改一下
            # convert into integer because we can't by fraction of shares
            # 不能买入分数的份额
            actions = actions * self.hmax
            actions = actions // 100 * 100
            # print("begin_total_asset:{}".format(begin_total_asset))

            # action 就是股票交易的份额，包含每一支股票对应买卖份额的数组。其中，正为买入，负为卖出，0 为持有
            argsort_actions = np.argsort(actions)
            # 获取卖出的清单
            # .shape[0] 取交易卖出的股票数量, 因为 actions 本身只有一维， 即表示持仓股票中每个股票的加减仓数量
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            # 获取买入的清单
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            potential_reward = 0
            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                # 对于卖出以实际账户的变动计算 reward
                sell_shares, sell_amouunt = self._sell_stock(index, actions[index])
                # 卖出为负, 所以乘以 -1
                actions[index] = sell_shares * -1
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")
                # 为什么是减？因为，如果卖出后，未来股价继续涨，则不应该卖出股票，所以负向激励。反之，则为逃顶，为正向激励
                potential_reward -= self._reward_strategy(index) * sell_amouunt

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index], buy_amount = self._buy_stock(index, actions[index])
                # 对于买入，以未来的 discount cumulative reward 预期收益率计算 reward
                potential_reward += self._reward_strategy(index) * buy_amount

            # 交易的记录
            self.actions_memory.append(actions)

            # 更新 timetick & env data state
            # ==============================
            # state: s -> s+1
            self.day += 1
            # 更新环境的状态
            self.state = self._update_state()

            # 同上, 再计算一次期末的累计资产，因为进行了买卖交易
            cash_asset = sum(self.acct_info['cash_asset'])
            holding_asset = sum([
                sum(self.acct_info['pfo_holding'][stock_name]) * close_price
                for stock_name, close_price in zip(self.current_data.tic, self.current_data.close)
                ])
            end_total_asset = holding_asset + cash_asset
            # 当前reward的定义: 使用资产增值的数额，可以处理多股票的组合任务
            # 这种 reward 定义的就是短期激励!!!
            self.reward = end_total_asset - begin_total_asset

            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            # 记录真实的账户盈亏记录
            self.rewards_memory.append(self.reward)
            # add current state in state_recorder for each step
            self.state_memory.append(self.state)

            # 使用自定义的 reward 来训练 agent
            self.reward = potential_reward
            # self.reward = self.reward + potential_reward

        # truncate = False
        return self.state, self.reward, self.terminal, False, self.acct_info


    # 每个 espisode 之后要重新收集资料，”一个人的美酒可能是另一个人的毒药“
    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        '''
        # * 表示接受任意数量的可变参数
        '''
        # initiate state
        # ===============
        self.day = 0
        # ==============================================================================
        # 可以在这个地方添加 self.data 的重定义逻辑, 结合 self.episode 索引，切换不同的股票股价序列
        # ==============================================================================
        # self.data 是 self.df 数据的动态切片
        # 记录股价与指标的序列信息
        self.data = self.df.iloc[self.day * self.per_batch_size : (self.day+1) * self.per_batch_size]
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
            cash_asset = sum(self.acct_info['cash_asset'])
            holding_asset = sum([
                sum(self.acct_info['pfo_holding'][stock_name]) * close_price
                for stock_name, close_price in zip(self.current_data.tic, self.current_data.close)
                ])
            begin_total_asset = holding_asset + cash_asset
            self.asset_memory = [begin_total_asset]
        else:
            # 不需要初始化
            cash_asset = sum(self.acct_info['cash_asset'])
            holding_asset = sum([
                sum(self.acct_info['pfo_holding'][stock_name]) * close_price
                for stock_name, close_price in zip(self.current_data.tic, self.current_data.close)
                ])
            begin_total_asset = holding_asset + cash_asset
            self.asset_memory = [begin_total_asset]
        # =================================================================

        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        # 一个 reset, 计算一次 episode
        self.episode += 1
        return self.state, self.acct_info # info


    def render(self, mode="human", close=False):
        return self.state


    def _initial_acct_info(self):
        acct_info = {
            'cash_asset': [self.initial_amount],
            'pfo_holding': {},
            'pfo_price': {},
            'profit_shares_sold': {},
            }
        # 此处需要注意股票列表与持仓列表的mapping
        for idx, tic in enumerate(self.stock_pools):
            # 记录持仓量变化
            acct_info['pfo_holding'].setdefault(tic, [self.num_stock_shares[idx]])
            # 记录对应持仓的价格
            acct_info['pfo_price'].setdefault(tic, [0])
            acct_info['profit_shares_sold'].setdefault(tic, 0)
        return acct_info


    # 奖励函数设计
    def _reward_strategy(self, index):
        close_price = self.current_data.close.to_list()[index]
        stock_name = self.stock_pools[index]

        future_price = self.future_data.loc[self.future_data['tic'] == stock_name].close.to_list()
        # 在列表的首位添加输入序列的收盘价
        future_price.insert(0, close_price)
        price_change = np.diff(future_price)
        # 计算股价每日涨跌幅
        pct_change = price_change / np.array(future_price)[:-1]

        # 按照未来股价走势的预期收益作为奖励分数
        reward_return = sum([
            pct * (1 - np.power(self.reward_scaling, 1 / (i+1)))
            for i, pct in enumerate(pct_change)
            ])
        return reward_return


    # 初始状态
    def _initiate_state(self):
        # 是否完全初始化
        if self.initial:
            self.data = self.df.iloc[0 : self.per_batch_size]
            self.current_data = self.data.iloc[-self.stock_dim:]
            self.future_data = self.df.iloc[
                (self.day+1) * self.per_batch_size : (self.day+1) * self.per_batch_size + self.future_days]
            state = self._state_reshape()
        else:
            state = self._update_state()
        return state


    # 更新环境的状态，与 initial 状态不同
    def _update_state(self):
        self.data = self.df.iloc[self.day : (self.day + self.per_batch_size)]
        self.current_data = self.data.iloc[-self.stock_dim:]
        self.future_data = self.df.iloc[
            self.day * self.per_batch_size : self.day * self.per_batch_size + self.future_days]

        state = self._state_reshape()
        return state


    # 为环境增加账户可用现金 + 持仓数据
    def _state_reshape(self):
        state = self.data.drop(columns=['date', 'tic']).values
        state = state.reshape(1, -1).tolist()[0]

        acct_cash_asset = sum(self.acct_info['cash_asset'])
        holding_shares = [sum(shares) for _, shares in self.acct_info['pfo_holding'].items()]

        state.extend(holding_shares)
        state.append(acct_cash_asset)
        state = np.array(state)
        return state


    # 获取交易的起始日期
    def _get_date(self):
        '''
        本函数收集一个日期列表, 作为agent交易日期索引
        '''
        # 多个股票的情况下
        if self.stock_dim > 1:
            # 从起始位置开始
            date = self.data.date.unique()[0]
        # 单只股票
        else:
            # 获取滚动日期
            date = self.data.date
        return date


    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if self.stock_dim > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list, 'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # print(df_states)
        return df_states


    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value


    def save_action_memory(self):
        if self.stock_dim > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions


    # 随机种子
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    # 获取 SB3 环境
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs