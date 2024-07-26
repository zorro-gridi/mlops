# %%
import logging
from pprint import pprint
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
import pandas as pd


import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dir_list = [dname for dname in current_dir.as_posix().split('/')]
home_dir = '/'.join([dirname for dirname in dir_list[:dir_list.index('zorro')+1]])
env_path = '/'.join([dirname for dirname in dir_list[:dir_list.index('pycharm')+1]])
sys.path.append(env_path)

from mlops.ml_ops.rl.finrl.rule.v2 import FundTradeRules_V2
from mlops.ml_ops.rl.finrl.rule.v5 import FundTradeRules_V5
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
        Remark:
            1. 在初始化 env 的时候, 使用了_initial 类函数，例如 _initial_acct_info() 和 _initiate_state()
               函数中引用的一些变量，一定要提前定义
        Args of config:
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
            custom_rule_version: default None, 添加用户自定义的交易规则外挂的版本
            mode: 三种模式
                1. train: 策略训练模式
                2. infer: 策略推理测试模式
                3. live:  策略生产应用模式
        '''
        self.pfo_type = config.get('pfo_type', 'fund')                          # 资产的默认类型，fund / stock，需要读取样例数据
        self.default_data = pd.read_csv(current_dir / f'{self.pfo_type}.csv')
        self.acct_info = config.get('acct_info', None)                          # 是否用户自定义初始化账户信息
        self.day = config.get('day', 0)
        self.df = config.get('df', self.default_data)                           # 给df指定一个默认的数据源，使得可以实例化
        self.raw_data = config.get('raw_data', self.df)                         # 指数的原始涨跌数据，用于训练计算区间的收益率，和历史反转点

        # 2024-06-30 新增: 指数与基金代码的映射关系;
        # data demo: pd.DataFrame, columns: [user_id, plan_id, tic, fundcode, update_date]
        self.idx_2_fund = config.get('idx_2_fund', None)

        self.mode = config.get('mode', 'train') # 使用 bot 的模式
        # !!! important: 在生产模式下，self.df 只需要最新一条数据
        if self.mode in ['live']:
            self.df = self.raw_data.iloc[-1:]
            self.fund_data = config.get('fund_data', None)                      # 基金的原始涨跌数据，用于计算区间的收益率
        else:
            self.fund_data = config.get('fund_data', self.raw_data)

        # TODO: hmax 需要设计一个预期函数，即估计出最大定投次数, 换算得到
        self.hmax = config.get('hmax', 200)                                     # base model 的配置
        self.base_amount = 10000
        self.initial_amount = config.get('initial_amount', self.base_amount)    # get the initial cash

        # 单笔的最大买入金额，主要和 action 的分布有关:
        # 本 env action 分布范围~[-5, 5], hmax=100, 最大 500 元
        # 设置 per_buy_order_max_amt 的目的在于，当 action 的分布范围更广时，例如到10，则分布的最高可买入金额为 1000元
        # 若此时限制 per_buy_order_max_amt = 800， 则最高买入额为 800 元，意义在此 !
        self.per_buy_order_max_amt = config.get('per_buy_order_max_amt', self.initial_amount)

        self.per_unit_qty = config.get('per_unit_qty', 100)       # 每笔交易最小的股数，股票为100
        self.per_unit_amount = config.get('per_unit_amount', 200) # 每笔最小的交易额，基金一般为10元

        # 使用用户自定义的规则外挂的版本
        self.custom_rule_version = config.get('custom_rule_version', None)
        if self.custom_rule_version:
            self.custom_base_rule = eval(f'FundTradeRules_V{self.custom_rule_version}')(self)
            logging.warning(f'---------> 使用版本"{self.custom_rule_version}"的自定义交易规则外挂')

        # 当日的交易数据特征
        self.stock_pools = self.df.tic.unique()
        self.stock_dim = len(self.stock_pools)

        self.window_size = config.get('window_size', 1)
        self.future_days = config.get('future_days', 1) * self.stock_dim
        self.per_batch_size = self.stock_dim * self.window_size

        self.buy_cost_pct = config.get('buy_cost_pct', [0] * self.stock_dim)
        self.sell_cost_pct = config.get('sell_cost_pct', [0] * self.stock_dim)
        self.num_stock_shares = config.get('num_stock_shares', [0] * self.stock_dim) # 初始化的用户持仓数据
        self.reward_scaling = config.get('reward_scaling', None)
        self.tech_indicator_list = config.get('tech_indicator_list', None)

        # 2024-07-19 新增：标的物当日的涨跌幅; 不使用列表组合，未来需要组合时，可以单独使用并列的tradebot
        self.live_markup = config.get('live_markup', 0)

        # 是否完全重置环境与账户信息
        self.initial = config.get('initial', True)

        # initalize acct info & state
        self.acct_info = config.get('acct_info', None)
        self.user_id = config.get('user_id', 'zorro')
        self.plan_id = config.get('plan_id')
        self.acct_info = self.acct_info if self.acct_info else self._initial_acct_info()

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
        self.truncate = False
        self.print_verbosity = config.get('print_verbosity', None)
        self.model_name = config.get('model_name', None)
        self.iteration = config.get('iteration', None)

        self.reward = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        self.soldout = 0            # 清仓的次数
        self.goal_achieved = False  # 是否达到预期收益

        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self._seed()

        self.output_dir = config.get('output_dir', None)
        if self.output_dir:
            if not Path(self.output_dir).exists():
                Path(self.output_dir).mkdir(exist_ok=True)


    def _sell_stock(self, index, action):
        '''
        Desc:
            卖出 action. 这个函数交易的是一个股票
        Args:
            index 是一个索引，从 self.state 中取出对应的股票的持仓份额、或着股价
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
                if profit_shares_rest > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(abs(action), profit_shares_rest)

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
                        self.acct_info['cash_asset'][self._get_date()] = sell_amount
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

        # # holding_price: 平均持仓成本
        # stock_shares = sum(self.acct_info['pfo_holding'][stock_name])
        # pfo_asset = sum(np.array(self.acct_info['pfo_holding'][stock_name]) * np.array(self.acct_info['pfo_price'][stock_name]))
        # holding_price = round(pfo_asset / stock_shares, 3) if stock_shares > 0 else close_price # 因为初始仓位为 0

        def _do_buy():
            buy_num_shares = 0
            buy_amount = 0
            # check if the stock is able to buy
            if (
                # 判断当前这个股票是否可以交易（定义为技术指标，且为指标列表的第一个索引）
                1==1
                ):
                # 基于单笔最大交易限制的买入策略
                cash_asset = sum(self.acct_info['cash_asset'].values())
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
                            self.acct_info['cash_asset'][self._get_date()] = -close_price * buy_num_shares - buy_fee
                            # TODO: 买入股票，增加持仓; 此处都要改成字典模式
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


    def step(self, actions: np.array, mode='train'):
        '''
        Desc:
            在环境中执行一个动作。函数调用的 _sell_stock 和 _byu_stock 函数的 index 参数来源于对 actions 的排序
        Args:
            actions:, 交易的份额数组
        '''
        # logging.warning(f'-----------> 这是一条测试信息: step mode: "{mode}", actions: {actions}, terminal: {self.terminal}')
        begin_total_asset = self._get_acct_asset()

        if any([self.terminal, self.goal_achieved, self.truncate]):
            # logging.warning(f'is truncate env -------> {self.truncate}')
            # 交易后的累计资产
            end_total_asset = begin_total_asset

            # 以下全部为辅助信息:
            # ================
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
            # ==========================================================================
            return self.state, self.reward, self.terminal, True, self.acct_info
        else:
            # actions initially is scaled between 0 to 1
            # self.hmax 表示每一笔交易需要买入的最低股票数量
            # 不能买入分数的份额
            # actions = actions * self.hmax

            actions = actions * self.hmax
            actions = actions // self.per_unit_qty * self.per_unit_qty
            # print(f'------------> 最终确定的 action: {actions}')

            # *****************************************
            # 此处添加用户自定义的交易 actions 规则
            # *****************************************
            if self.custom_rule_version:
                actions = self.custom_base_rule.apply_trade_rules(actions)

            # action 就是股票交易的份额，包含每一支股票对应买卖份额的数组。其中，正为买入，负为卖出，0 为持有
            argsort_actions = np.argsort(actions)
            # !!! Important: 注意❌: buy_index 或 sell_index 一定要包含 0, 因为虽然策略的action为0, 但是还有人为的规则
            # 获取卖出的清单
            # .shape[0] 取交易卖出的股票数量, 因为 actions 本身只有一维， 即表示持仓股票中每个股票的加减仓数量
            sell_index = argsort_actions[:np.where(actions <= 0)[0].shape[0]]
            # 获取买入的清单
            buy_index = argsort_actions[::-1][:np.where(actions >= 0)[0].shape[0]]

            # 这里交易一组股票
            for index in sell_index:
                # 对于卖出以实际账户的变动计算 reward
                sell_shares, sell_amouunt = self._sell_stock(index, actions[index])
                # 卖出为负, 所以乘以 -1
                actions[index] = sell_shares * -1
            for index in buy_index:
                actions[index], buy_amount = self._buy_stock(index, actions[index])

            # =========================================================
            # 更新 timetick & env data state. Important: 注意放在最后的位置
            # =========================================================
            # state: s -> s+1
            self.day += 1
            # 是否 TimeLimit & truncate
            # 因为 self.df 的最后 self.future_days 行用来计算 cumulative reward，非训练数据需要剔除
            # 此处的 index 已经使用 factory 函数变成整数索引了
            if mode in ['train']:
                self.terminal = self.day == len(self.df.iloc[0:-self.future_days].index.unique())-self.per_batch_size+self.stock_dim
            elif mode in ['infer', 'live']:
                self.terminal = self.day == len(self.df.index.unique())-self.per_batch_size+self.stock_dim
            else:
                logging.warning(f'env step method "mode" params optional list: ["train", "infer", "live"]')
                raise

            if not self.terminal:
                # 更新环境的状态
                self.state = self._update_state()
                # self.day 必须要 +1 才能计算收益 reward
                # ********************************************
                # 同上, 再计算一次期末的累计资产，因为, 进行了买卖交易
                end_total_asset = self._get_acct_asset()
                # 当前 reward 的定义: 使用资产增值的数额，可以处理多股票的组合任务
                # 这种 reward 定义的就是短期激励!!!
                # 使用收益率作为reward的好处是在下跌的时候加仓可以平摊收益率, 提高reward, 鼓励加仓; 反之, 鼓励减仓
                # self.reward 是每一步交易的独立收益，所以计算累计收益时是: sum(self.reward)
                self.reward = round((end_total_asset - begin_total_asset) / self.initial_amount, 7)

                # 以下添加策略操作记录
                # ********************************************
                # 交易的记录
                self.actions_memory.append(actions)
                # 记录账户的累计资产记录
                self.asset_memory.append(end_total_asset)
                self.date_memory.append(self._get_date())
                # 记录真实的账户盈亏记录
                self.rewards_memory.append(self.reward)
                # logging.warning(f'step logging total acct asset --------> {self.asset_memory[-1]}')
                # 系统默认第4个返回的对象是 self.truncate
            return self.state, self.reward, self.terminal, self.truncate, self.acct_info


    # 每个 espisode 之后要重新收集资料。俗话说，”一个人的美酒可能是另一个人的毒药“, 是说有的行为对有些人来说好，对另些人却不好
    def reset(self, *, seed=None, options=None):
        '''
        Desc:
            重置环境数据，从头开始一个新的 episode
        Args:
            *: 表示接受任意数量的可变参数
            seed: 随机种子
        '''
        super().reset(seed=seed, options=options)
        self.day = 0
        # ======================================================================================
        # 可以在这个地方添加 self.data 的重定义逻辑, 结合 self.episode 索引，切换不同的股票股价序列：想法❌
        # “一个人的美酒🍷可能是另一个人的毒药💊”。 因此，不同股票，基金的走势不能混用
        # ======================================================================================
        # self.data 是 self.df 数据的动态切片
        # 记录股价与指标的序列信息
        self.data = self.df.iloc[self.day * self.per_batch_size : (self.day+1) * self.per_batch_size]
        # 因为有时候会设置 window_size 窗口
        # 记录股票当前最新的股价信息
        self.current_data = self.data.iloc[-self.stock_dim:]
        self.future_data = self.df.iloc[(self.day+1) * self.per_batch_size : (self.day+1) * self.per_batch_size + self.future_days]
        # ===================================

        # 先计算当前的账户累计金额
        # 更新 self.asset_memory
        # 需要区分是首次 reset 还是 episode 之后 reset
        # =================================================================
        if self.initial: # 这个 initial 表示每次 reset 都重新重置账户信息
            self.acct_info = self._initial_acct_info()

        begin_total_asset = self._get_acct_asset()
        self.asset_memory = [begin_total_asset]
        # =================================================================

        # initiate state
        # ===============
        # 再重新初始化状态。注意：初始化账户要在 state 之前，因为 state 中要添加 cahs_asset 信息
        self.state = self._initiate_state()

        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.truncate = False

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
                'plan_id': self.plan_id,
                'cash_asset': [1000, 1200, 900, 1200],  # 账户的现金流水
                'pfo_holding': {
                    '000001': [100, 200, -100, 400],    # pfo 的持仓流水
                },
                'pfo_price': {
                    '000001': [10, 60, 450, 200],       # 买入的价格
                },
                'profit_shares_sold': {
                    '000001': 100,                      # 卖出的份额流水
                },
                'pfo_shares_redeem': {
                    '000001': [{
                        'buy_date': '1900-01-01',       # 持仓买入日期
                        'buy_num_shares':               # 买入的金额
                        'shares': 0,                    # 扣除手续费的到账的金额（买入份额）, 对于卖部分仓位的时候, shares会被拆分
                        'hold': 0,                      # 持仓剩余的份额
                        'yield': 0,                     # 扣除卖出手续费的持仓净收益率
                        'soldout': 1,                   # 持仓是否被清仓
                        'selling_date': 'null',         # 持仓卖出日期, 默认 null, 方便入数据库
                        'hold_id': 20位数的str           # 单笔持仓id, 主键
                    }] # 记录单只基金的每一笔定投
                }
            }
        '''
        # 如果传入了初始化的 acct info，则不使用默认的初始化
        if self.acct_info is not None:
            return self.acct_info

        acct_info = {
            'plan_id': self.plan_id,
            'user_id': self.user_id,
            'order': [],
            'cash_asset': {'initial': self.initial_amount},
            'pfo_holding': {},          # 持仓的变化流水, 数据类型: Dict[List[Dict]]
            'pfo_price': {},            # 买卖的价格流水, 数据类型: Dict[List[Dict]]
            'profit_shares_sold': {},   # 已卖出的盈利份额, 数据类型: Dict[List[Dict]]
            'pfo_shares_redeem': {},    # 记录持仓买入的时间，同时，卖出时更新对应的持仓变化；主要应用于基金统计, 数据类型: Dict[List[Dict]]
            }
        # 此处需要注意股票列表与持仓列表的 mapping
        for idx, tic in enumerate(self.stock_pools):
            # 记录持仓量变化
            acct_info['pfo_holding'].setdefault(tic, [self.num_stock_shares[idx]])
            # 记录对应持仓的价格
            acct_info['pfo_price'].setdefault(tic, [0])
            acct_info['profit_shares_sold'].setdefault(tic, 0)
        return acct_info


    def _get_acct_asset(self):
        '''
        Desc:
            统计当前账户的资产价值, 包括持仓市值+现金金额
        '''
        cash_asset = sum(self.acct_info['cash_asset'].values())
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
        # 如果到达 dataframe 的底部, 终止模拟...
        if len(self.data) == 0:
            logging.warning(f'----------> Observation Reached Point, Termilated !!!')
            self.terminal = True
            return

        self.current_data = self.data.iloc[-self.stock_dim:]
        self.future_data = self.df.iloc[
            self.day * self.per_batch_size : self.day * self.per_batch_size + self.future_days]

        state = self._state_reshape()
        # logging.warning(f'update env state shape ------------> {state.shape}')
        return state


    def _get_pfo_ratio(self):
        '''
        Desc:
            计算账户的仓位
        '''
        return 0


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
        acct_cash_asset = round(sum(self.acct_info['cash_asset'].values()), 0)
        # TODO: 添加加仓空间为环境的一部分
        acct_pfo_ratio = round(self._get_pfo_ratio(), 1)
        # holding_shares = [sum(shares) for _, shares in self.acct_info['pfo_holding'].items()]
        # state.extend(holding_shares)
        # 将 acct_cash_asset 统一放缩到标准大小, 将账户金额的观察值放缩到以 base_amount 为单位
        state.append(round(acct_cash_asset / self.base_amount, 2))
        # TODO: 将仓位加入 state 存在弊端：当仓位已满时，环境不会自己重置???
        state.append(acct_pfo_ratio)

        # logging.warning(f'state cash asset ---------> {acct_cash_asset}')
        # Vectorized
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
            # logging.warning(f'{self.data.date.unique()[0]}')
            date = self.data.date.unique()[0]
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
