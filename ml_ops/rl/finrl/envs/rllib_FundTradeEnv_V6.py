# %%
# from gymnasium import spaces
import logging
from ray.rllib.env import EnvContext
import random

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dir_list = [dname for dname in current_dir.as_posix().split('/')]
home_dir = '/'.join([dirname for dirname in dir_list[:dir_list.index('zorro')+1]])
env_path = '/'.join([dirname for dirname in dir_list[:dir_list.index('pycharm')+1]])
sys.path.append(env_path)

from mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V1 import FundQuantTradeEnv_V1
from mlops.ml_ops.rl.finrl.envs.rllib_BaseTradeEnv import BaseTradeEnv


class FundQuantTradeEnv_V6(FundQuantTradeEnv_V1):
    '''
    基于版本 1 的 FundTrader Env 第6版
    '''
    def __init__(self, config: EnvContext):
        '''
        Update 更新如下:
            1. 策略网络自由买卖版本，可结合用户自定义买卖规则的外挂
        Conclusion:
            实验结论:
            1. 实验无效。指数跌幅太多, 导致策略直接选择不投入, 无法下注 !!!
        '''
        super().__init__(config)

    def _buy_stock(self, index, action):
        '''
        Desc:
            买入 action. 这个函数交易的是 1 个股票
        Args:
            index 是一个索引，从 self.state 中取出对应的股票的持仓份额 或者 股价
            action 是一个标量数值，表示针对制定 index 股票进行加减仓操作; 在 self.action_space 中定义
        '''
        stock_name = self.current_data['tic'].tolist()[index]

        def _do_buy():
            buy_num_shares = 0
            buy_amount = 0

            cash_asset = sum(self.acct_info['cash_asset'].values())
            buy_num_shares = min(cash_asset, self.per_buy_order_max_amt, action)

            if self.buying_signal(index):
                # 计算可买入的最多股票数量（基于单笔交易金额限制的）
                if buy_num_shares > self.per_unit_amount:
                    buy_amount = buy_num_shares * (1 - self.buy_cost_pct[index])
                    buy_fee = buy_num_shares * self.buy_cost_pct[index]

                    # 记录持仓的买入日期
                    self.acct_info['pfo_shares_redeem'].setdefault(stock_name, [])
                    # 推理 & 生产模式需要排除重复交易
                    if self.mode in ['infer', 'live'] and self._check_holding_duplicate(stock_name, trade_date='buy_date'):
                        return 0, 0

                    self.acct_info['pfo_shares_redeem'][stock_name].append({
                        'buy_date': self._get_date(),
                        'selling_date': '2500-01-01',
                        'shares': buy_amount,
                        'hold': buy_amount,
                        'yield': 0,
                        'soldout': 0,
                        'hold_id': str(random.randint(1e18, 9e18)),
                        'redeem_balance': buy_amount,
                        })
                    # 更新账户的可用本金
                    # 买入股票，现金账户减少金额
                    self.acct_info['cash_asset'][self._get_date()] = round(-buy_num_shares, 2)

                    # 更新买入的手续费
                    self.cost += buy_fee
                    # 更新交易频次，不能写在 step 函数中
                    self.trades += 1
                    # logging.warning(f"acct info ---> {self.acct_info['pfo_shares_redeem']}")

            # 返回买入的份额数量
            return buy_num_shares, buy_amount

        buy_num_shares, buy_amount = _do_buy()
        return buy_num_shares, buy_amount


    # %%
    def _sell_stock(self, index, action):
        '''
        Desc:
            基于策略建议, 卖出持仓
        '''
        action = abs(action)
        stock_name = self.current_data['tic'].to_list()[index]
        # 当前的剩余累计持仓
        # cash_asset = sum(self.acct_info['cash_asset'].values())
        stock_shares, _ = self._get_acct_pfo_shares()
        # logging.warning(f'当前账户持仓 ---------------> 现金: {cash_asset}, 份额: {stock_shares}')

        def _do_sell_normal():
            '''
            Desc:
                定义卖出交易的前提逻辑，例如：账户是否还有持仓？股票当前是否可以交易？
            '''
            sell_num_shares = 0 # 卖出份额，默认等于 sell_amount，输出后再转换，不影响
            sell_amount = 0

            if self.selling_signal(index):
                # 判断当前是否有该股票的持仓 & 股价是否大于 0
                if stock_shares > 0:
                    sell_amount = min(action, stock_shares)
                    return_ratio = self._caculate_selling_return(
                        stock_name, sell_amount, mode='LiveTrade')

            return sell_num_shares, sell_amount
        sell_num_shares, sell_amount = _do_sell_normal()
        return sell_num_shares, sell_amount


    def step(self, actions, **kwargs):
        actions = actions - 5
        return BaseTradeEnv.step(self, actions, **kwargs)