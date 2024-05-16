# %%
# from gymnasium import spaces
import logging
from ray.rllib.env import EnvContext


import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dir_list = [dname for dname in current_dir.as_posix().split('/')]
home_dir = '/'.join([dirname for dirname in dir_list[:dir_list.index('zorro')+1]])
env_path = '/'.join([dirname for dirname in dir_list[:dir_list.index('pycharm')+1]])
sys.path.append(env_path)

from mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V1 import FundQuantTradeEnv_V1



class FundQuantTradeEnv_V4(FundQuantTradeEnv_V1):
    '''
    基于版本一的 FundTrader Env 第四版
    '''
    def __init__(self, config: EnvContext):
        '''
        Update 更新如下:
            1. 返璞归真版:
                1.1 买入时，根据策略空间优化决策（但是, 依然有最大单笔金额限制）；
                1.2 卖出时，当前若无盈利持仓时，再根据策略空间决策
        Conclusion:
            1. 交易频率中等，收益率温和
            2. 貌似适合测试单边下跌的指数，不适合作为执行策略，策略收益比较不稳定。
        '''
        super().__init__(config)


    def _sell_stock(self, index, action):
        '''
        Desc:
            基于策略建议, 卖出持仓
        '''
        action = abs(action)
        stock_name = self.current_data['tic'].to_list()[index]
        # 当前的剩余累计持仓
        # cash_asset = sum(self.acct_info['cash_asset'])
        stock_shares, _ = self._get_acct_pfo_shares()
        # logging.warning(f'当前账户持仓 ---------------> 现金: {cash_asset}, 份额: {stock_shares}')

        # 1. 当前可卖出的最大盈利持仓
        max_profit_shares = self._get_max_yield_shares(stock_name, min_yield=self.min_yield)

        def _do_sell_normal():
            '''
            Desc:
                定义卖出交易的前提逻辑，例如：账户是否还有持仓？股票当前是否可以交易？
            '''
            sell_num_shares = 0 # 卖出份额，默认等于 sell_amount，输出后再转换，不影响
            sell_amount = 0

            # 判断卖出的条件: 刚好与买入相反
            if self.selling_signal(index):
                # 判断当前是否有该股票的持仓 & 股价是否大于 0
                if stock_shares > 0:
                    # 判断是否有盈利的持仓
                    if max_profit_shares > 0:
                        # 先取策略与盈利之大
                        sell_amount = max(action, max_profit_shares)
                        # 再大中取小
                        sell_amount = min(sell_amount, stock_shares)
                    else:
                        sell_amount = min(action, stock_shares)
                    return_ratio = self._caculate_selling_return(stock_name, sell_amount, mode='LiveTrade')

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
        # 更新仓位控制线
        pfo_ratio_guideline = self._set_pfo_ratio()
        pfo_ratio = self._get_pfo_ratio()
        is_buying_accept = self.buying_signal(index)

        # 控制仓位
        if pfo_ratio > pfo_ratio_guideline:
            if self.verbose == 1:
                logging.warning(f'-------> 当前仓位: {pfo_ratio}, 已达到仓位控制线 {pfo_ratio_guideline}, 暂停加仓 !!!')
            return 0, 0

        stock_name = self.current_data.tic.to_list()[index]

        def _do_buy():
            buy_num_shares = 0
            buy_amount = 0

            # 判断买入的条件:
            if is_buying_accept:
                # 基于单笔最大交易限制的买入策略
                # logging.warning(f'acct cash list ----------> {self.acct_info["cash_asset"]}')
                cash_asset = sum(self.acct_info['cash_asset'])
                available_cash = min(cash_asset, self.per_buy_order_max_amt)
                # 注意：与股票不同，基金直接使用买卖金额，模型输出金额后再换算份额
                available_shares = available_cash

                # 计算可买入的最多股票数量（基于单笔交易金额限制的）
                if available_shares > 0:
                    buy_num_shares = min(available_shares, action)
                    if buy_num_shares > self.per_unit_amount:
                        buy_amount = buy_num_shares * (1 - self.buy_cost_pct[index])
                        buy_fee = buy_num_shares * self.buy_cost_pct[index]

                        # 更新账户的可用本金
                        # 买入股票，现金账户减少金额
                        self.acct_info['cash_asset'].append(round(-buy_num_shares, 2))
                        # 记录持仓的买入日期
                        self.acct_info['pfo_shares_redeem'].setdefault(stock_name, [])

                        if self.mode in ['infer', 'live'] and self._check_holding_duplicate(stock_name, trade_date='buy_date'):
                            return 0, 0
                        self.acct_info['pfo_shares_redeem'][stock_name].append({
                            'buy_date': self._get_date(),
                            'selling_date': '2500-01-01',
                            'shares': buy_amount,
                            'hold': buy_amount,
                            'yield': 0,
                            'soldout': 0,
                            })

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