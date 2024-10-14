# %%
from gymnasium import spaces
import logging
from ray.rllib.env import EnvContext
import random
import time

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dir_list = [dname for dname in current_dir.as_posix().split('/')]
home_dir = '/'.join([dirname for dirname in dir_list[:dir_list.index('zorro')+1]])
env_path = '/'.join([dirname for dirname in dir_list[:dir_list.index('pycharm')+1]])
sys.path.append(env_path)

from mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv_V2 import FundQuantTradeEnv_V2



class FundQuantTradeEnv_V3(FundQuantTradeEnv_V2):
    '''
    基于版本二的 FundTrader Env 第三版
    '''
    def __init__(self, config: EnvContext):
        '''
        Update 更新如下:
            1. 继续更新 selling 策略
                1.1 当仓位策略提示减仓时，如果策略没有卖出行动，无论持仓是否盈利，应卖出部分持仓, 主动降低仓位。
                    主要解决问题：因为仓位控制原因，当仓位达到控制线之后，系统规定无法继续加仓，导致策略无法继续训练
            2. 更新买入策略
                2.1 当仓位策略线转换到提示加仓时, 主动补偿仓位的差额。TODO: 差额的确定方式还可以优化
        Remark:
            目前的仓位管理策略比较主观，需要建模决策
        Conclusion:
            1. 交易频率很高，手续费占比大
            2. 此策略依赖有效的仓位策略配合，待完善了量化仓位策略，再尝试...
        '''
        super().__init__(config)
        self.action_space = spaces.MultiDiscrete([6] * self.stock_dim)


    def _sell_stock(self, index, action):
        '''
        Desc:
            主动卖出超过仓位红线的多余仓位
        '''
        stock_name = self.current_data['tic'].to_list()[index]
        close_price = self.current_data['close'].to_list()[index]
        # 当前的剩余累计持仓
        # cash_asset = sum(self.acct_info['cash_asset'].values())
        stock_shares, _ = self._get_acct_pfo_shares()
        # logging.warning(f'当前账户持仓 ---------------> 现金: {cash_asset}, 份额: {stock_shares}')

        # 1. 当前可卖出的最大盈利持仓
        max_profit_shares = self._cal_max_selling_amount_with_min_yield(stock_name, min_yield=self.min_yield)
        # logging.warning(f'当前盈利持仓 ---------------> {max_profit_shares}')

        # check if the stock is able to sell, for simlicity we just add it in techical index
        # 也就是说，对应的股票是否可以交易，在技术指标中内置了。因为可能有些股票当日停牌，不可交易
        def _do_sell_normal():
            '''
            Desc:
                定义卖出交易的前提逻辑，例如：账户是否还有持仓？股票当前是否可以交易？
            '''
            sell_num_shares = 0 # 卖出份额，默认等于 sell_amount，输出后再转换，不影响
            sell_amount = 0

            last_day = max(self.day-1, 0)
            over_pfo_ratio = 0
            pfo_ratio_guide = self._set_pfo_ratio()
            pfo_ratio_act = self._get_pfo_ratio()

            if self.pfo_ratio_guide:
                last_pfo_ratio_guide = self.pfo_ratio_guide[last_day]
                # 方案一: 实际仓位超过仓位策略，即减仓
                over_pfo_ratio = round(pfo_ratio_act - pfo_ratio_guide, 3)
                # 方案二: 仓位策略线下移，才减仓
                # over_pfo_ratio = round(last_pfo_ratio_guide - pfo_ratio_guide, 3)

            # 判断卖出的条件: 刚好与买入相反
            if self.selling_signal(index):
                # 判断当前是否有该股票的持仓 & 股价是否大于 0
                if max_profit_shares > 0 and stock_shares > 0:
                    # logging.warning(f'action vs max_profit: {abs(action)}, {max_profit_shares:0.2f}')
                    sell_num_shares = max_profit_shares

                    if sell_num_shares > 0:
                        # 计算卖出可获得的金额，已经考虑交易费用
                        sell_amount = sell_num_shares

                # 如果策略不建议卖出, 则主动降低仓位
                # 在仓位策略转换的时候，如果当前的仓位策略提示减仓，主动降低仓位
                if sell_amount == 0 and over_pfo_ratio >= 0.05:
                    decrease_pfo_amount = round(self.initial_amount * over_pfo_ratio, 2)
                    sell_amount =  min(decrease_pfo_amount, stock_shares)

                    logging.warning(f'''
                        -------->
                        当前建议仓位: {pfo_ratio_guide}, 上次建议仓位: {last_pfo_ratio_guide}
                        当前实际仓位: {pfo_ratio_act}, 当前超建议仓位: {over_pfo_ratio}
                        ''')
                    logging.warning(f'当前仓位过高，主动减仓 ----> {over_pfo_ratio},  decrease_pfo_amount: {sell_amount}')

                if sell_amount > 0:
                    _, fee_rate = self._caculate_selling_return(stock_name, sell_amount, mode='LiveTrade')
                    # 生产模式不用更新持仓
                    self.acct_info['order'].append({
                        'order_id': str(random.randint(1e18, 9e18)),
                        'order_date': self._get_date(),
                        'order_type': 1,
                        'order_amount': sell_num_shares,
                        'fundcode': self._get_plan_idx_to_fundcode(stock_name, self._get_date()),
                        'fee_rate': fee_rate,
                        'order_fee': 'null',
                        'net_worth': 'null',
                        'received_amount': 'null',
                        'opt_type': 3,
                        'order_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'order_source': 'gridi',
                        })

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

        last_day = max(self.day-1, 0)
        plus_pfo_ratio = 0  # 补仓空间初始化
        plus_buy_amount = 0 # 最大可补的仓位初始化
        is_buying_accept = self.buying_signal(index)

        if self.pfo_ratio_guide:
            last_pfo_ratio_guide = self.pfo_ratio_guide[last_day]
            plus_pfo_ratio = last_pfo_ratio_guide - pfo_ratio_guideline
            plus_buy_amount = round(self.initial_amount * plus_pfo_ratio, 1)

        # 如果当前已到仓位指导线，则停止加仓
        if pfo_ratio > pfo_ratio_guideline:
            if self.verbose == 1:
                logging.warning(f'''
                    ------->
                    trade date: {self._get_date()}
                    指数牛熊位置: {self.current_data['closed_phase'].max()}, 百分位: {self.current_data['closed_phase_percentile'].max()}
                    当前仓位: {pfo_ratio}, 已达到仓位控制线 {pfo_ratio_guideline}, 暂停加仓 !!!
                    ''')
            return 0, 0

        stock_name = self.current_data.tic.to_list()[index]
        # 基金使用收盘涨跌幅，收盘价在基金的模拟环境中没有实际使用
        # close_price = self.current_data.close.to_list()[index]

        def _do_buy():
            buy_num_shares = 0
            buy_amount = 0

            # 判断买入的条件:
            if is_buying_accept:
                # 基于单笔最大交易限制的买入策略
                # logging.warning(f'acct cash list ----------> {self.acct_info["cash_asset"]}')
                cash_asset = sum(self.acct_info['cash_asset'].values())
                available_cash = min(cash_asset, self.per_buy_order_max_amt)
                available_shares = available_cash

                # 计算可买入的最多股票数量（基于单笔交易金额限制的）
                if available_shares > 0:
                    # 控仓技术: 加到策略的仓位线，最大可补的仓位
                    pfo_ratio_adj = pfo_ratio_guideline - pfo_ratio - action / self.initial_amount
                    # 仓位补偿
                    if pfo_ratio_adj > 0:
                        # 补偿动态的仓位百分位缺口
                        max_plus_amount = min(self.initial_amount * pfo_ratio_adj, plus_buy_amount)
                        buy_num_shares = min(available_shares, action + max_plus_amount)
                    # pfo_ratio_adj < 0, 仓位压缩，表示加仓会超仓位线，将仓位压缩到仓位线
                    else:
                        action_adj = self.initial_amount * (pfo_ratio_guideline - pfo_ratio)
                        buy_num_shares = min(available_shares, action_adj)

                    if buy_num_shares >= self.per_unit_amount:
                        buy_amount = buy_num_shares * (1 - self.buy_cost_pct[index])
                        buy_fee = buy_num_shares * self.buy_cost_pct[index]

                        # 记录持仓的买入信息
                        self.acct_info['pfo_shares_redeem'].setdefault(stock_name, [])
                        if self.mode in ['infer', 'live'] and self._check_holding_duplicate(stock_name, trade_date='buy_date'):
                            return 0, 0

                        # 排除当日的date key, 需要更新
                        self.acct_info['pfo_shares_redeem'][stock_name] = [
                            record for record in self.acct_info['pfo_shares_redeem'][stock_name]
                            if record['buy_date'] != self._get_date()
                            ]
                        if self.mode not in ['live']:
                            self.acct_info['pfo_shares_redeem'][stock_name].append({
                                'buy_date': self._get_date(),
                                'selling_date': 'null',
                                # 2024-06-29 修复，使用原始的买入金额
                                'shares': buy_num_shares,
                                # 买入的确认份额，待当日净值更新后再更新; 训练模式下使用金额
                                'hold': 'null' if self.mode == 'live' else buy_amount,
                                'received_amount': buy_amount,  # 入账的金额
                                # 2024-06-28 bug 修复: 增加手续费持仓额度
                                'redeem_balance': 'null' if self.mode == 'live' else buy_amount,
                                'buy_price': 'null' if self.mode == 'live' else 1,
                                'sell_price': 'null', # 买入的确认净值
                                'sold_shares': 0,     # 卖出的确认净值
                                # 此处的 yield 指持仓的涨跌幅, 不含买卖的费率
                                'yield': 0,
                                'soldout': 0,
                                # 2024-06-27 bug 修复: 增加持仓 id, 主键唯一
                                'hold_id': str(random.randint(1e18, 9e18)),
                                # 2024-07-12 添加；当卖出拆分holding时，需要新的主键
                                'record_id': str(random.randint(1e18, 9e18)),
                                'fundcode': self._get_plan_idx_to_fundcode(stock_name, self._get_date()),
                                'buy_rate': round(self.buy_cost_pct[index], 5),
                                'redeem_rate': 'null',
                                'etldate': time.strftime('%Y-%m-%d %H:%M:%S'),
                                })

                            # 更新账户的可用本金; 买入股票，现金账户减少金额
                            self.acct_info['cash_asset'][self._get_date()] = round(-buy_num_shares, 2)

                            # 更新买入的手续费
                            self.cost += buy_fee
                            # 更新交易频次，不能写在 step 函数中
                            self.trades += 1
                            # logging.warning(f"acct info ---> {self.acct_info['pfo_shares_redeem']}")

                        # 单笔交易的格式
                        self.acct_info['order'].append({
                            'order_id': str(random.randint(1e18, 9e18)),
                            'order_date': self._get_date(),
                            'order_type': 0,
                            'order_amount': buy_num_shares,
                            'fundcode': self._get_plan_idx_to_fundcode(stock_name, self._get_date()),
                            'fee_rate': round(self.buy_cost_pct[index], 5),
                            'order_fee': buy_num_shares * round(self.buy_cost_pct[index], 5),
                            'net_worth': 'null',
                            'received_amount': buy_amount,
                            'opt_type': 3,
                            'order_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'order_source': 'gridi',
                            })

            # 返回买入的份额数量
            return buy_num_shares, buy_amount

        buy_num_shares, buy_amount = _do_buy()
        return buy_num_shares * self.buy_times, buy_amount * self.buy_times

# %%
1+1
