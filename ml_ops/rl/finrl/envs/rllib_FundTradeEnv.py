from __future__ import annotations
import logging
from pathlib import Path

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from ray.rllib.env import EnvContext
from datetime import datetime
from copy import copy


import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dir_list = [dname for dname in current_dir.as_posix().split('/')]
home_dir = '/'.join([dirname for dirname in dir_list[:dir_list.index('zorro')+1]])
env_path = '/'.join([dirname for dirname in dir_list[:dir_list.index('pycharm')+1]])
sys.path.append(env_path)

from mlops.ml_ops.rl.finrl.envs.rllib_BaseTradeEnv import BaseTradeEnv


matplotlib.use("Agg")


"""
@Author: Zorro
@Date: 2024-01-01
@Desc:
    本代码定义了基于 rllib 强化学习训练框架的 gym 环境模型
@Release log:
    1. 尝试了将“加仓空间=仓位红线-当前仓位”加入环境观察，并且主动在 action 中补一些仓位，重新训练后，发现效果并不好
"""


class FundQuantTradeEnv(BaseTradeEnv):
    """
    Desc:
        A fund trading environment for OpenAI gym
        忽略 _buy_stock 这个名称，实际是 _buy_fund
    """
    metadata = {"render_modes": ["human"]}


    def __init__(self, config: EnvContext):
        '''
        Desc:
            1. 添加、并修改父类的属性: 写在 super() 继承的后面
            2. 增加属性依赖: 写在 super() 继承的前面
        '''
        super().__init__(config)

        # 增加或修改的属性可以写在 super() 后面
        self.goal_yield = config.get('goal_yield', 1.0) # 训练的时候要保障数据训练完，推理的时候需要定义
        self.phase_yield = config.get('phase_yield', 0.02)
        # 基金的净值很小，而且可以买很少的份数，可以使用连续空间; 缺点：训练太慢
        # self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,), dtype="float32")
        # 最终方案 -> action_space: MultiDiscrete, 多维离散空间
        self.action_space = spaces.MultiDiscrete([11] * self.stock_dim)
        self.temperature = config.get('temperature', 0.5) # 市场交易热度, 影响买卖的频率。其中，0.5-中性; > 0.5-贪婪; < 0.5-谨慎
        # self.complete_times = 3 # 仓位不足时，补仓的倒数比例，这个比例应该和牛熊点位的仓位比例相匹配
        # 例如，市场由牛转猴、熊，仓位比例需要从80%下降到60%，如果初始仓位20%，则 20% + 80% / 3 < 60%，那么这个 3 的比例就是合理的
        # 这个数的具体值还需要调整, 都是主观设置，取消！！！


    def _update_acct_holdings_debit_yield(self):
        ''''
        Desc:
            计算当前持仓的对应的扣除卖出手续费之后的实际收益率.
        '''
        # logging.warning(f'acct info -------> {self.acct_info["pfo_shares_redeem"]}')
        # 更新持仓的最新市值
        cur_date = self._get_date()
        update_holdings = self.acct_info['pfo_shares_redeem']

        if update_holdings:
            for fund_code, holdings in update_holdings.items():
                for s in holdings:
                    buy_date = s['buy_date']

                    if s['soldout'] == 0:
                        days_diff = self._calculate_date_diff(buy_date, cur_date)
                        # logging.warning(f'buy date: --------> {buy_date}, current date --------> {cur_date}')
                        s['yield'] = round(
                            self._caculate_holding_yield(fund_code, buy_date, cur_date)
                            - self._get_redeem_rate(days_diff)
                            ,4)

            self.acct_info['pfo_shares_redeem'] = update_holdings
            return update_holdings
        else:
            return {}


    def _set_pfo_ratio(self):
        '''
        Desc: Importan!
            账户的仓位控制策略。基于上证指数的相对牛熊点位判断
        '''
        ratio_strategy = {
            0: 0.8,
            1: 0.5,
            2: 0.3,
            }
        sz_point_phase = self.current_data['sz_closed_phase'].unique()[0]
        idx_pint_phase = self.current_data['closed_phase'].unique()[0]
        pfo_ratio_guideline = (ratio_strategy[sz_point_phase] + ratio_strategy[idx_pint_phase]) / 2
        return pfo_ratio_guideline


    def _get_acct_pfo_shares(self):
        '''
        Desc:
            调用 _update_acct_holdings_debit_yield 方法，计算账户资产的最新市值
        '''
        update_holdings = self._update_acct_holdings_debit_yield()
        if update_holdings:
            # logging.warning(f'update_holdings ------> {update_holdings}')
            # pfo_shares: 实际申购的市值
            pfo_shares = sum([
                s['hold'] for holding in update_holdings.values() for s in holding])
            # pfo_asset: 实际卖出的市值
            pfo_asset = sum([
                s['hold'] * (1 + s['yield']) for holding in update_holdings.values() for s in holding])
            return pfo_shares, pfo_asset
        else:
            return 0, 0


    def _get_acct_asset(self):
        '''
        Desc:
            调用 _get_acct_pfo_shares 方法，统计当前账户的资产价值, 包括持仓扣除若卖出手续费的市值+现金金额
        '''

        # if len(self.acct_info['cash_asset']) > 1:
        #     logging.warning(f"acct hold date ---------------> {self.acct_info['pfo_shares_redeem']}")
        #     logging.warning(f"acct cash list ---------------> {self.acct_info['cash_asset']}")

        cash_asset = sum(self.acct_info['cash_asset'])
        # 统计持仓的当前市值
        _, pfo_asset = self._get_acct_pfo_shares()
        total_asset = pfo_asset + cash_asset
        # logging.warning(f'current acct ----> pfo_asset: {pfo_asset}, cash_asset: {cash_asset}')
        return total_asset


    def step(self, actions):
        '''
        Desc:
            继承父类的 step 方法
        '''
        # MultiDiscrete start 参数在实际运行中不起作用，需要手动调节 actions
        actions = actions - 5
        # logging.warning(f'actions ---------------> {actions}')

        # 达到目标收益清仓
        # TODO: 写一个触发清仓的条件
        pfo_yield = self._get_pfo_soldout_yield()
        cumsum_yield = self._get_acct_cumsum_yield()

        # 达到阶段清仓条件: 1. 持仓达到目标；2. 账户达到目标
        if pfo_yield >= self.phase_yield or cumsum_yield >= self.goal_yield:
            # 执行清仓操作: 先执行动作，再变更状态
            self.acct_pfo_soldout()

            # 达到整体收益率目标，发出停止交易的信号: self.goal_achieved = 1
            # 注意，这个信号只能在交易的时候使用
            if cumsum_yield >= self.goal_yield:
                logging.warning(f'当前账户【清仓累计收益率】: {cumsum_yield:0.4f}, 达到【预期收益率】目标: {self.goal_yield}, 账户清仓 !!!')
                self.goal_achieved = True
            else:
                logging.warning(f'当前账户【持仓清仓收益率】: {pfo_yield:0.4f}, 达到【阶段收益率】目标: {self.phase_yield}, 账户清仓 !!!')

        return super().step(actions)


    def _calculate_date_diff(self, start_date, end_date):
        ''''
        Desc:
            统计日期的天数间隔
        '''
        date_format = '%Y-%m-%d'

        # 将日期字符串解析为 datetime 对象
        start_datetime = datetime.strptime(start_date, date_format)
        end_datetime = datetime.strptime(end_date, date_format)

        # 计算日期差值
        date_difference = (end_datetime - start_datetime).days
        return date_difference


    def _get_redeem_rate(self, days):
        '''
        Desc:
            返回赎回基金的手续费率
        '''
        days_range_max = 730
        if days >= days_range_max:
            logging.warning(f'-------> days: {days}, limit: {days_range_max}, days redeem rate mapping not set!!!')
            raise

        if days < 7:
            return 1.5 / 100
        elif days >= 7 and days < 30:
            return 0.75 / 100
        elif days >= 30 and days <= 365:
            return 0.5 / 100
        elif days >= 365 and days < 730:
            return 0.3 / 100


    def _get_max_yield_shares(self, fund_code, min_yield):
        '''
        Desc:
            计算给定卖出日期，持仓账户中累计最小盈利的持仓数量
        Args:
            fund_code: 基金名称, 指数名称等
            min_yield: 最小盈利阈值
        Return:
            max_yield_shares: 当前可卖出的已盈利的最大持仓数量
        '''
        update_holdings = self._update_acct_holdings_debit_yield()
        max_yield_shares = 0

        if not update_holdings:
            return 0
        holdings = update_holdings[fund_code]
        holdings = [
            s for s in holdings
            if s['soldout'] == 0 and s['yield'] >= min_yield
            ]

        if holdings:
            # logging.warning(f'test acct holding yield -------->\n{pd.DataFrame(holdings)}')
            max_yield_shares = sum([s['hold'] for s in holdings])
            # logging.warning(f'当前可卖的累计盈利的持仓 ------------> 份额: {shares}, yield: {selling_return}')
        return max_yield_shares


    def _caculate_holding_yield(self, fund_code, buy_date, sell_date):
        '''
        Desc:
            基金基金卖出时的毛收益率
        Args:
            fund_code: 基金名称, 指数名称等
            buy_date: 买入日期
            sell_date: 卖出日期
        Return:
            fund_yield: 卖出的收益率%
        '''
        fund_cond = (self.df.tic == fund_code) & (self.df.date > buy_date) & (self.df.date <= sell_date)
        fund_data = self.df.loc[fund_cond].copy()

        if len(fund_data) > 0:
            fund_yield = fund_data['close'].sum() / 100
            return fund_yield
        return 0


    def _caculate_soldout_cost_fee(self):
        '''
        Desc:
            计算清仓时的卖出手续费
        '''
        acct_holdings = self._update_acct_holdings_debit_yield()
        soldout_date = self._get_date()

        soldout_fee = sum([
            s['hold'] * self._get_redeem_rate(self._calculate_date_diff(s['buy_date'], soldout_date))
            for holdings in acct_holdings.values() for s in holdings
            ])
        return soldout_fee


    def _get_pfo_soldout_yield(self):
        '''
        Desc:
            计算账户持仓【清仓时】扣除手续费的卖出收益率 = 所有持仓市值 / 所有持仓的成本 - 1
        Returns:
            pfo_yield: 卖出的持仓收益率
        '''
        pfo_shares, pfo_asset = self._get_acct_pfo_shares()
        pfo_yield = pfo_asset / pfo_shares - 1 if pfo_shares > 0 else 0
        return pfo_yield


    def _get_acct_cumsum_yield(self):
        '''
        Desc:
            计算当前账户【清仓后】扣除卖出手续费的累计收益率 = 期末资产 / 期初资产 - 1
        Returns:
            账户累计收益率
        '''
        cumsum_yield = self._get_acct_asset() / self.initial_amount - 1
        if cumsum_yield >= self.goal_yield:
            self.goal_achieved = True
        return cumsum_yield


    def _get_pfo_ratio(self):
        '''
        Desc:
            计算账户的仓位
        '''
        # TODO: pfo_shares 扣除了买入的手续费，不准确
        pfo_shares, pfo_asset = self._get_acct_pfo_shares()
        pfo_ratio = round(pfo_shares / self.initial_amount, 3)
        return pfo_ratio


    def acct_pfo_soldout(self):
        '''
        Desc:
            执行账户的清仓操作
        '''
        acct_holdings = self._update_acct_holdings_debit_yield()
        soldout_date = self._get_date()

        soldout_fee = self._caculate_soldout_cost_fee()
        _, selling_return = self._get_acct_pfo_shares()
        self.cost += soldout_fee
        self.soldout += 1
        self.trades += 1

        # 更新持仓
        for _, holdings in acct_holdings.items():
            for s in holdings:
                if s['soldout'] == 0:
                    s['soldout'] = 1
                    s['selling_date'] = soldout_date
                    s['hold'] = 0

        self.acct_info['pfo_shares_redeem'] = acct_holdings
        # 卖出股票，现金账户增加金额
        self.acct_info['cash_asset'].append(round(selling_return, 2))


    def _caculate_selling_return(self, fund_code, sell_amount, mode='LiveTrade'):
        '''
        Desc:
            考虑到手续费率按持仓时间的不同, 该函数完成两个功能：
            1. 按照持仓先买先出的原则，更新账户的买入持仓在卖出后的状态
            2. 计算卖出的手续费，并更新累计交易成本 self.cost
            ps: 本函数在卖出持仓的时候调用
        Args:
            fund_code: 基金代码、指数代码，名称等
            sell_amount: 需要卖出的原始持仓金额, 该函数将自动转换计算卖出shares的到账金额
            mode: option, 交易模式
                "LiveTrade": 真实交易模式, 该模式会实时更新账户的信息。关于实际交易数据的更新，只能在 LiveTrade 模式下进行
                "BackTest": 数据回测模式, 该模式仅测试策略的收益，并不更新实际的账户数据。例如，测试部分卖出与清仓时的平均收益率
        Return:
            return_ratio: 返回扣除卖出手续费之后的实际收益率
        '''
        update_holdings = self._update_acct_holdings_debit_yield()
        acct_holdings = copy(update_holdings[fund_code])

        sold_holdings = [
            s for s in acct_holdings if s['soldout'] == 1]
        still_holdings = [
            s for s in acct_holdings if s['soldout'] == 0]

        if still_holdings and sell_amount > 0:
            # logging.warning(f'pfo_shares_redeem ----------> {still_holdings}')
            # 计算扣除卖出手续费的收益率就是为了按照收益率真实排序
            sorted_holdings = list(sorted(still_holdings, key=lambda x: x['yield'], reverse=True))
            # logging.warning(f'sorted_holdings ------->\n{pd.DataFrame(sorted_holdings)}')

            update_items_list = []                          # 保存拆分的卖出
            selling_cost = 0                                # 统计卖出手续费成本
            selling_shares = copy(sell_amount)              # 统计卖出的原始持仓金额
            selling_value = 0                               # 统计卖出时机到账的金额

            selling_date = self._get_date()
            # 遍历持仓记录，根据卖出金额逐渐减少持有份额
            for i, shares_info in enumerate(sorted_holdings):
                # 在循环体中 sell_amount 越减越少
                if round(sell_amount, 2) == 0:
                    break

                date = shares_info['buy_date']
                shares = shares_info['hold']
                # 获取持仓的天数
                holding_days = self._calculate_date_diff(date, selling_date)
                # 计算该笔持仓赎回的费率
                redeem_rate = self._get_redeem_rate(holding_days)
                shares_yield = shares_info['yield']

                # 注意: 下面逻辑有点绕
                if sell_amount >= shares:
                    # 计算持仓的当前市值
                    shares_value = shares * (1 + shares_yield)
                    selling_value += shares_value

                    # 更新卖出后账户的 pfo_shares_redeem 信息
                    # 卖出金额大于等于当前日期的持有份额，则将份额设为0
                    sorted_holdings[i]['hold'] = 0
                    sorted_holdings[i]['soldout'] = 1
                    sorted_holdings[i]['selling_date'] = selling_date

                    # 分仓卖出，更新 sell_amount 的值
                    sell_amount -= shares

                    # 手续费是分仓独立的，不需要累加
                    redeem_fee = shares_value * redeem_rate
                    selling_cost += redeem_fee
                else:
                    # rest_value: 剩余待卖出的原始份额市值
                    rest_value = sell_amount * (1 + shares_yield)
                    selling_value += rest_value

                    # TODO: 这一步更新有错
                    # 卖出金额小于当前日期的持有份额，则减去卖出金额
                    update_item = copy(sorted_holdings[i])
                    # 拆分卖空的部分
                    update_item['shares'] = round(sell_amount, 2)
                    update_item['hold'] = 0
                    update_item['soldout'] = 1
                    update_item['selling_date'] = selling_date
                    update_items_list.append(update_item)

                    # 更新未卖空的部分
                    sorted_holdings[i]['shares'] = round(shares - sell_amount, 2)
                    sorted_holdings[i]['hold'] = round(shares - sell_amount, 2)
                    sell_amount = 0

                    # 手续费是分仓独立的，不需要累加
                    redeem_fee = rest_value * redeem_rate
                    selling_cost += redeem_fee

                # logging.warning(f'selliing date: {date}, sell_amount original: {selling_shares},  rest: {sell_amount}')

            # 不用减 selling_cost 了，因为 holding 中的 yield 记录的是卖出扣除手续费的收益率
            selling_return = selling_value
            # 要计算扣除收费之后的净收益率
            # 此时的 sell_amount == 0, 因为, 在循环中减完了, 所以除 selling_shares
            return_ratio = round(selling_return / selling_shares - 1, 4)

            if mode == 'LiveTrade':
                sorted_holdings.extend(sold_holdings)
                sorted_holdings.extend(update_items_list)
                self.acct_info['pfo_shares_redeem'][fund_code] = sorted_holdings
                # 卖出股票，现金账户增加金额
                self.acct_info['cash_asset'].append(round(selling_return, 2))
                self.cost += selling_cost

                logging.warning(f'''
                    ---------->
                    卖出日期: {selling_date}, 卖出份额: {selling_shares:0.2f}, 回收现金: {selling_return:0.2f}, 卖出手续费: {selling_cost:0.2f}
                    卖出收益率: {return_ratio:0.4f}, 仓位: {self._get_pfo_ratio():0.2f}, 仓位控制线: {self._set_pfo_ratio():0.2f}
                    trades: {self.trades + 1}
                    ''')
            return return_ratio
        else:
            return 0


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
        if pfo_ratio > pfo_ratio_guideline:
            logging.warning(f'-------> 已达到仓位控制线 {pfo_ratio_guideline}, 暂停加仓 !!!')
            return 0, 0

        stock_name = self.current_data.tic.to_list()[index]
        # 基金使用收盘涨跌幅，收盘价在基金的模拟环境中没有实际使用
        # close_price = self.current_data.close.to_list()[index]

        _mark_point = self.current_data.y_point.tolist()[index]
        _pred_points = self.current_data.y_pred.tolist()[index]

        def _do_buy():
            buy_num_shares = 0
            buy_amount = 0

            # 判断买入的条件:
            if any([
                # 1. 抄底
                _mark_point < 0 and abs(_mark_point) >= abs(_pred_points) * self.temperature,
                # 2. 追涨
                _mark_point > 0 and _mark_point <= _pred_points * (1 - self.temperature)
                ]):

                # 基于单笔最大交易限制的买入策略
                # logging.warning(f'acct cash list ----------> {self.acct_info["cash_asset"]}')
                cash_asset = sum(self.acct_info['cash_asset'])
                available_cash = min(cash_asset, self.per_buy_order_max_amt)
                # 注意：与股票不同，基金直接使用买卖金额，模型输出金额后再换算份额
                available_shares = available_cash

                # 计算可买入的最多股票数量（基于单笔交易金额限制的）
                if available_shares > 0:
                    # 控仓技术
                    pfo_ratio_adj = pfo_ratio_guideline - pfo_ratio - action / self.initial_amount
                    # 仓位补偿：补偿比例太主观，取消！！！TODO: 考虑在牛熊转换的时候加重仓
                    if pfo_ratio_adj > 0:
                        # 暂时不做操作
                        buy_num_shares = min(available_shares, action)
                    # 仓位压缩
                    else:
                        action_adj = self.initial_amount * (pfo_ratio_guideline - pfo_ratio)
                        buy_num_shares = min(available_shares, action_adj)

                    if buy_num_shares > self.per_unit_amount:
                        buy_amount = buy_num_shares * (1 - self.buy_cost_pct[index])
                        buy_fee = buy_num_shares * self.buy_cost_pct[index]

                        # 更新账户的可用本金
                        # 买入股票，现金账户减少金额
                        self.acct_info['cash_asset'].append(round(-buy_num_shares, 2))
                        # 买入股票，增加持仓
                        # self.acct_info['pfo_holding'][stock_name].append(buy_amount)
                        # self.acct_info['pfo_price'][stock_name].append(close_price)

                        # 记录持仓的买入日期
                        self.acct_info['pfo_shares_redeem'].setdefault(stock_name, [])
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


    def _sell_stock(self, index, action):
        '''
        Desc:
            卖出 action. 这个函数交易的是一个股票
        Args:
            index 是一个索引，用于从 self.state 中取出对应的股票的持仓份额 或着 股价
            action 是一个标量数值，表示针对制定 index 股票进行加减仓操作；在 self.action_space 中定义
        '''
        stock_name = self.current_data['tic'].to_list()[index]
        close_price = self.current_data['close'].to_list()[index]
        # 当前的剩余累计持仓
        # cash_asset = sum(self.acct_info['cash_asset'])
        stock_shares, _ = self._get_acct_pfo_shares()
        # logging.warning(f'当前账户持仓 ---------------> 现金: {cash_asset}, 份额: {stock_shares}')

        # 1. 当前可卖出的最大盈利持仓
        max_profit_shares = self._get_max_yield_shares(stock_name, min_yield=1/100)
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

            _mark_point = self.current_data.y_point.tolist()[index]
            _pred_points = self.current_data.y_pred.tolist()[index]

            # 判断卖出的条件: 刚好与买入相反
            if any([
                # 1. 止盈
                _mark_point > 0 and _mark_point > _pred_points * self.temperature,
                # 2. 杀跌
                _mark_point < 0 and abs(_mark_point) < abs(_pred_points) * (1 - self.temperature)
                ]):

                # 判断当前是否有该股票的持仓 & 股价是否大于 0
                if max_profit_shares > 0 and stock_shares > 0:
                    # Sell only if current asset is > 0
                    # 此处与股票不同，注意 ！！！
                    logging.warning(f'action vs max_profit: {abs(action)}, {max_profit_shares:0.2f}')
                    sell_num_shares = max(abs(action), max_profit_shares)

                    if sell_num_shares > 0:
                        # 记录累计已卖出的盈利头寸
                        # logging.warning(f'do sell stock action: {sell_num_shares} quantities.')
                        self.acct_info['profit_shares_sold'][stock_name] += sell_num_shares
                        # 计算卖出可获得的金额，考虑交易费用
                        sell_amount = sell_num_shares

                        # 卖出股票，仓位减少
                        self.acct_info['pfo_holding'][stock_name].append(-sell_num_shares)
                        self.acct_info['pfo_price'][stock_name].append(close_price)
                        return_ratio = self._caculate_selling_return(stock_name, sell_amount, mode='LiveTrade')
                        self.trades += 1

            return sell_num_shares, sell_amount

        sell_num_shares, sell_amount = _do_sell_normal()
        return sell_num_shares, sell_amount
