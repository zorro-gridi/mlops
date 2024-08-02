import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

import pandas as pd
import logging
import time
import random
from pathlib import Path
from pprint import pprint

# from typing import List
import gymnasium as gym
import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from gymnasium import spaces
# from gymnasium.utils import seeding
# from stable_baselines3.common.vec_env import DummyVecEnv
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
# from mlops.ml_ops.rl.finrl.rule.v2 import FundTradeRules_V2
# from mlops.ml_ops.rl.finrl.rule.v5 import FundTradeRules_V5

matplotlib.use("Agg")


"""
@Author: Zorro
@Date: 2024-01-01
@Desc:
    本代码定义了基于 rllib 强化学习训练框架的 gym 环境模型
"""


class FundQuantTradeEnv_V1(BaseTradeEnv):
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
        Features:
            1. 买入时，进行仓位压缩。即买入后的总仓位不能超过仓位策略指导线
            2. 遇到持续满仓时触发早停技术, 主要在 infer mode 推理的情况下使用，节省推理时间
                @release log: 2024-04-12 删除！原因：早停导致策略无法学习，训练和推理环节都不能使用
            3. 卖出时，【策略建议份额】与【盈利持仓】取最大数 (不取最小的原因，因为策略空间有范围限制)
            4. 账户在满足持仓预期收益、或账户整体收益后, 自动实现清仓
        Release log:
            1. 2024-04-17: 增加反弹、反转点的检测，在该点位后的 3 日内不卖出股票, 具体逻辑参考 self.selling_signal()
        Conclusion:
            1. 交易频率较低，整体收益相对温和
            2. 对于单边行情，该策略表现出不适应。特别是，单边下跌行情中，当仓位达到控制线之后，因为限制了策略空间，不能主动有效的减仓，导致策略无法继续训练
        '''
        super().__init__(config)

        # 增加或修改的属性可以写在 super() 后面
        self.goal_yield = config.get('goal_yield', np.inf) # 设置为np.inf 保障训练数据训练完，不至于达到目标退出训练，推理的时候需要定义
        self.phase_yield = config.get('phase_yield', 0.02)
        # 基金的净值很小，而且可以买很少的份数，可以使用连续空间; 缺点：训练太慢
        # self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,), dtype="float32")
        # 最终方案 -> action_space: MultiDiscrete, 多维离散空间
        # 这个 * self.stock_dim 注意写在 MultiDiscrete 里面
        self.action_space = spaces.MultiDiscrete([11] * self.stock_dim) # action: 0~10 的范围
        # 市场交易热度, 影响买卖的频率。其中，0.5-中性; > 0.5-贪婪; < 0.5-谨慎
        self.temperature = config.get('temperature', 0.5)
        # self.complete_times = 3 # 仓位不足时，补仓的倒数比例，这个比例应该和牛熊点位的仓位比例相匹配
        # 例如，市场由牛转猴、熊，仓位比例需要从80%下降到60%，如果初始仓位20%，则 20% + 80% / 3 < 60%，那么这个 3 的比例就是合理的
        # 这个数的具体值还需要调整, 都是主观设置，取消！！！
        self.min_yield = config.get('min_yield', 1/100)

        # 记录仓位控制红线
        self.pfo_ratio_guide = {}
        # 初始化仓位记录, 接着在 self.step 中每次也要先更新仓位
        self._set_pfo_ratio()
        # 反弹，反转点的起始位置
        self.reverse_point_day = -np.inf

        # verbose 辅助信息
        self.verbose = config.get('verbose', 0)
        # 2024-07-21 新增: 卖出费率的字典
        self.sell_cost_pct = config.get('sell_cost_pct', dict())
        # 每次实例化都应该先更新持仓的收益
        self._update_acct_holdings_debit_yield()


    def _get_plan_idx_to_fundcode(self, tic_code, buy_date):
        '''
        Desc:
            获取用户配置的指数定投计划匹配的基金代码
        Args:
            tic_code: 计划定投的指数名称
            buy_date: 定投日期
        Release log:
            2024-06-30: 新增
        '''
        # logging.warning(f'-----------> self.idx_2_fund:\n{self.idx_2_fund}')
        # 训练模式下，只使用指数本身测试
        if self.mode in ['train', 'infer']:
            return tic_code
        else:
            # 取 buy_date 前，历史配置的最新一条数据
            idx_2_fundcode = self.idx_2_fund[
                (self.idx_2_fund['user_id'] == self.user_id) &
                (self.idx_2_fund['plan_id'] == self.plan_id) &
                (self.idx_2_fund['tic'] == tic_code)
                # TODO: ！Important
                # 此处细节需要注意: 去小于当前日期配置的最后一个映射关系, 暂时不需要
                # (self.idx_2_fund['update_date'] <= buy_date)
            ]['fundcode'].iloc[-1]
            # logging.warning(f'------------> idx_2_fundcode: {idx_2_fundcode}')
            return idx_2_fundcode


    def _update_acct_holdings_debit_yield(self):
        '''
        Desc:
            计算当前持仓扣除当日卖出手续费之后的实际收益率.
        Return:
            返回更新收益后的持仓明细
        '''
        # logging.warning(f'acct info -------> {self.acct_info["pfo_shares_redeem"]}')
        # 更新持仓的最新市值
        curr_date = self._get_date()
        update_holdings = self.acct_info['pfo_shares_redeem']
        if self.mode == 'live':
            return update_holdings

        class holding_yield():
            env_cls = self
            def __init__(self, update_holdings) -> None:
                '''
                Args:
                    update_holdings: 待更新的持仓信息
                '''
                self.update_holdings = update_holdings

            def update_holding_yield(self, fund_code, holding_idx):
                '''
                Desc:
                    更新持仓的实际收益, 已经考虑了当日卖出的费率
                Args:
                    fund_code: 对应 idx_name, 定投指数的名称
                    holding_idx: 持仓序列的索引
                Remark:
                    考虑过将买入日期作为持仓字典的 key, 但是持仓分笔卖出时, 会导致 key 重复。dict 不允许重复的 key
                '''
                hold_info = self.update_holdings[fund_code][holding_idx]
                buy_price = hold_info['buy_price']
                soldout = hold_info['soldout']
                buy_date = hold_info['buy_date']
                selling_date = hold_info['selling_date']
                # TODO: train/infer mode 因为没有入数据库，所以 selling_date 默认为 'null'
                selling_date = curr_date if selling_date == 'null' else selling_date

                # 只更新未卖出的
                # 数据库中为 str 类型
                if soldout in ['0',]:
                    # 返回持有天数
                    # days_diff = self.env_cls._calculate_date_diff(buy_date, selling_date)
                    # 卖出收益率 = 持仓收益率 - 卖出费率；其中，持仓收益率 = 收益率 - 买入费率
                    buy_yield = self.env_cls._caculate_holding_yield(fund_code, buy_date, selling_date)
                    # logging.warning(f'------------> buy yield: {buy_yield:0.4f}')
                    # selling_fee = self.env_cls._get_redeem_rate(days_diff)
                    # 更新持仓的实际收益率
                    self.update_holdings[fund_code][holding_idx]['yield'] = round(buy_yield, 4)
                    # TODO: 为什么要更新卖出价格???
                    self.update_holdings[fund_code][holding_idx]['sell_price'] = round(buy_price * (1+buy_yield), 4)

        # 更新账户的持仓收益
        if update_holdings:
            holding_yield_inst = holding_yield(update_holdings)
            [
                holding_yield_inst.update_holding_yield(tic, holding_idx)
                for tic, holdings in update_holdings.items()
                for holding_idx, _ in enumerate(holdings)
                ]

            self.acct_info['pfo_shares_redeem'] = holding_yield_inst.update_holdings
            return holding_yield_inst.update_holdings
        else:
            return {}


    # %%
    def _check_holding_duplicate(self, stock_name, trade_date='buy_date'):
        '''
        Desc:
            检查交易日期是否已采取策略行动。主要使用在推理环节。
            主要是避免非交易时段重复提交交易行为。
        Args:
            stock_name: 交易的标的名称
            trade_date: 交易日期的类型, 可选参数 ["buy_date", "selling_date"]
        '''
        acct_holdings_list = self.acct_info['pfo_shares_redeem'][stock_name]
        # 定投的交易日列表
        trade_date_log = set([hold[trade_date] for hold in acct_holdings_list])

        trade_date = self._get_date()
        # logging.warning(f'-------> trade_date: {trade_date}')

        current_date = time.strftime('%Y-%m-%d')
        is_traded = trade_date in trade_date_log

        # 如果存在历史交易，且非当日交易的为重复交易（因为当日交易允许更新）
        hist_duplicate_cond = all([
            is_traded,
            trade_date != current_date,
            ])

        if hist_duplicate_cond:
            logging.warning(f'Warning ---> _check_holding_duplicate: {trade_date} 历史已经采取买入交易, 请不要重复交易!!!')
            return True

        elif trade_date != current_date:
            logging.warning(f'Warning ---> _check_holding_duplicate: {current_date} 为非交易日, 无法交易 !!!')
            return True

        else:
            logging.warning(f'Warning ---> 未发现交易重复, 正常执行交易🙋‍♂️')
            return False


    def _set_pfo_ratio(self):
        '''
        Desc: Important !!!
            账户的仓位控制策略。基于上证指数的相对牛熊点位判断
            主要更新点:
            1. 添加仓位记录
        Features:
            1. 根据指数和大盘的牛熊点位，动态更新仓位线
            2. 根据点位线的相对百分数，作为调仓的加权百分比。这样做的好处是让仓位管理线控制的更平滑，避免断崖，导致突然无法加、减仓，策略无法学习
        # TODO: Important !!!
        #     当进行多指数定投时，每个指数对应的仓位不同，在买入时不知道如何分配实际的买入金额
        '''
        ratio_strategy = {
            0: 0.8,
            1: 0.5,
            2: 0.3,
            }

        pfo_ratio_guideline = 0
        # 兼容多指数持仓策略
        for index in range(self.stock_dim):
            sz_point_phase = self.current_data['sz_closed_phase'].tolist()[index]
            sz_point_percentile = self.current_data['sz_closed_phase_percentile'].tolist()[index]

            idx_point_phase = self.current_data['closed_phase'].tolist()[index]
            idx_point_percentile = self.current_data['closed_phase_percentile'].tolist()[index]

            # 反弹 / 反转点处一次性提高仓位
            if self.current_data['is_reverse_point'].tolist()[index] == 1:
                idx_pfo_ratio = ratio_strategy[idx_point_phase]

            else:
                idx_pfo_ratio = round(
                    (ratio_strategy[sz_point_phase] * (1 - sz_point_percentile)
                    + ratio_strategy[idx_point_phase] * (1 - idx_point_percentile)
                    ) / 2, 3)

            # 使用持仓指数中建议的最大仓位
            if idx_pfo_ratio > pfo_ratio_guideline:
                pfo_ratio_guideline = idx_pfo_ratio

        # 记录仓位
        self.pfo_ratio_guide[self.day] = pfo_ratio_guideline
        return round(pfo_ratio_guideline, 3)


    def _get_acct_pfo_shares(self):
        '''
        Desc:
            调用 _update_acct_holdings_debit_yield 方法，计算账户资产的最新市值
        Return:
            pfo_shares: 持仓的份额
            pfo_asset: 持仓的市值
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

        cash_asset = sum(self.acct_info['cash_asset'].values())
        # 统计持仓的当前市值
        _, pfo_asset = self._get_acct_pfo_shares()
        total_asset = pfo_asset + cash_asset
        # logging.warning(f'current acct ----> pfo_asset: {pfo_asset}, cash_asset: {cash_asset}')
        return total_asset


    def step(self, actions, **kwargs):
        '''
        Desc:
            继承并改写父类的 step 方法，主要功能如下：
            1. 更新 actions 的分布
            2. 执行agent的买卖策略之前, 先执行账户自定义管理策略：检查账户的持仓收益率和清仓累计收益率，达到预期则清仓
            3. 判断是否清仓的条件后, 再执行agent的买卖策略
        '''
        # 每一步先更新当前的仓位控制线
        self._set_pfo_ratio()
        # MultiDiscrete start 参数在实际运行中不起作用，需要手动调节 actions
        actions = actions - 5
        # logging.warning(f'actions ---------------> {actions}')

        # TODO: 写一个触发清仓的条件
        pfo_yield = self._get_pfo_soldout_yield()
        cumsum_yield = self._get_acct_cumsum_yield()

        # 达到阶段清仓条件: 1. 持仓达到阶段目标；2. 账户收益达到预期目标
        if (pfo_yield >= self.phase_yield) or (cumsum_yield >= self.goal_yield):
            # 执行清仓操作: 先执行动作，再变更状态
            self.acct_pfo_soldout()

            # 达到整体收益率目标，发出停止交易的信号: self.goal_achieved = 1
            # 注意，这个信号只能在交易的时候使用
            # 达到目标收益清仓
            if cumsum_yield >= self.goal_yield:
                logging.warning(f'当前账户【清仓累计收益率】: {cumsum_yield:0.4f}, 达到【预期收益率】目标: {self.goal_yield}, 账户清仓 !!!')
                self.goal_achieved = True
            else:
                logging.warning(f'当前账户【持仓清仓收益率】: {pfo_yield:0.4f}, 达到【阶段收益率】目标: {self.phase_yield}, 账户清仓 !!!')

        return super().step(actions, **kwargs)


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
            根据持有天数, 返回赎回基金的手续费率
        '''
        # days_range_max = 730
        # if days >= days_range_max:
        #     logging.warning(f'-------> days: {days}, limit: {days_range_max}, days redeem rate mapping not set!!!')
        #     raise
        # redeem_rate_info = {
        #     7: 1.5 / 100,
        #     30: 0.75 / 100,
        #     365: 0.5 / 100,
        #     730: 0.25 / 100,
        #     }

        if not self.sell_cost_pct:
            raise

        redeem_rate = 0
        for d, rate in self.sell_cost_pct.items():
            if days < d:
                redeem_rate = rate
                break

        return redeem_rate


    def _stat_redemm_rate_balance(self, fund_code):
        '''
        Desc:
            统计持仓账户的卖出手续费率余额分布
        Release log:
            2024-06-28: 新增
        '''
        acct_holdings = self.acct_info['pfo_shares_redeem'][fund_code]
        still_holdings = [h for h in acct_holdings if h['soldout'] == '0']

        redeem_rate_balance = {}
        for h in still_holdings:
            hold_shares = h['hold']
            buy_date = h['buy_date']
            curr_date = self._get_date()
            days_diff = self._calculate_date_diff(buy_date, curr_date)
            redeem_rate = self._get_redeem_rate(days_diff)

            redeem_rate_balance.setdefault(redeem_rate, 0)
            redeem_rate_balance[redeem_rate] += hold_shares

        redeem_rate_balance = dict(sorted(redeem_rate_balance.items()))
        return redeem_rate_balance


    def _cal_fifo_redeem_rate(self, fund_code, sell_amount, mode='LiveTrade'):
        '''
        Desc:
            实现给定一个卖出份额, 计算预计的卖出综合费率
            2024-06-27: 按照"FIFO 先进先出"的规则计算实际卖出费率。
        Args:
            sell_amount: 卖出的份额
            mode: 测试或者生产模式，区别在于是否更新账户数据. Option: ["LiveTrade", "Backtest"]
        Return:
            返回卖出的费率
        Release log:
            2024-06-27: 新增
            2024-06-28: 增加 redeem_balance 剩余手续费余额处理逻辑
        '''
        if sell_amount == 0:
            return 0
        acct_holdings = self.acct_info['pfo_shares_redeem'][fund_code]
        # 已兑换掉费率额度的单独摘开
        redeemOut_holdings = [h for h in acct_holdings if h['redeem_balance'] <= 0]
        # 未兑换费率额度的循环计算卖出费率
        still_holdings = [h for h in acct_holdings if h['redeem_balance'] > 0]
        logging.warning(f'-----------> acct redeem balance holdings:\n{pd.DataFrame(still_holdings)}\n')

        # 越早买入的份额，需要越早清仓; still_holdings 是列表
        sort_holdings = list(sorted(still_holdings, key=lambda x: x['buy_date']))
        total_fee = 0
        sell_amount_init = copy(sell_amount)

        for idx, h in enumerate(sort_holdings):
            hold_shares = h['hold']
            buy_date = h['buy_date']
            curr_date = self._get_date()
            days_diff = self._calculate_date_diff(buy_date, curr_date)
            redeem_rate = self._get_redeem_rate(days_diff)

            if sell_amount > hold_shares:
                redeem_fee = hold_shares * redeem_rate
                sort_holdings[idx]['redeem_balance'] = 0
            else:
                redeem_fee = sell_amount * redeem_rate
                sort_holdings[idx]['redeem_balance'] = hold_shares - sell_amount

            total_fee += redeem_fee
            sell_amount -= hold_shares
            if sell_amount <= 0:
                break

        # 更新持仓的 redeem_balance 信息
        if mode == 'LiveTrade':
            # 合并清空的holding和已更新的holding
            redeemOut_holdings.extend(sort_holdings)
            self.acct_info['pfo_shares_redeem'][fund_code] = redeemOut_holdings

        total_redeem_rate = round(total_fee / sell_amount_init, 5)
        return total_redeem_rate


    def _cal_max_selling_amount_with_min_yield(self, fund_code, min_yield=0.01):
        '''
        Desc:
            计算考虑 FIFO 规则，且满足最小止盈的可卖出的最大份额
        Args:
            fund_code: indexname 指数名称
        Release log:
            2024-06-27: 新增
        '''
        # 获取账户达到预期收益的所有持仓份额（该函数也同步更新了持仓收益）
        # 先更新持仓的收益率；
        acct_holdings = self._update_acct_holdings_debit_yield()
        if not acct_holdings:
            logging.warning(f'----------------> 没有发现账户持仓!!!')
            return 0
        # logging.warning(f'-----------> acct holdings:')
        # pprint(acct_holdings)

        tic_holdings = acct_holdings[fund_code]
        # tic_holdings = self.acct_info['pfo_shares_redeem'][fund_code]
        still_holdings = [h for h in tic_holdings if h['soldout'] == '0' and h['hold'] > 0]
        # 此处得按 yield 收益率逆序排序
        sort_holdings = list(sorted(still_holdings, key=lambda x: x['yield'], reverse=True))
        # logging.warning(f'-----------> sort_holdings:')
        # pprint(sort_holdings)

        max_selling_amount = 0              # 循环中累计的卖出累计份额
        max_received_value = 0              # 循环中累计的卖出可到账金额
        final_max_selling_amount = 0        # 最终决策的卖出累计数量
        # max_selling_fee = 0                 # 最终卖出时的费率份额
        # find_redeem_rate = 0                # 最终决策卖出份额的综合费率
        total_selling_yield = 0             # 循环中卖出的累计收益率

        for h in sort_holdings:
            # 在卖出阶段，如果被拆分，此处的 buy_shares 就是一笔的部分份额
            # 买入时到账的份额
            buy_shares = round(h['shares'] * (1 - h['buy_rate']) / h['buy_price'], 2)
            buy_date = h['buy_date']
            # 可卖出的持仓份额
            sell_amount = h['hold']
            hold_yield = h['yield'] + self.live_markup

            # 动态最下止盈收益率
            dyn_min_yield = self._caculate_holding_min_yield(fund_code, buy_date)
            # 计算卖出一笔持仓基于 fifo 规则的费率
            redeem_rate = self._cal_fifo_redeem_rate(fund_code, sell_amount, mode='Backtest')
            # 考虑 申购 & 赎回 的净收益率
            selling_yield = round(hold_yield - redeem_rate, 6)
            logging.warning(f'''
                ----------->
                buy_date: {buy_date} sell_amount: {sell_amount}
                yield: {h['yield']} live_markup: {self.live_markup} redeem_rate: {redeem_rate}
                dyn_min_yield: {dyn_min_yield} selling_yield: {selling_yield}
                ''')

            # 如果该份额卖出的收益率比动态止盈收益率低，则跳过不卖
            if selling_yield < dyn_min_yield:
                logging.warning(f'------------> 没有找到达到预期目标 {dyn_min_yield} 的持仓')
                break

            # TODO: 此处有两种模式: 选择模式一
            # 一， 整体（即考虑亏损持仓）总卖出收益达到 min_yield
            # 二， 必须每一笔都达到 min_yield
            # max_selling_fee += round(sell_amount * redeem_rate, 2)
            max_selling_amount += sell_amount
            max_received_value += sell_amount * (1 + selling_yield)
            # 卖出的综合赎回收益率
            total_selling_yield = max_received_value / max_selling_amount - 1
            logging.warning(f'-----------> total_selling_yield: {total_selling_yield:.4f}')

            # !!! important 此处的条件逻辑有点绕:
            # 1. 必须要达到最小止盈收益率：因为卖出止盈必须达到最小止盈收益率；
            # 2. 卖出的份额不能超过达到目标止盈收益的累计持仓份额, 解释如下:
                # 2.1 达到目标止盈的累计持仓肯定优先卖出, 因此, 这个总数是理论上🉑️卖出的总数
                # 2.2 卖出的整体份额又必须达到最小止盈收益率
            # 综合 2.1/2.2 的条件，卖出的份额判断即完整统一, 触发任何一个条件则停止搜素，定格最大可卖出持仓
            if total_selling_yield <= min_yield:
                break

        final_max_selling_amount = max_selling_amount
        # find_redeem_rate = round(max_selling_fee / max_selling_amount, 4)
        return final_max_selling_amount


    def _get_max_yield_shares(self, fund_code, min_yield=None):
        '''
        Desc:
            计算给定卖出日期，持仓账户达到最小盈利的累计持仓数量
        Args:
            fund_code: 计划定投的指数名称
            min_yield: 最小盈利阈值
        Return:
            max_yield_shares: 当前可卖出的已盈利的最大持仓数量
        Release log:
            1. 2024-04-18: 将每一笔持仓的最小预期收益率改成 _caculate_holding_min_yield 函数动态计算
        TODO:
            持仓的最大止盈策略写的太死，低位买入的持仓可以适度提高止盈限制
        '''
        update_holdings = self._update_acct_holdings_debit_yield()
        max_yield_shares = 0

        if not update_holdings:
            return 0

        holdings = update_holdings[fund_code]
        holdings = [
            s for s in holdings
            if s['soldout'] == '0'
                and s['hold'] > 0
                # and s['yield'] >= min_yield
                # 使用动态最小期望收益率
                and s['yield'] >= (
                    min_yield if min_yield else self._caculate_holding_min_yield(fund_code, s['buy_date'])
                    )
            ]

        if holdings:
            # logging.warning(f'test acct holding yield -------->\n{pd.DataFrame(holdings)}')
            max_yield_shares = sum([s['hold'] for s in holdings])
            # logging.warning(f'当前可卖的累计盈利的持仓 ------------> 份额: {shares}, yield: {selling_return}')
        return max_yield_shares


    def _caculate_holding_yield(self, tic_code, buy_date, sell_date):
        '''
        Desc:
            统计基金卖出时的毛收益率, 不考虑申购费率、和赎回费率
        Args:
            tic_code: 计划定投的指数名称, egg: 医药生物，传媒, ...
            buy_date: 买入日期
            sell_date: 卖出日期
        Return:
            fund_yield: 卖出的收益率%
        Remark:
            主要应用在 train 模式，或者 infer 模型训练好之后，回测具体基金代码
        '''
        # 训练阶段只使用指数走势训练
        if self.mode in ['train']:
            markup_data = self.raw_data
        elif self.mode in ['infer']:
            markup_data = self.fund_data
            # 取 buy_date 前，历史配置的最新一条数据
            idx_2_fundcode = self._get_plan_idx_to_fundcode(tic_code, buy_date)
            # logging.warning(f'---------------> idx_2_fundcode: {tic_code} vs {idx_2_fundcode}')
        else:
            raise Exception(f'-----------> 尚未定义的模式')

        # logging.warning(f'-----------> markup_data: {markup_data}')
        # markup_data: 计算持仓基金标的涨跌幅的原始行情数据
        fund_data = markup_data.loc[
            (markup_data['tic'] == idx_2_fundcode) &
            (markup_data['date'] > buy_date) &
            (markup_data['date'] <= sell_date)
            ].copy()

        # logging.warning(f'--------------> fund networth data:\n{fund_data}')
        if len(fund_data) > 0:
            # 这个计算是否有错？答：没错！因为筛选条件已经过滤了买入当日的涨跌幅
            # TODO: 基金其实可以使用净值直接计算涨跌幅，指数可以用点位计算；但是RL模型没有使用指数点位作为变量，因此在训练时无法使用点位计算涨跌幅
            fund_yield = float(fund_data['close'].sum()) / 100
            # logging.warning(f'----------> buy_date: {buy_date}, selling_date: {sell_date}, selling yield: {fund_yield}:.4f')
            return fund_yield

        # logging.warning(f'--------------> 当日新买入，无法计算收益')
        return 0


    def _caculate_soldout_cost_fee(self):
        '''
        Desc:
            计算清仓时的卖出手续费; 卖出手续费 = 持仓量 * 卖出费率
        '''
        acct_holdings = self._update_acct_holdings_debit_yield()
        soldout_date = self._get_date()

        soldout_fee = sum([
            s['hold'] * self._get_redeem_rate(self._calculate_date_diff(s['buy_date'], soldout_date))
            for holdings in acct_holdings.values()
            for s in holdings
            if s['soldout'] == '0'
            ])
        return soldout_fee


    def _get_pfo_soldout_yield(self):
        '''
        Desc:
            计算账户持仓【清仓时】扣除手续费的卖出收益率 = 所有持仓市值 / 所有持仓的成本 - 1
        Returns:
            pfo_yield: 卖出的持仓收益率
        '''
        # TODO: pfo_shares 在生产条件是持仓的份额，还不是市值
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
            执行账户所有持仓的清仓操作, 并更新账户的持仓信息
        '''
        acct_holdings = self._update_acct_holdings_debit_yield()
        soldout_date = self._get_date()
        soldout_fee = self._caculate_soldout_cost_fee()
        _, selling_return = self._get_acct_pfo_shares()
        self.cost += soldout_fee
        self.soldout += 1
        self.trades += 1

        # 生产模式的清仓操作与训练环境不同
        # 此处的 fundcode 只是指数的名称, 未映射到基金代码
        total_sell_num_shares = 0
        for fundcode, holdings in acct_holdings.items():
            for h in holdings:
                sell_num_shares = h['hold']
                if h['soldout'] == '1' or sell_num_shares == 0:
                    continue
                total_sell_num_shares += sell_num_shares

        # 清仓合成一笔订单, 相同的订单类型，订单日期不能重复
        self.acct_info['order'].append({
            'order_id': str(random.randint(1e18, 9e18)),
            'order_date': soldout_date,
            'order_type': 1,
            'order_amount': total_sell_num_shares,
            'fundcode': self._get_plan_idx_to_fundcode(fundcode, self._get_date()),
            'fee_rate': 'null',
            'order_fee': 'null',
            'net_worth': 'null',
            'received_amount': 'null',
            'opt_type': 1,
            'order_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            })

        # 使用列表推倒式速度更快
        class holdings_soldout():
            env_cls = self
            def __init__(self, acct_holdings) -> None:
                self.acct_holdings = acct_holdings

            def hold_soldout(self, fund_code, hold_idx):
                self.acct_holdings[fund_code][hold_idx]['soldout'] = 1
                self.acct_holdings[fund_code][hold_idx]['selling_date'] = soldout_date
                self.acct_holdings[fund_code][hold_idx]['hold'] = 0

        holding_soldout_inst = holdings_soldout(acct_holdings)
        [
            holding_soldout_inst.hold_soldout(fund_code, hold_idx)
            for fund_code, holdings in acct_holdings.items()
            for hold_idx, s in enumerate(holdings)
            if  s['soldout'] == '0'
            ]

        self.acct_info['pfo_shares_redeem'] = holding_soldout_inst.acct_holdings
        # 卖出股票，现金账户增加金额
        self.acct_info['cash_asset'][self._get_date()] = round(selling_return, 2)


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
        Release log:
            2024-06-27:
                1. 给持仓添加 hold_id 主键
                2. 修复基于 FIFO 规则的卖出判断逻辑; TODO: 需要修改每一笔持仓的实际卖出费率
            2024-07-16:
                1. 新增返回卖出的手续费率 redeem_rate
        '''
        task_start = time.time()
        update_holdings = self._update_acct_holdings_debit_yield()
        acct_holdings = copy(update_holdings[fund_code])

        sold_holdings = [
            s for s in acct_holdings if s['soldout'] == 1]
        still_holdings = [
            s for s in acct_holdings if s['soldout'] == '0']

        if still_holdings and sell_amount > 0:
            # logging.warning(f'pfo_shares_redeem ----------> {still_holdings}')
            # 计算扣除卖出手续费的收益率就是为了按照收益率真实排序
            # sorted_holdings: sorted of still holding
            sorted_holdings = list(sorted(still_holdings, key=lambda x: x['yield'], reverse=True))
            # logging.warning(f'sorted_holdings ------->\n{pd.DataFrame(sorted_holdings)}')

            selling_shares = copy(sell_amount)                # 统计卖出的原始持仓金额
            selling_date = self._get_date()

            # 使用类方法列表推倒式循环，更快
            class update_holding:
                '''
                Desc:
                    类示例初始化
                '''
                env_clf = self
                def __init__(self, sell_amount):
                    '''
                    Desc:
                        初始化属性
                    Attr:
                        sell_amount: 需要卖出的金额
                        selling_cost: 卖出的交易手续费
                        selling_value: 卖出的实际到账金额
                        update_items_list: 当执行卖出操作的份额小于持仓收益最高的份额, 需要额外添加的持仓记录
                    '''
                    self.sell_amount = sell_amount
                    # 卖出份额的手续费
                    self.selling_cost = 0
                    self.selling_value = 0
                    self.update_items_list = []

                def _update_holding_info_after_selling(self, i, shares_info):
                    # 在循环体中 sell_amount 越减越少
                    if round(self.sell_amount, 2) == 0:
                        return

                    # buy_date = shares_info['buy_date']
                    shares = shares_info['hold']
                    # 获取持仓的天数
                    # holding_days = self.env_clf._calculate_date_diff(buy_date, selling_date)
                    # 根据持仓天数, 计算该笔持仓的赎回费率
                    # redeem_rate = self.env_clf._get_redeem_rate(holding_days)
                    shares_yield = shares_info['yield']

                    # 注意: 下面逻辑有点绕
                    # *******************************************************************
                    # 1. 如果要卖出的金额比单笔持仓高, 则先将该笔持仓设置为卖空状态, 即 soldout = 1
                    if self.sell_amount >= shares:
                        # 根据卖出份额，计算对应的卖出费率
                        redeem_rate = self.env_clf._cal_fifo_redeem_rate(fund_code, shares, mode=mode)
                        # 计算该笔持仓的当前市值, 扣除卖出手续费(申购手续费已经在买入的市值中扣除, 不用减)
                        shares_value = shares * (1 + shares_yield) * (1 - redeem_rate)
                        # 累加预计回收的金额
                        self.selling_value += shares_value

                        # 更新卖出后账户的 pfo_shares_redeem 信息
                        # 卖出金额大于等于当前日期的持有份额，则将份额设为0
                        sorted_holdings[i]['hold'] = 0
                        sorted_holdings[i]['soldout'] = 1
                        sorted_holdings[i]['sold_shares'] = shares
                        sorted_holdings[i]['selling_date'] = selling_date
                        sorted_holdings[i]['redeem_rate'] = redeem_rate
                        sorted_holdings[i]['sell_price'] = (1 + shares_yield)
                        # 分仓卖出，更新剩余待卖出金额 sell_amount
                        self.sell_amount -= shares

                        # 手续费是分仓独立的，不需要累加
                        redeem_fee = shares_value * redeem_rate
                        self.selling_cost += redeem_fee
                    # 2. 如果要卖出的金额小于单笔持仓金额，需要将单笔持仓拆分为两部分：!important
                    #    卖出金额的部分需设置为卖空状态: soldout = 1, 另一部分则继续保留
                    else:
                        # 根据卖出份额，计算对应的卖出费率
                        redeem_rate = self.env_clf._cal_fifo_redeem_rate(fund_code, self.sell_amount, mode=mode)
                        # rest_value: 剩余待卖出的原始份额市值
                        rest_value = self.sell_amount * (1 + shares_yield) * (1 - redeem_rate)
                        self.selling_value += rest_value

                        # 在持仓中构造一份卖空的部分, 这部分后续也需要一起 extend 进持仓的明细中
                        update_item = copy(sorted_holdings[i])
                        update_item['shares'] = round(self.sell_amount, 2)
                        update_item['hold'] = 0
                        update_item['soldout'] = 1
                        update_item['sold_shares'] = self.sell_amount
                        update_item['selling_date'] = selling_date
                        update_item['redeem_rate'] = redeem_rate
                        # 因为训练数据 buy_price = 1, hold 即为买入的到账金额；注意：不适用于生产数据（需要乘以 sell_price）
                        update_item['sell_price'] = (1 + shares_yield)
                        # 2024-06-27 bug 修复: 拆分 hold 重新赋值一个 hold id, 确保主键唯一
                        update_item['hold_id'] = str(random.randint(1e18, 9e18))

                        self.update_items_list.append(update_item)

                        # 更新未卖空的部分
                        sorted_holdings[i]['shares'] = round(shares - self.sell_amount, 2)
                        sorted_holdings[i]['hold'] = round(shares - self.sell_amount, 2)
                        self.sell_amount = 0

                        # 手续费是分仓独立的，不需要累加
                        redeem_fee = rest_value * redeem_rate
                        self.selling_cost += redeem_fee

                # 主要测试卖出轮次中的剩余待卖出金额的变化
                # logging.warning(f'selliing date: {date}, sell_amount original: {selling_shares},  rest: {sell_amount}')

            cal_holdings = update_holding(sell_amount)
            # 列表推倒式加快迭代速度
            [cal_holdings._update_holding_info_after_selling(i, share_info)
             for i, share_info in enumerate(sorted_holdings)]

            # !!! 不用减 selling_cost 了，因为 holding 中的 yield 记录的是卖出扣除手续费的收益率
            # 这样做的目的是保证卖出时使用考虑到卖出手续费的优先持仓部分
            # selling_return = selling_value
            selling_return = cal_holdings.selling_value
            # 要计算扣除收费之后的净收益率
            # 此时的 sell_amount == 0, 因为, 在循环中减完了, 所以除 selling_shares
            return_ratio = round(selling_return / selling_shares - 1, 4)

            if mode == 'LiveTrade':
                sorted_holdings.extend(sold_holdings)
                sorted_holdings.extend(cal_holdings.update_items_list)
                self.acct_info['pfo_shares_redeem'][fund_code] = sorted_holdings

                # 卖出股票，现金账户增加金额
                self.acct_info['cash_asset'][self._get_date()] = round(selling_return, 2)
                self.cost += cal_holdings.selling_cost
                self.trades += 1

                if self.verbose == 1:
                    task_end = time.time()
                    logging.warning(f'''
                        ---------->
                        卖出日期: {selling_date}, 卖出份额: {selling_shares:0.2f}, 回收现金: {selling_return:0.2f}, 卖出手续费: {cal_holdings.selling_cost:0.2f}
                        卖出收益率: {return_ratio:0.4f}, 仓位: {self._get_pfo_ratio():0.2f}, 仓位控制线: {self._set_pfo_ratio():0.2f}
                        trades: {self.trades}
                        time consume: {(task_end - task_start):0.2f} s
                        ''')
                    # logging.warning(f'selling yield time consume ----> {(task_end - task_start):0.2f} s')

            redeem_rate = round(cal_holdings.selling_cost / selling_shares, 4)
            return return_ratio, redeem_rate
        else:
            return 0, 0


    def _check_reverse_point(self, index):
        '''
        Desc:
            记录最新的反弹 / 反转点的位置
        '''
        is_reverse_point = self.current_data['is_reverse_point'].tolist()[index] == 1
        if is_reverse_point:
            # reverse_point_day：将day设置为反转day
            self.reverse_point_day = self.day
        return is_reverse_point

    def buying_signal(self, index):
        '''
        Desc:
            判断买入的市场条件。区分训练模式与实盘交易模式。还需要结合长短期的均线趋势
        '''
        is_reverse_point = self._check_reverse_point(index)
        _mark_point = self.current_data.y_point.tolist()[index]
        _pred_points = self.current_data.y_pred.tolist()[index]
        test_loss = int(self.current_data.test_loss.tolist()[index])
        # 主要针对买入，此时_mark_point为负
        adj_pred_points = _pred_points - test_loss

        # TODO: 根本错误是因为连跌天数预测的不准
        chaodi_signal = any([
            all([
                _mark_point <= _pred_points * (1 - self.temperature),
                _mark_point < 0 and _mark_point >= _pred_points,
                ]),
            _mark_point < 0 and _mark_point <= adj_pred_points,
            ])

        zhuizhang_signal = all([
            _mark_point <= _pred_points * self.temperature,
            _mark_point > 0,
            ])

        return any([
            # 1. 下跌时，抄底
           chaodi_signal,
            # 2. 上涨时，追涨
            zhuizhang_signal,
            # 3. 反弹 / 反转的地步也可以买入
            is_reverse_point == 1
            ])

    def selling_signal(self, index):
        '''
        Desc:
            判断卖出的市场条件。特殊情况: 遇到反转点, 3天内不许卖出
        '''
        is_reverse_point = self._check_reverse_point(index)
        _mark_point = self.current_data.y_point.tolist()[index]
        _pred_points = self.current_data.y_pred.tolist()[index]

        shadie_signal = all([
            _mark_point >= _pred_points * self.temperature,
            _mark_point < 0,
            ])

        # 判断卖出的条件: 刚好与买入相反
        return any([
            # 1. 上涨时，止盈
            _mark_point >= _pred_points * self.temperature and _mark_point > 0,
            # 2. 下跌时，杀跌
            shadie_signal,
            # 3. 反弹、反转 3 天内不卖出
            # TODO: 会不会与上面的逻辑1， 2冲突
            self.day - self.reverse_point_day >= 3 and is_reverse_point == 1,
            # 4. 当日大涨, 如果不是反转点也可以减仓
            self.live_markup >= 1 and is_reverse_point == 0,
            ])


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

        if pfo_ratio > pfo_ratio_guideline:
            if self.verbose == 1:
                logging.warning(f'-------> trade date: {self._get_date()}, 当前仓位: {pfo_ratio}, 已达到仓位控制线 {pfo_ratio_guideline}, 暂停加仓 !!!')

            # 取消早停，不合理，策略必须一直进行下去
            # self.stop_buying += 1
            # if self.stop_buying >= self.early_stop_times:
            #     self.truncate = True
            #     logging.warning(f'meet stop buying times -----------> {self.stop_buying}')
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
                # 最大可补的仓位
                pfo_amount = round(self.initial_amount * (pfo_ratio_guideline - pfo_ratio), 1)
                available_cash = min(cash_asset, self.per_buy_order_max_amt, pfo_amount)
                # 注意：与股票不同，基金直接使用买卖金额，模型输出金额后再换算份额
                available_shares = available_cash

                # 计算可买入的最多股票数量（基于单笔交易金额限制的）
                if available_shares > 0:
                    # logging.warning(f'-----------> buying amont choices: rule: {available_shares} vs action: {action}')
                    buy_num_shares = min(available_shares, action)
                    if buy_num_shares > self.per_unit_amount:
                        buy_amount = buy_num_shares * (1 - self.buy_cost_pct[index])
                        buy_fee = buy_num_shares * self.buy_cost_pct[index]

                        # 记录持仓的买入日期
                        self.acct_info['pfo_shares_redeem'].setdefault(stock_name, [])
                        if self.mode in ['infer', 'live'] and self._check_holding_duplicate(stock_name, trade_date='buy_date'):
                            return 0, 0

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
                                'received_amount': 'null' if self.mode == 'live' else buy_amount,  # 入账的金额
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

                            # 更新账户的可用本金
                            # 买入股票，现金账户减少金额
                            self.acct_info['cash_asset'][self._get_date()] = round(-buy_num_shares, 2)
                            # 更新买入的手续费
                            self.cost += buy_fee
                            # 更新交易频次，不能写在 step 函数中
                            self.trades += 1
                            # logging.warning(f"acct info ---> {self.acct_info['pfo_shares_redeem']}")

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
                            })
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
        action = abs(action)
        stock_name = self.current_data['tic'].to_list()[index]
        close_price = self.current_data['close'].to_list()[index]
        # 当前的剩余累计持仓
        # cash_asset = sum(self.acct_info['cash_asset'].values())
        stock_shares, _ = self._get_acct_pfo_shares()
        # logging.warning(f'当前账户持仓 ---------------> 现金: {cash_asset}, 份额: {stock_shares}')

        # 1. 当前可卖出的最大盈利持仓
        # 2024-06-28 更新
        max_profit_shares = self._cal_max_selling_amount_with_min_yield(stock_name, min_yield=self.min_yield)
        # logging.warning(f'当前盈利持仓 ---------------> {max_profit_shares}')
        # rl action 对应的卖出综合费率
        # action_redeem_rate = self._cal_fifo_redeem_rate(stock_name, action, mode='Backtest')
        # check if the stock is able to sell, for simlicity we just add it in techical index
        # 也就是说，对应的股票是否可以交易，在技术指标中内置了。因为可能有些股票当日停牌，不可交易
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
                if max_profit_shares > 0 and stock_shares > 0:
                    # Sell only if current asset is > 0
                    # 此处与股票不同，注意 ！！！
                    # 只能卖出盈利的持仓
                    # logging.warning(f'action vs max_profit: {abs(action)} vs {max_profit_shares:0.2f}')
                    sell_num_shares = min(action, max_profit_shares)
                    if sell_num_shares > 0:
                        # 记录累计已卖出的盈利头寸
                        # logging.warning(f'do sell stock action: {sell_num_shares} quantities.')
                        # self.acct_info['profit_shares_sold'][stock_name] += sell_num_shares
                        # 计算卖出可获得的金额，考虑交易费用
                        sell_amount = sell_num_shares

                        _, fee_rate = self._caculate_selling_return(stock_name, sell_amount, mode='LiveTrade')
                        # 生产模式下，直接返回卖出份额，待 GRIDi 产品更新持仓
                        self.acct_info['order'].append({
                            'order_id': str(random.randint(1e18, 9e18)),
                            'order_date': self._get_date(),
                            'order_type': 0,
                            'order_amount': sell_num_shares,
                            'fundcode': self._get_plan_idx_to_fundcode(stock_name, self._get_date()),
                            'fee_rate': fee_rate,
                            'order_fee': 'null',
                            'net_worth': 'null',
                            'received_amount': sell_num_shares,
                            'opt_type': 3,
                            'order_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                            })
            return sell_num_shares, sell_amount
        sell_num_shares, sell_amount = _do_sell_normal()
        return sell_num_shares, sell_amount


    def _caculate_holding_min_yield(self, fund_code, buy_date):
        '''
        Desc:
            计算每一笔买入持仓的最小预期收益率, 特别是对于在反弹、反转底部买入的持仓，需要扩大期望收益率。
            本预期收益策略仅针对波动性的指数设计，对于指数的单边行情不适用
        Args:
            fund_code: 计划定投的指数名称
            buy_date: 买入日期
        Release log:
            1. 2024-04-18: 新增
            2. TODO: 这个动态收益还是放在外部模型计算
        '''
        fund_data = self.raw_data.loc[
            (self.raw_data['tic'] == fund_code) &
            (self.raw_data['date'] == buy_date)
            ]
        if len(fund_data) == 0:
            raise Exception(f'----------> Exception: self.raw_data 中找不到 {fund_code} & {buy_date} 数据记录')

        is_reverse_point = fund_data['is_reverse_point'].max()
        idx_percentile = fund_data['closed_phase_percentile'].max()
        idx_phase = fund_data['closed_phase'].max()

        # 反转的预期收益率
        reverse_rate = 0.06
        phase_exp_yield = {
            0: [1 / 100, 1 / 100],
            1: [1 / 100, 1 / 100],
            2: [0.5 / 100, 1 / 100],
            }

        if is_reverse_point == 1:
            return reverse_rate
        else:
            # 最高的止盈范围
            clip_yield = phase_exp_yield[idx_phase][1]
            if idx_percentile > 0:
                exp_yield = phase_exp_yield[idx_phase][0] * (1 / idx_percentile)
                exp_yield = round(min(exp_yield, clip_yield), 3)
            else:
                exp_yield = clip_yield
            return exp_yield