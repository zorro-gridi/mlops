import numpy as np
import logging

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dir_list = [dname for dname in current_dir.as_posix().split('/')]
home_dir = '/'.join([dirname for dirname in dir_list[:dir_list.index('zorro')+1]])
env_path = '/'.join([dirname for dirname in dir_list[:dir_list.index('pycharm')+1]])
sys.path.append(env_path)

from mlops.ml_ops.rl.finrl.rule.v2 import FundTradeRules_V2


class FundTradeRules_V5(FundTradeRules_V2):
    def __init__(self, fund_env) -> None:
        self.fund_env = fund_env

    def buy_rules(self, index, action):
        '''
        Desc:
            *
        Args:
            index:
            action:
        '''
        # 更新仓位控制线
        pfo_ratio_guideline = self.fund_env._set_pfo_ratio()
        pfo_ratio = self.fund_env._get_pfo_ratio()

        last_day = max(self.fund_env.day-1, 0)
        plus_pfo_ratio = 0  # 补仓空间初始化
        plus_buy_amount = 0 # 最大可补的仓位初始化
        is_buying_accept = self.fund_env.buying_signal(index)

        if self.fund_env.pfo_ratio_guide:
            last_pfo_ratio_guide = self.fund_env.pfo_ratio_guide[last_day]
            plus_pfo_ratio = last_pfo_ratio_guide - pfo_ratio_guideline
            plus_buy_amount = round(self.fund_env.initial_amount * plus_pfo_ratio, 1)

        # 如果当前已到仓位指导线，则停止加仓
        if pfo_ratio >= pfo_ratio_guideline:
            if self.fund_env.verbose == 1:
                logging.warning(f'''
                    ------->
                    trade date: {self.fund_env._get_date()}
                    指数牛熊位置: {self.fund_env.current_data['closed_phase'].max()}, 百分位: {self.fund_env.current_data['closed_phase_percentile'].max()}
                    当前仓位: {pfo_ratio}, 已达到仓位控制线 {pfo_ratio_guideline}, 暂停加仓 !!!
                    ''')
            return 0, 0

        is_reverse_point = self.fund_env.current_data['is_reverse_point'].tolist()[index] == 1
        # 基金使用收盘涨跌幅，收盘价在基金的模拟环境中没有实际使用
        # close_price = self.fund_env.current_data.close.to_list()[index]

        def _do_buy():
            buy_num_shares = 0
            buy_amount = 0

            # 判断买入的条件:
            if is_buying_accept:
                # 基于单笔最大交易限制的买入策略
                # logging.warning(f'acct cash list ----------> {self.fund_env.acct_info["cash_asset"]}')
                cash_asset = sum(self.fund_env.acct_info['cash_asset'])
                available_cash = min(cash_asset, self.fund_env.per_buy_order_max_amt)
                available_shares = available_cash

                # 计算可买入的最多股票数量（基于单笔交易金额限制的）
                if available_shares > 0:
                    # 控仓技术: 加到策略的仓位线，最大可补的仓位
                    pfo_ratio_adj = pfo_ratio_guideline - pfo_ratio - action / self.fund_env.initial_amount
                    # 仓位补偿
                    if is_reverse_point:
                        # 在反转/反弹点处，一次加满仓位
                        pfo_ratio_room = int(self.fund_env.initial_amount * (pfo_ratio_guideline - pfo_ratio))
                        # 特殊：取账户现金和建议买入仓位的最小
                        # 由反转预测模型可以知道，在底部预测的点比较稠密，通过分批买入可以买到更低的点，同时减少预测错误的风险成本，一举多得！！！
                        buy_num_shares = int(min(cash_asset, pfo_ratio_room / 3))

                        if self.fund_env.verbose == 1:
                            logging.warning(f'''
                                -------->
                                到达反弹或反转的底部位置，加满策略仓位线 !
                                当前仓位: {pfo_ratio}, 指导线: {pfo_ratio_guideline}, 加仓金额: {buy_num_shares}
                                ''')

                    elif pfo_ratio_adj > 0 and not is_reverse_point:
                        # 补偿动态的仓位百分位缺口
                        max_plus_amount = min(self.fund_env.initial_amount * pfo_ratio_adj, plus_buy_amount)
                        buy_num_shares = min(available_shares, action + max_plus_amount)

                    # pfo_ratio_adj < 0, 仓位压缩，表示加仓会超仓位线，将仓位压缩到仓位线
                    else:
                        action_adj = self.fund_env.initial_amount * (pfo_ratio_guideline - pfo_ratio)
                        buy_num_shares = min(available_shares, action_adj)

                    if buy_num_shares >= self.fund_env.per_unit_amount:
                        buy_amount = buy_num_shares * (1 - self.fund_env.buy_cost_pct[index])

            # 返回买入的份额数量
            return buy_num_shares, buy_amount

        buy_num_shares, buy_amount = _do_buy()
        return buy_num_shares, buy_amount
