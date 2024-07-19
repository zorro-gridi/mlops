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



class FundQuantTradeEnv_V5(FundQuantTradeEnv_V2):
    '''
    基于版本二的 FundTrader Env 第五版
    不能继承自版本三。因为版本三针对卖出进行了超仓位主动降仓，可能会导致在反转点买入的仓位，立马遭到卖出
    '''
    def __init__(self, config: EnvContext):
        '''
        Update 更新如下:
            1. 附加：买入时，如果碰到【反弹、反转点】，立即加满到仓位知道线，进行一次性补仓
        Conclusion:
            待实验结论...
        '''
        super().__init__(config)

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
            # 昨日的仓位指导线
            last_pfo_ratio_guide = self.pfo_ratio_guide[last_day]
            # 加仓空间
            plus_pfo_ratio = last_pfo_ratio_guide - pfo_ratio_guideline
            # 可加仓金额
            plus_buy_amount = round(self.initial_amount * plus_pfo_ratio, 1)

        # 如果当前已到仓位指导线，则停止加仓
        if pfo_ratio >= pfo_ratio_guideline:
            if self.verbose == 1:
                logging.warning(f'''
                    ------->
                    trade date: {self._get_date()}
                    指数牛熊位置: {self.current_data['closed_phase'].max()}, 百分位: {self.current_data['closed_phase_percentile'].max()}
                    当前仓位: {pfo_ratio}, 已达到仓位控制线 {pfo_ratio_guideline}, 暂停加仓 !!!
                    ''')
            return 0, 0

        stock_name = self.current_data.tic.to_list()[index]
        is_reverse_point = self.current_data['is_reverse_point'].tolist()[index] == 1
        # 基金使用收盘涨跌幅，收盘价在基金的模拟环境中没有实际使用
        # close_price = self.current_data.close.to_list()[index]

        def _do_buy():
            buy_num_shares = 0
            buy_amount = 0
            opt_type = 3

            # 判断买入的条件:
            if is_buying_accept:
                # 基于单笔最大交易限制的买入策略
                # logging.warning(f'acct cash list ----------> {self.acct_info["cash_asset"]}')
                cash_asset = sum(self.acct_info['cash_asset'].values())
                # 最大加仓金额 & 账户余额 取最小
                available_cash = min(cash_asset, self.per_buy_order_max_amt)
                available_shares = available_cash

                # 计算可买入的最多股票数量（基于单笔交易金额限制的）
                if available_shares > 0:
                    # 控仓技术: 加到策略的仓位线，最大可补的仓位
                    pfo_ratio_adj = pfo_ratio_guideline - pfo_ratio - action / self.initial_amount
                    # 仓位补偿
                    if is_reverse_point:
                        # 更新操作类型为抄底
                        opt_type = 2
                        # 在反转/反弹点处，一次加满仓位
                        pfo_ratio_room = int(self.initial_amount * (pfo_ratio_guideline - pfo_ratio))
                        # 特殊：取账户现金和建议买入仓位的最小
                        # 由反转预测模型可以知道，在底部预测的点比较稠密，通过分批买入可以买到更低的点，同时减少预测错误的风险成本，一举多得！！！
                        buy_num_shares = int(min(available_shares, pfo_ratio_room / 3))

                        if self.verbose == 1:
                            logging.warning(f'''
                                -------->
                                当前交易日期: {self._get_date()}, 到达反弹、或反转的底部位置，加满策略仓位线 !
                                当前仓位: {pfo_ratio}, 指导线: {pfo_ratio_guideline}
                                账户余额: {cash_asset:0.2f}, 加仓空间: {pfo_ratio_room:0.2f}
                                可加仓金额: {available_shares}, 实际加仓加仓金额: {buy_num_shares:0.2f}
                                ''')

                    elif pfo_ratio_adj > 0 and not is_reverse_point:
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
                            # 更新账户的可用本金
                            # 买入股票，现金账户减少金额
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
                            'opt_type': opt_type,
                            })

            # 返回买入的份额数量
            return buy_num_shares, buy_amount

        buy_num_shares, buy_amount = _do_buy()
        return buy_num_shares, buy_amount


# %%