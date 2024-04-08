# %%
from gymnasium import spaces
import logging

import sys
from pathlib import Path

from ray.rllib.env import EnvContext

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
        更新如下:
        1. 继续修改 selling 策略，当仓位策略提示应减仓时，无论是否盈利，应卖出部分持仓, 降低仓位
        2. 更新买入策略，当仓位策略提示加仓时，加满到仓位线指导线
        '''
        super().__init__(config)
        self.action_space = spaces.MultiDiscrete([6] * self.stock_dim)
        # 记录仓位控制红线
        self.pfo_ratio_guide = {}


    def _set_pfo_ratio(self):
        '''
        Desc: Importan!
            账户的仓位控制策略。基于上证指数的相对牛熊点位判断
            主要更新点:
            1. 添加仓位记录
        '''
        ratio_strategy = {
            0: 0.8,
            1: 0.5,
            2: 0.3,
            }
        sz_point_phase = self.current_data['sz_closed_phase'].unique()[0]
        idx_pint_phase = self.current_data['closed_phase'].unique()[0]
        pfo_ratio_guideline = (ratio_strategy[sz_point_phase] + ratio_strategy[idx_pint_phase]) / 2
        # 记录仓位
        self.pfo_ratio_guide[self.day] = pfo_ratio_guideline
        return pfo_ratio_guideline


    def _sell_stock(self, index, action):
        '''
        Desc:
            主动卖出超过仓位红线的多余仓位
        '''
        stock_name = self.current_data['tic'].to_list()[index]
        close_price = self.current_data['close'].to_list()[index]
        # 当前的剩余累计持仓
        # cash_asset = sum(self.acct_info['cash_asset'])
        stock_shares, _ = self._get_acct_pfo_shares()
        # logging.warning(f'当前账户持仓 ---------------> 现金: {cash_asset}, 份额: {stock_shares}')

        # 1. 当前可卖出的最大盈利持仓
        max_profit_shares = self._get_max_yield_shares(stock_name, min_yield=self.min_yield)
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

            last_day = max(self.day-1, 0)
            over_pfo_ratio = 0
            pfo_ratio_guide = self._set_pfo_ratio()
            pfo_ratio_act = self._get_pfo_ratio()
            last_pfo_ratio_guide = self.pfo_ratio_guide[last_day]

            if self.pfo_ratio_guide:
                # 方案一: 实际仓位超过仓位策略，即减仓
                # over_pfo_ratio = round(pfo_ratio_act - pfo_ratio_guide, 3)
                # 方案二: 仓位策略线下移，才减仓
                over_pfo_ratio = round(last_pfo_ratio_guide - pfo_ratio_guide, 3)
                logging.warning(f'''
                    -------->
                    当前建议仓位: {pfo_ratio_guide}, 上次建议仓位: {last_pfo_ratio_guide}
                    当前实际仓位: {pfo_ratio_act}, 当前超建议仓位: {over_pfo_ratio}
                    ''')

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
                    sell_num_shares = max_profit_shares

                    if sell_num_shares > 0:
                        # 计算卖出可获得的金额，已经考虑交易费用
                        sell_amount = sell_num_shares

                # 只在杀跌模式下卖出仓位
                if _mark_point < 0 and abs(_mark_point) < abs(_pred_points) * (1 - self.temperature):
                    # 在仓位策略转换的时候，如果当前的仓位策略提示减仓，主动降低仓位
                    decrease_pfo_amount = 0
                    if over_pfo_ratio > 0:
                        logging.warning(f'当前仓位过高，主动减仓 -------> {over_pfo_ratio}')
                        decrease_pfo_amount = round(self.initial_amount * over_pfo_ratio, 2)

                # 优先取策略空间的 sell_amount
                logging.warning(f'---------> sell_amount vs decrease_pfo_amount: {sell_amount} vs {decrease_pfo_amount}')
                sell_amount = sell_amount if sell_amount > 0 else decrease_pfo_amount
                # 卖出股票，仓位减少
                self.acct_info['pfo_holding'][stock_name].append(-sell_num_shares)
                self.acct_info['pfo_price'][stock_name].append(close_price)
                return_ratio = self._caculate_selling_return(stock_name, sell_amount, mode='LiveTrade')
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
        # 更新仓位控制线
        pfo_ratio_guideline = self._set_pfo_ratio()
        pfo_ratio = self._get_pfo_ratio()

        last_day = max(self.day-1, 0)
        plus_pfo_ratio = 0
        plus_buy_amount = 0
        if self.pfo_ratio_guide:
            last_pfo_ratio_guide = self.pfo_ratio_guide[last_day]
            plus_pfo_ratio = last_pfo_ratio_guide - pfo_ratio_guideline
            plus_buy_amount = round(self.initial_amount * plus_pfo_ratio, 1)

        if pfo_ratio > pfo_ratio_guideline:
            logging.warning(f'-------> 当前仓位: {pfo_ratio}, 已达到仓位控制线 {pfo_ratio_guideline}, 暂停加仓 !!!')

            self.stop_buying += 1
            if self.stop_buying >= self.early_stop_times:
                self.truncate = True
                logging.warning(f'meet stop buying times -----------> {self.stop_buying}')

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
                # 最大可补的仓位
                pfo_amount = round(self.initial_amount * (pfo_ratio_guideline - pfo_ratio), 1)
                available_cash = min(cash_asset, self.per_buy_order_max_amt, pfo_amount)

                # 注意：与股票不同，基金直接使用买卖金额，模型输出金额后再换算份额
                available_shares = max(available_cash, plus_buy_amount)

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

# %%