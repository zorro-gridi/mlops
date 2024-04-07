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

from mlops.ml_ops.rl.finrl.envs.rllib_FundTradeEnv import FundQuantTradeEnv



class FundQuantTradeEnv_V2(FundQuantTradeEnv):
    '''
    FundTrader Env 第二版
    '''
    def __init__(self, config: EnvContext):
        '''
        更新如下:
        1. 修改策略空间，只对买入策略进行训练，卖出策略遵守人选的规则
        '''
        super().__init__(config)
        self.action_space = spaces.MultiDiscrete([6] * self.stock_dim)


    def step(self, actions):
        '''
        Desc:
            继承并更新父类的 step 方法, 主要改进如下：
            1. 取消 actions 的卖出策略分布，改为如下规则：
            1.1 按照用户自定义策略卖出，即清仓是否达到预期收益率
            1.2 是否有达到预期盈利的持仓可卖出
        '''
        # acct_holdings = self.acct_info['pfo_shares_redeem']
        # logging.warning(f'acct holdings ------------> {acct_holdings}')
        # logging.warning(f'strategy actions test ---------> {actions}')

        # 因为, v1版step要给 actions 减 5，因此，需要提前加回来
        self.state, self.reward, self.terminal, self.truncate, self.acct_info = super().step(actions+5)
        if not (self.terminal or self.truncate):
            for i in range(self.stock_dim):
                _, _ = self._sell_stock(i, 0)
        return self.state, self.reward, self.terminal, self.truncate, self.acct_info


    def _sell_stock(self, index, action):
        '''
        Desc:
            更新的卖出策略，即不从策略空间中探索，而是根据设定的规则
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
                    sell_num_shares = max_profit_shares

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
