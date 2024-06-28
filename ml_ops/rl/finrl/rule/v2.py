
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dir_list = [dname for dname in current_dir.as_posix().split('/')]
home_dir = '/'.join([dirname for dirname in dir_list[:dir_list.index('zorro')+1]])
env_path = '/'.join([dirname for dirname in dir_list[:dir_list.index('pycharm')+1]])
sys.path.append(env_path)

from mlops.ml_ops.rl.finrl.rule.v1 import FundTradeRules_V1


class FundTradeRules_V2(FundTradeRules_V1):
    def __init__(self, fund_env) -> None:
        self.fund_env = fund_env

    def sell_rule(self, index, action=0):
        '''
        Desc:
            经过用户自定义规则转换的交易量
        Args:
            index: 交易的目标基金的索引
        Return:
            返回用户自定义规则的卖出数量
        '''
        stock_name = self.fund_env.current_data['tic'].to_list()[index]
        # 当前的剩余累计持仓
        # cash_asset = sum(self.fund_env.acct_info['cash_asset'])
        stock_shares, _ = self.fund_env._get_acct_pfo_shares()
        # logging.warning(f'当前账户持仓 ---------------> 现金: {cash_asset}, 份额: {stock_shares}')

        # 1. 当前可卖出的最大盈利持仓
        max_profit_shares = self.fund_env._cal_max_selling_amount_with_min_yield(stock_name, min_yield=self.fund_env.min_yield)
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

            # 判断卖出的条件: 刚好与买入相反
            if self.fund_env.selling_signal(index):
                # 判断当前是否有该股票的持仓
                if max_profit_shares > 0 and stock_shares > 0:
                    # Sell only if current asset is > 0
                    # 此处与股票不同，注意 ！！！
                    # logging.warning(f'max_profit: {max_profit_shares:0.2f}')
                    sell_num_shares = max_profit_shares
                    # 记录累计已卖出的盈利头寸
                    # logging.warning(f'do sell stock action: {sell_num_shares} quantities.')
                    self.fund_env.acct_info['profit_shares_sold'][stock_name] += sell_num_shares
                    # 计算卖出可获得的金额，考虑交易费用
                    sell_amount = sell_num_shares

            return sell_num_shares, sell_amount

        sell_num_shares, sell_amount = _do_sell_normal()
        return sell_num_shares, sell_amount