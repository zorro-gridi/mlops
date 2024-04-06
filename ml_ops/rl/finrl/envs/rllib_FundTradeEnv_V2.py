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
            继承父类的 step 方法
        '''
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

        for i in range(self.stock_dim):
            _, _ = self._sell_stock(i, 0)
        return super().step(actions)


    def _sell_stock(self, index, action):
        '''
        Desc:
            更新的卖出策略，即不从策略空间中探索，而是根据设定的规则
        '''
        action = 0
        return super()._sell_stock(index, action)