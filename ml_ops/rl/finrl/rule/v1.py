import logging
import numpy as np


class FundTradeRules_V1():
    def __init__(self, fund_env) -> None:
        self.fund_env = fund_env


    def _check_reverse_point(self, index):
        '''
        Desc:
            记录最新的反弹 / 反转点的位置
        '''
        is_reverse_point = self.fund_env.current_data['is_reverse_point'].tolist()[index] == 1
        if is_reverse_point:
            self.fund_env.reverse_point_day = self.day


    def buying_signal(self, index):
        '''
        Desc:
            判断买入的市场条件
        '''
        self._check_reverse_point(index)

        _mark_point = self.fund_env.current_data.y_point.tolist()[index]
        _pred_points = self.fund_env.current_data.y_pred.tolist()[index]

        return any([
            # 1. 下跌时，抄底
            _mark_point <= _pred_points * (1 - self.fund_env.temperature) and _mark_point < 0,
            # 2. 上涨时，追涨
            _mark_point <= _pred_points * self.fund_env.temperature and _mark_point > 0,
            # 3. 反弹 / 反转的地步也可以买入
            self.fund_env.current_data['is_reverse_point'].tolist()[index] == 1
            ])

    def selling_signal(self, index):
        '''
        Desc:
            判断卖出的市场条件
        '''
        self._check_reverse_point(index)

        _mark_point = self.fund_env.current_data.y_point.tolist()[index]
        _pred_points = self.fund_env.current_data.y_pred.tolist()[index]

        # 判断卖出的条件: 刚好与买入相反
        return any([
            # 1. 上涨时，止盈
            _mark_point >= _pred_points * self.fund_env.temperature and _mark_point > 0,
            # 2. 下跌时，杀跌
            _mark_point >= _pred_points * (1 - self.fund_env.temperature) and _mark_point < 0,
            # 3. 反弹、反转 3 天内不卖出
            self.day - self.fund_env.reverse_point_day >= 3,
            ])

    def buy_rule(self):
        pass

    def sell_rule(self):
        pass

    def transform_action(self, actions):
        '''
        Desc:
            将 actions value 序列转变为 actions index 索引
        Args:
            actions: 买卖的份额 value 的序列
        '''
        # action 就是股票交易的份额，包含每一支股票对应买卖份额的数组。其中，正为买入，负为卖出，0 为持有
        argsort_actions = np.argsort(actions)
        # 获取卖出的清单
        # .shape[0] 取交易卖出的股票数量, 因为 actions 本身只有一维， 即表示持仓股票中每个股票的加减仓数量
        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
        # 获取买入的清单
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]
        return buy_index, sell_index


    def apply_trade_rules(self, actions):
        '''
        Desc:
            应用自定义的交易规则
        Args:
            actions: 买卖的份额 value 的序列
        '''
        transformed_actions = []
        for index, action in enumerate(actions):
            if action > 0:
                _, trans_action = self.buy_rule(index, action)
            elif action < 0:
                _, trans_action = self.sell_rule(index, action)
            else:
                trans_action = 0
            transformed_actions.append(trans_action)
        return transformed_actions
