import logging


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