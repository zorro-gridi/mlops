import pandas as pd
import numpy as np


def chunk_series(datas: pd.DataFrame, checkpoint=0.1):
    '''
    Desc:
        将序列按照涨跌分组切块, data 中必须包含 "markup" 涨跌幅指标
    Args:
        data: 需要切分的序列
        checkpoint: 合并的阈值。abs(x) <= checkpoint 则合并到上一个分块序列
    Returns:
        pd.DataFrame
    '''
    data = datas.copy()
    markup_list = list(data['markup'])
    for i in range(1, len(markup_list)):
        last_sign = np.sign(markup_list[i-1])
        if abs(markup_list[i]) <= checkpoint:
            markup_list[i] = abs(markup_list[i]) * last_sign
    # markup 字段重新赋值
    data['markup'] = markup_list
    data['last_markup'] = data.sort_values(by=['trade_date'])['markup'].shift(1)

    data['is_split_point'] = [
        i if np.sign(x) != np.sign(y) else -1
        for i, (x, y) in enumerate(zip(data['markup'], data['last_markup']))]

    markup_list = list(data['markup'])
    split_idx_list = [i for i in data['is_split_point'] if i != -1]
    split_idx_list.append(len(data))

    # 涨跌分快序列
    global markup_chunk_list
    markup_chunk_list = [markup_list[split_idx_list[i]:split_idx_list[i+1]] for i in range(len(split_idx_list)-1)]

    # 将涨跌分快序列 markup_chunk_list 统计成连续涨跌天数
    # markup_vol: 涨跌幅切分序列
    # markup_days: 涨跌天数统计序列
    vars_data = dict(
        markup_days=[len(d) if sum(d) > 0 else -len(d) for d in markup_chunk_list],
        markup_vol=[sum(d) for d in markup_chunk_list],
        )
    vars_data = pd.DataFrame(vars_data)
    return vars_data



def compute_vars_list(series, chunk_index: list, pool_func='np.max'):
    '''
    Desc:
        本函数实现将外部变量按照分快序列的不同长度，切分外部变量序列, 并针对序列组进行池化
        说明: 因为预测涨跌天数，将序列进行了分段汇总，如果传入外部变量序列，因此也需要将外部变量的序列分型分段汇总（即池化操作）
    Args:
        chunk_index: np.array 在本例中，是涨跌天数的分段汇总序列。数据样本: [1, -2, 4, -5, 3, -3,...]
    Returns:
        返回序列分组、并池化后的外部变量 np.ndarray
    '''
    def outvars_markup_days(seq):
        '''
        Desc:
            计算外部变量分段序列的涨跌天数, 必须返回 float 对象
        '''
        up_days = sum(np.array(seq) > 0)
        return up_days

    def get_last_phase_point(seq):
        '''
        Desc:
            获取分割序列的最后一个牛熊分割线标签。分类标签，可以为 int
        '''
        return str(seq[-1])

    seq_idx = chunk_index.copy()
    # 计算累计涨跌幅
    seq_idx.insert(0, 0)
    # 先将序列正数化，变为实际涨跌天数，然后求和汇总，用于切分其它变量块
    seq_idx = abs(np.array(seq_idx)).cumsum()

    vars_value = list(series)
    # 按 chunk_index（涨跌幅序列）分割外部变量序列
    vars_value = [vars_value[seq_idx[i-1]:seq_idx[i]] for i in range(1, len(seq_idx))]

    # 外部变量池化操作
    pool_func = eval(pool_func)
    vars_value = [pool_func(v) for v in vars_value]
    assert len(chunk_index) == len(vars_value)
    return vars_value


def build_multi_vars_datas(data, chunk_index, vars_config: dict, target_series=None):
    '''
    Desc:
        本函数实现将外部变量集成到预测目标变量中
    Args:
        data: pd.DataFrame
        chunk_index: （本例中）表示指数的连续涨跌幅序列，必选参数
        vars_config: dict, 外部变量的配置字典
            example: {
                'var1': 'np.max',
                'var2': 'np.sum',
                }
        target_series: 预测的目标变量列表，可选 累计涨跌幅 or 累计涨跌幅天数；默认为 chunk_index
    Returns:
        pd.DataFrame 包含各个特征的 dataframe 对象
    '''
    # 设计合理的数据结构非常重要
    var_value_list = {
        var_name: compute_vars_list(data[var_name], chunk_index=chunk_index, pool_func=pool_func)
        for var_name, pool_func in vars_config.items()
        }
    # target_series 为 None 时默认为 chunk_index
    if not target_series:
        target_series = chunk_index

    # 最后，将预测变量添加进来
    var_value_list['target'] = target_series
    var_datas = pd.DataFrame(var_value_list)
    return var_datas
