
import numpy as np


def transform_list(input_list):
    '''
    Desc:
        将相同的编号记录分成一组, 且序号递增
    '''
    if not input_list:
        return []

    output_list = []
    group_number = 1

    for i in range(len(input_list)):
        output_list.append(group_number)
        if i < len(input_list) - 1 and input_list[i] != input_list[i + 1]:
            group_number += 1

    return output_list


def calculate_quantile(target, seq):
    '''
    Desc:
        计算给定一个数 target, 计算其在已知序列 seq 中的分位数（分数制）
    '''
    max_value = max(np.abs(seq))
    return round(target / max_value, 2)
