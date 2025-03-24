

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
        计算给定一个数 target 计算在已知序列 seq 中的分位数，分数制
    '''
    sorted_data = sorted(seq)
    length = len(sorted_data)
    count = 0
    for num in sorted_data:
        if num <= target:
            count = count + 1
    return round(count / length, 2)
