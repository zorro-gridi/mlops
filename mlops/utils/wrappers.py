from copy import copy


class DictWrapper():
    '''
    desc: 本class实现获取指定类的 class.__dict__ 参数化访问的功能
    '''
    def __init__(self, dictionary):
        self._dict = copy(dictionary)

    def __getattr__(self, key: str):
        return self._dict[key]

    # 不能设置 __setattr__ 方法，因为 __dict__ 本身是自身的属性，不能递归修改
    # 否则：RecursionError: maximum recursion depth exceeded

    # 或者使用 copy 创建 class.__dict__ 的副本
    # def __setattr__(self, key, value):
    #     self._dict[key] = value
