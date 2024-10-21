from scipy.stats import normaltest, shapiro, kstest, skew, kurtosis
import logging
from sklearn.neighbors import KernelDensity
import statsmodels.api as sm
import pylab
import numpy as np
import seaborn as sns


def SeqNormDisTest(s, plot=False):
    '''
    Desc:
        对一组序列进行正态性检验，包含 qq-plot, hist-kde plot, W-test, k-s test 等
    Args:
        s: 待检测的序列
        plot: 是否作图
    Return:
        统计量字典
    '''
    # W 检验
    _, p_value = shapiro(s)
    # k-s 检验
    # _, p_value = kstest(s, cdf='norm')

    # 偏度 & 峰度
    skewness = skew(s)
    kurto = kurtosis(s)
    alpha = 0.05

    if round(p_value, 2) >= alpha:
        logging.warning(f'p value: {p_value:.3f}, skew: {skewness:0.3f}, kurto: {kurto:.03f}, 切分序列满足正态分布')

        if plot:
            # Q-Q图
            sm.qqplot(s, line='s')
            pylab.show()
            # dist 分布
            sns.displot(s, kde=True)

        return {
            'skewness': skewness,
            'kurto': kurto,
            'p_value': p_value,
            }
    else:
        logging.warning(f'p value: {p_value:.3f}, skew: {skewness:0.3f}, kurto: {kurto:.03f}, 切分序列不满足正态分布')
        return None