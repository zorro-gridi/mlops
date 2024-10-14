
# %%
import pandas as pd
import matplotlib.pyplot as plt

# 假设数据保存在名为 df 的 DataFrame 中
df = pd.DataFrame({
    'buy_date': ['2024-03-08', '2024-01-02', '2024-03-06', '2024-03-05', '2024-03-14'],
    'selling_date': ['2024-03-20', '2024-03-20', '2024-03-20', '2024-03-20', '2024-03-20'],
    'shares': [3994.0, 3994.0, 3994.0, 3994.0, 3994.0],
    'hold': [0.0, 0.0, 0.0, 0.0, 0.0],
    'yield': [0.0923, 0.0796, 0.0780, 0.0672, 0.0630],
    'soldout': [1, 1, 1, 1, 1],
    'tic': ['传媒', '传媒', '传媒', '传媒', '传媒']
    })

# 将日期列转换为日期类型
df['buy_date'] = pd.to_datetime(df['buy_date'])
df['selling_date'] = pd.to_datetime(df['selling_date'])

# 创建一个图形和一个坐标轴对象
fig, ax = plt.subplots()

# 根据 buy_date 绘制折线图
ax.plot(df['buy_date'], df['shares'], label='Buy Date')
# 根据 selling_date 绘制折线图
ax.plot(df['selling_date'], df['shares'], label='Selling Date')

# 设置 x 轴标签
ax.set_xlabel('Date')

# 设置 y 轴标签
ax.set_ylabel('Shares')

# 添加图例
ax.legend()

# 自动调整日期标签的格式
fig.autofmt_xdate()

# 显示图形
plt.show()