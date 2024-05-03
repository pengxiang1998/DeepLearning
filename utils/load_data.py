from pandas import read_csv, concat
import matplotlib.pyplot as plt
from pandas import DataFrame

# 转换成监督数据，四列数据，3->1，三组预测一组
def series_to_supervised(data, n_in, n_out, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    # 将3组输入数据依次向下移动3，2，1行，将数据加入cols列表（技巧：(n_in, 0, -1)中的-1指倒序循环，步长为1）
    for i in range(n_in, 0, -1):
    	cols.append(df.shift(i))
    	names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    # 将一组输出数据加入cols列表（技巧：其中i=0）
    for i in range(0, n_out):
    	cols.append(df.shift(-i))
    	if i == 0:
    		names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    	else:
    		names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # cols列表(list)中现在有四块经过下移后的数据(即：df(-3),df(-2),df(-1),df)，将四块数据按列 并排合并
    agg = concat(cols, axis=1)
    # 给合并后的数据添加列名
    agg.columns = names
    print(agg)
    # 删除NaN值列
    if dropnan:
    	agg.dropna(inplace=True)
    return agg


def split_data(observed_data, split_ratio=(6, 2, 2)):
    total = split_ratio[0] + split_ratio[1] + split_ratio[2]
    length = len(observed_data)
    train_cnt = int((split_ratio[0] / total) * length)
    test_cnt = int((split_ratio[2] / total) * length)
    return observed_data[:train_cnt], observed_data[train_cnt:-test_cnt], observed_data[-test_cnt:]


def view_show():
    data = read_csv('../datasets/Convid_19_american.csv', header=0, index_col=0)
    data=data.diff()
    print(data)
    # 纯随机性检验：LB检验，看是否是白噪声，白噪声无规律没法用
    from statsmodels.stats.diagnostic import acorr_ljungbox
    df = acorr_ljungbox(data['new_confirmed_count'], lags=[5, 10, 15, 20], return_df=True)
    print(df)
    #纯随机性检验：Q检验
    df = acorr_ljungbox(data['new_confirmed_count'], lags=[5, 10, 15, 20], boxpierce=True, return_df=True)
    print(df)
    #稳定性检验：自相关图检验
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(data['new_confirmed_count'], lags=25)
    plt.show()


    length=len(data.columns.tolist())#获取表头列表
    rows=int(length/2)
    plt.rcParams['font.family'] = 'SimHei'
    for i in range(length-1):
        plt.subplot(rows, 2, i+1)
        plt.title(data.columns.tolist()[i])
        plt.plot([i for i in range(1, 144)], data[data.columns.tolist()[i]])
    plt.suptitle('疫情统计', fontsize=20, color='red', backgroundcolor='yellow')
    plt.tight_layout(rect=(0, 0, 1, 1))  # 使子图标题和全局标题与坐标轴不重叠
    plt.show()



