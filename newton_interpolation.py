import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from pylab import mpl
# import math  # used in Pycharm
# # %matplotlib inline  # used for jupyter notebook

def _get_diff_table(X, Y):
    """
    得到插商表
    """
    n = len(X)
    A = np.zeros([n, n])

    for i in range(0, n):
        A[i][0] = Y[i]

    for j in range(1, n):
        for i in range(j, n):
            A[i][j] = (A[i][j - 1] - A[i - 1][j - 1]) / (X[i] - X[i - j])

    return A


def _newton_interpolation(X,Y,x):
    """
    计算x点的插值
    """
    sum=Y[0]
    temp=np.zeros((len(X),len(X)))
    #将第一行赋值
    for i in range(0,len(X)):
        temp[i,0]=Y[i]
    temp_sum=1.0
    for i in range(1,len(X)):
        #x的多项式
        temp_sum=temp_sum*(x-X[i-1])
        #计算均差
        for j in range(i,len(X)):
            temp[j,i]=(temp[j,i-1]-temp[j-1,i-1])/(X[j]-X[j-i])
        sum+=temp_sum*temp[i,i]
    return sum


def _get_newton_value(T, SF, num):
    sfa = _get_diff_table(T, SF)
    df = pd.DataFrame(sfa)
    # print(df)
    xsf = np.linspace(np.min(T), np.max(T), num, endpoint=True)
    ysf = []
    for x in xsf:
        ysf.append(_newton_interpolation(T, SF, x))
    # print(ysf)
    return ysf


def main():
    data_60 = '/Users/sissi/PycharmProjects/test/data/CLINKER 6O IBM.xlsx'
    df_data_60 = pd.read_excel(data_60)

    # 把时间对齐
    df_data_60['Sample Date'] = df_data_60['Sample Date'].dt.floor('1T')
    df_data_60 = df_data_60.dropna(axis=0, how='any')
    df_data_60.index = pd.to_datetime(df_data_60['Sample Date'])
    columns_clean = df_data_60.describe().dropna(how='any', axis=1).columns
    # print(columns_clean)
    df_data_60_clean = df_data_60[columns_clean].dropna(axis=0, how='any')
    # print(df_data_60_clean)
    print(df_data_60_clean.index)

    # 获得文件中每行的时间间隔
    prev_time = df_data_60_clean.index[0]
    # print(prev_time)
    gaps = []
    for time in df_data_60_clean.index[1:]:
        gap = pd.Timedelta(time - prev_time).seconds / 3600.0 + pd.Timedelta(time - prev_time).days * 24
        # gap = time - prev_time
        prev_time = time
        # 对时间进行四舍五入以取整
        gaps.append(int("{:.0f}".format(gap)))
    # print(gaps)
    # print(len(gaps))

    # # 插入数据
    sample_times = []
    bds, lsfs, mss, mas, c3ss, frees, ss = [], [], [], [], [], [], []
    start_time = df_data_60_clean.index[0]
    for i in range(len(gaps)):
        # sample time
        # print(i, ': start_time = ', start_time, 'gap =', gap)
        t = pd.date_range(start=start_time, periods=12 * gaps[i] + 1, freq='5T')
        start_time = t[-1]
        sample_times.extend(t)

        x = [i, i + 1]
        # 列
        bds.extend(_get_newton_value(x, df_data_60_clean['No. BD'][i:i + 2], 12 * gaps[i] + 1))
        lsfs.extend(_get_newton_value(x, df_data_60_clean['LSF Clinker'][i:i + 2], 12 * gaps[i] + 1))
        mss.extend(_get_newton_value(x, df_data_60_clean['MS'][i:i + 2], 12 * gaps[i] + 1))
        mas.extend(_get_newton_value(x, df_data_60_clean['MA'][i:i + 2], 12 * gaps[i] + 1))
        c3ss.extend(_get_newton_value(x, df_data_60_clean['C3S (%)'][i:i + 2], 12 * gaps[i] + 1))
        frees.extend(_get_newton_value(x, df_data_60_clean['Free lime (%)'][i:i + 2], 12 * gaps[i] + 1))
        ss.extend(_get_newton_value(x, df_data_60_clean['S/ALK_Cl'][i:i + 2], 12 * gaps[i] + 1))

    print(len(sample_times), len(bds), len(lsfs), len(mss), len(mas), len(c3ss), len(frees), len(ss))

    new_dict = {'Sample Date': sample_times, 'No. BD': bds, 'LSF Clinker': lsfs, 'MS': mss, 'MA': mas,
                'C3S (%)': c3ss, 'Free lime (%)': frees, 'S/ALK_Cl': ss}
    new_dict_df = pd.DataFrame.from_dict(new_dict)
    print(len(new_dict_df.drop_duplicates()))
    # print(new_dict_df)
    # print(new_dict_df.drop_duplicates())

    # 写入文件
    wf = pd.DataFrame(new_dict_df.drop_duplicates())
    wf.to_excel('/Users/sissi/PycharmProjects/test/data/CLINKER 6O IBM 5T.xlsx', index=True)


if __name__ == "__main__":
    main()