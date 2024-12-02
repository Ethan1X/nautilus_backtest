# -*- coding: utf-8 -*-
"""
Depth校验
author: lifangyu
主要内容：
以新数据为基准，校验旧数据的depth是否正确
1. 从新数据和旧数据还原n档depth，找出每个tp中有问题的档数（价格&量），画图并统计分布
2. 从新数据和旧数据还原ticker，按照tp排序融合到一起后，每个tp和【之前最近的&新旧不同的】tp数据做price diff，并画图看重合度

**注意：
1. 新数据（depths里面的数据）可用时间段：
    2022070200-2022072002：depths里的文件命名为utc时间，其他没有问题可用来校验
    2022072011之后：新数据无其他问题
2. 旧数据（deeptradingbooks）可用时间段：
    到2022.09.18：可用
    2022.09.18-2023.1.4：文件命名为utc时间
"""

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from util.recover_depth import *    # 恢复全量depth


def process_ticker(d1, d2):
    """
    校验恢复的depth的盘口数据：按时间戳排序后，画出每个tp的price diff方便查看重合度（只看价格是否相等）
    d1: 恢复的旧数据depth的第一档 {['bids': [], 'asks': [], 'time': int]}
    d2: 恢复的新数据depth的第一档
    """

    df1 = pd.DataFrame(d1)
    df2 = pd.DataFrame(d2)
    df1['type'] = 0   # old
    df2['type'] = 1   # new

    # process data
    df = pd.concat([df1, df2], ignore_index=True)
    df['tp'] = df['time']
    df['time'] = pd.to_datetime(df['tp'], unit='ms')
    # Convert dict data to 4 columns
    df['bids_p'] = df['bids'].apply(lambda x: x[0]['p'] if x else np.nan)
    df['asks_p'] = df['asks'].apply(lambda x: x[0]['p'] if x else np.nan)
    df['bids_s'] = df['bids'].apply(lambda x: x[0]['s'] if x else np.nan)
    df['asks_s'] = df['asks'].apply(lambda x: x[0]['s'] if x else np.nan)
    df.drop(columns=['bids', 'asks'], inplace=True, axis=1)
    # 统一数据起点&终点在同时有新旧数据的时候，避免一段时间内全是同种数据，没有比较意义
    df.sort_values(by='tp', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    start = max(df[df['type'] == 0].index[0], df[df['type'] == 1].index[0])
    end = min(df[df['type'] == 0].index[-1], df[df['type'] == 1].index[-1])
    df = df.iloc[start-1:end+1]
    df.reset_index(drop=True, inplace=True)

    # 找到每行数据之前【最近的】&【不同type的】数据对应index
    gp_label = (df['type'] != df['type'].shift()).cumsum()
    df['pre_idx'] = df.groupby(gp_label)['type'].transform('idxmax')-1   # 找到重复序列的第一个idx (idxmax/idxmin都是一样的)
    # 将对应的数据merge到每行的右边
    df = df.merge(df[['bids_p', 'asks_p', 'bids_s', 'asks_s', 'tp']],
                  left_on='pre_idx', right_index=True, suffixes=('', '_pre'))

    # 计算price diff
    df['bids_diff'] = df['bids_p'] - df['bids_p_pre']
    df['asks_diff'] = df['asks_p'] - df['asks_p_pre']

    return df


def plot_ticker(df, title='ticker_diff'):
    """
    将process_ticker得到的df，（可在经过筛选后）画出price diff的图
    """
    data = [
        go.Scatter(x=df['time'],
                   y=df['bids_diff'],
                   line=dict(color='blue', width=1),
                   name='bids',
                   text='curr_p:' + df['bids_p'].astype(str) + '<br>pre_p:' + df['bids_p_pre'].astype(
                       str) + '<br>curr_s:' + df['bids_s'].astype(str) + '<br>pre_s:' + df['bids_s_pre'].astype(str),
                   yaxis='y1'),
        go.Scatter(x=df['time'],
                   y=df['asks_diff'],
                   line=dict(color='red', width=1),
                   name='asks',
                   text='curr_p:' + df['asks_p'].astype(str) + '<br>pre_p:' + df['asks_p_pre'].astype(
                       str) + '<br>curr_s:' + df['asks_s'].astype(str) + '<br>pre_s:' + df['asks_s_pre'].astype(str),
                   yaxis='y1'),
    ]

    fig = go.Figure(data=data)
    fig.write_html(f'{title}.html', auto_open=True)


def compare_ps(old_one, new_one):
    """
    将旧数据和新数据进行比较，校验某个tp有问题的档数
        1. 在新数据里遍历，check该价格是否存在于旧数据中，如果不存在，wrong_p+1
        2. 如果价格存在，check size是否相等，如果不等，wrong_s+1
    old_one: 恢复的旧数据depth的一个tp的前n档 ['bids': [], 'asks': [], 'time': int]
    new_one: 恢复的新数据depth的一个tp的前n档
    """

    wrong_p, wrong_s = 0, 0
    for order_type in ['bids', 'asks']:
        # Set查找=O(1), dict查找=O(1)
        old_p_set = {i['p'] for i in old_one[order_type]}
        old_s_dict = {i['p']: i['s'] for i in old_one[order_type]}
        new_p_set = {i['p'] for i in new_one[order_type]}
        new_s_dict = {i['p']: i['s'] for i in new_one[order_type]}

        for p in new_p_set:
            if p not in old_p_set:
                wrong_p += 1
            elif new_s_dict[p] != old_s_dict[p]:
                wrong_s += 1

    return wrong_p, wrong_s


def compare_depth(d1, d2, plot=False):
    """
    将旧数据和新数据进行比较，统计每个tp有问题的档数；
    因为tp对不上，根据旧数据的tp，找到新数据里离他最近的两个tp（一前一后）分别做统计，返回错误数最少的那个 （可尝试pd.merge_asof）
    d1: 恢复的旧数据depth {tp: {'bids': [], 'asks': [], 'time': int}}
    d2: 恢复的新数据depth
    """

    result = {}
    d2_idx = 0
    start_tp = max(d1[0]['time'], d2[0]['time'])  # 统一起始时间
    for d1_one in d1:
        d1_tp = d1_one['time']
        if d1_tp < start_tp:
            result[d1_tp] = [np.nan, np.nan, np.nan]   # 保留nan，index能对应上，方便查找
            continue
        for i in range(d2_idx, len(d2)):
            if d2[i]['time'] > d1_tp:
                d2_idx = i-1    # 找最近的tp对应的idx
                break

        wrong_p1, wrong_s1 = compare_ps(d1_one, d2[d2_idx])
        wrong_p2, wrong_s2 = compare_ps(d1_one, d2[d2_idx+1])

        if wrong_p1 + wrong_s1 <= wrong_p2 + wrong_s2:
            result[d1_tp] = [wrong_p1, wrong_s1, d2_idx]
        else:
            result[d1_tp] = [wrong_p2, wrong_s2, d2_idx+1]

    df = pd.DataFrame(result, index=['wrong_p', 'wrong_s', 'new_idx']).T
    df = df.reset_index().rename(
        columns={'index': 'old_tp'})   # reset之后的index就是old_idx
    df['wrong_total'] = df['wrong_p'] + df['wrong_s']
    df['time'] = pd.to_datetime(df['old_tp'], unit='ms')
    
    if plot:
        data = go.Scatter(x=df['time'], y=df['wrong_total'], mode='markers')
        fig = go.Figure(data=data)
        fig.write_html('wrong_total_depth.html', auto_open=True)

    return df



if __name__ == '__main__':
    n = 1
    exchange = 'binance'
    symbol = 'eth_usdt'
    d1 = recoveryDepth(exchange, symbol, '2022071623', n)  # 注意时间是否都是UTC或者东八区
    d2 = recoveryDepth(exchange, symbol, '2022071615', n)
    
    # 校验ticker
    df = process_ticker(d1, d2)
    plot_ticker(df)
    df['tp_diff'] = df['tp'] - df['tp_pre']
    plot_ticker(df[df['tp_diff'] < 10], 'ticker_diff_10ms')
    
    # 统计错误档数并统计分位数
    df = compare_depth(d1, d2)
    df['wrong_total'].describe(percentiles=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.99])
    df['wrong_p'].describe(percentiles=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.99])