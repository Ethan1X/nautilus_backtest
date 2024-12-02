# 回测结果分析

import pandas as pd
import numpy as np
import datetime
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import tomli
import tomli_w
warnings.filterwarnings('ignore')


def get_order_strategy_type_stat(cache_path: str, order_df, strategy_type_df):
    # 统计信号单、止盈单、止损单的信息
    order_df = pd.merge(order_df, strategy_type_df, on='order_id', how='left')
    
    filled_df = order_df[order_df.filled_amount > 0]
    filled_df = filled_df.rename(columns={'ts': 'filled_ts'})
    open_df = order_df[order_df.status == 'OPENED'][['order_id', 'price', 'ts']]
    open_df = open_df.rename(columns={'price': 'open_price', 'ts': 'open_ts'})
    open_filled_df = pd.merge(filled_df, open_df, on='order_id', how='left')
    open_filled_df['open_filled_period'] = open_filled_df.filled_ts - open_filled_df.open_ts
    
    fig = make_subplots(rows=1, cols=4)
    # 不同类型订单的成交量
    df_ = open_filled_df.groupby('strategy_type')['filled_amount'].sum()
    # print(df_)
    # 不同类型订单的发单成交间隔
    fig.add_trace(go.Bar(x=df_.index, y=df_, name='strategy_type_filled_amount'), row=1, col=1)
    fig.add_trace(go.Histogram(x=open_filled_df[open_filled_df['strategy_type'] == 'signal_open']['open_filled_period'], nbinsx=100, name='signal_open_filled_period'), row=1, col=2)
    fig.add_trace(go.Histogram(x=open_filled_df[open_filled_df['strategy_type'] == 'stop_loss']['open_filled_period'], nbinsx=100, name='stop_loss_filled_period'), row=1, col=3)
    fig.add_trace(go.Histogram(x=open_filled_df[open_filled_df['strategy_type'] == 'stop_win']['open_filled_period'], nbinsx=100, name='stop_win_filled_period'), row=1, col=4)
    
    fig.update_layout(
                    title_text='strategy type info',
                    showlegend=True,
                )
    # fig.show()
    fig.write_html(f"{cache_path}/strategy_type_info.html")

    return open_filled_df

def get_signal_slippage(cache_path: str, order_df, signal_df):
    # 获取信号追单价差
    
    open_df = order_df[order_df.status == 'OPENED']
    filled_df = order_df[order_df.status == 'FILLED']
    signal_df = pd.merge(signal_df, open_df[['order_id', 'price', 'side']], 
                         on='order_id', how='left').rename(columns={'price': 'open_price'})
    signal_df = pd.merge(signal_df, filled_df[['order_id', 'price']], on='order_id', how='left').rename(columns={'price': 'filled_price'})
    signal_df = pd.merge(signal_df, open_df[['order_id', 'price']], left_on='strategy_id', right_on='order_id',
                         how='left').rename(columns={'price': 'signal_price'})
    
    signal_df = signal_df[signal_df.strategy_type == 'signal_open']
    signal_df['slippage'] = (signal_df['signal_price'] - signal_df['filled_price'])/signal_df['signal_price']
    buy_signal_df = signal_df[signal_df['side'] == 'BUY']
    sell_signal_df = signal_df[signal_df['side'] == 'SELL']

    return signal_df

def get_hour_returns_plot(cache_path: str):
    # 获取小时收益率分布图
    hour_returns = pd.read_csv(f'{cache_path}/freq_returns.csv')

    up_returns = hour_returns[hour_returns.iloc[:, 1] > 0]
    down_returns = hour_returns[hour_returns.iloc[:, 1] <= 0]
    
    fig = go.Figure(data=[
        go.Bar(x=up_returns.iloc[:, 0], y=up_returns.iloc[:, 1], name='win')
    ])
    fig.add_trace(
        go.Bar(x=down_returns.iloc[:, 0], y=down_returns.iloc[:, 1], name='loss')
    )
    
    fig.update_layout(
        title='hour returns',
        xaxis_title='time',
        yaxis_title='returns',
    )
    # fig.show()
    fig.write_html(f"{cache_path}/hour_returns.html")

def get_order_event_count(f, trading_days, open_order, cancel_order, filled_order):
    # 获取回测结果的日均挂撤次数
    
    print(f'日平均下单次数:{len(open_order)/trading_days:.2f}', file=f)
    
    print(f'日平均撤单次数:{len(cancel_order)/trading_days:.2f}', file=f)
    
    print(f'日平均成交次数:{len(filled_order)/trading_days:.2f}', file=f)

def get_signal_filled_stat(f, open_order, filled_order, signal_open_order, stop_loss_order):
    # 获取信号成交概率和追单统计
    signal_open_open_order = pd.merge(signal_open_order, open_order, on='order_id', how='inner')
    signal_open_filled_order = pd.merge(signal_open_order, filled_order, on='order_id', how='inner')
    stop_loss_open_order = pd.merge(stop_loss_order, open_order, on='order_id', how='left')
    stop_loss_filled_order = pd.merge(stop_loss_order, filled_order, on='order_id', how='left')
    
    signal_num = len(signal_open_open_order)
    signal_filled_num = len(signal_open_filled_order)
    print(f'信号成交概率:{signal_filled_num/signal_num*100:.2f}%', file=f)
    
    order_df_signal = signal_open_open_order.drop_duplicates(subset=['strategy_id'], keep='first')
    order_df_signal_filled = signal_open_filled_order.drop_duplicates(subset=['strategy_id'], keep='first')
    df_ = pd.merge(order_df_signal[['strategy_id', 'ts']], order_df_signal_filled[['strategy_id', 'ts']], on='strategy_id', how='left')
    df_['catch_period'] = df_.iloc[:, 2] - df_.iloc[:, 1]
    print(f'信号平均追单时长:{df_["catch_period"].mean():.2f}ms', file=f)
    
    order_df_signal = stop_loss_open_order.drop_duplicates(subset=['strategy_id'], keep='first')
    order_df_signal_filled = stop_loss_filled_order.drop_duplicates(subset=['strategy_id'], keep='first')
    df_ = pd.merge(order_df_signal[['strategy_id', 'ts']], order_df_signal_filled[['strategy_id', 'ts']], on='strategy_id', how='left')
    df_['catch_period'] = df_.iloc[:, 2] - df_.iloc[:, 1]
    print(f'止损平均追单时长:{df_["catch_period"].mean():.2f}ms', file=f)


def get_filled_period_stat(f, open_order, filled_order, signal_open_order, stop_loss_order, ts_period=3):
    # 统计订单提交到成交的时间差（3ms内成交的订单占比）
        
    filled_df = filled_order.rename(columns={'ts': 'filled_ts'})
    open_df = open_order.rename(columns={'price': 'open_price', 'ts': 'open_ts'})
    open_filled_df = pd.merge(filled_df, open_df, on='order_id', how='left')
    open_filled_df['open_filled_period'] = open_filled_df.filled_ts - open_filled_df.open_ts
    order_filled_num_3ms = open_filled_df[open_filled_df['open_filled_period'] < ts_period].shape[0]
    order_filled_num = open_filled_df.shape[0]

    signal_open_filled_df =  pd.merge(signal_open_order, open_filled_df, on='order_id', how='inner')
    signal_open_filled_num_3ms = signal_open_filled_df[signal_open_filled_df['open_filled_period'] <= ts_period].shape[0]

    stop_loss_filled_df = pd.merge(stop_loss_order, open_filled_df, on='order_id', how='inner')
    stop_loss_filled_num_3ms = stop_loss_filled_df[stop_loss_filled_df['open_filled_period'] <= ts_period].shape[0]

    print(f'3ms内成交订单数:{order_filled_num_3ms}', file=f)
    print(f'所有成交订单数:{order_filled_num}', file=f)
    print(f'3ms内成交订单数占比:{order_filled_num_3ms/order_filled_num * 100:.2f}%', file=f)
    print(f'信号单3ms内成交订单数占比:{signal_open_filled_num_3ms/signal_open_filled_df.shape[0] * 100:.2f}%', file=f)
    if stop_loss_filled_df.shape[0] != 0:
        print(f'止损单3ms内成交订单数占比:{stop_loss_filled_num_3ms/stop_loss_filled_df.shape[0] * 100:.2f}%', file=f)
    else:
        print(f'止损单3ms内成交订单数占比:0', file=f)

def get_signal_diff_price_stat(f, open_order, filled_order, signal_open_order, stop_loss_order):
    # 回测结果的order存储位置（所有订单信息）

    signal_open_df = pd.merge(signal_open_order, open_order, left_on='strategy_id', right_on='order_id', how='left', suffixes=('', '_open'))
    signal_open_df = pd.merge(signal_open_df, filled_order, on='order_id', how='left', suffixes=('', '_filled'))

    stop_loss_df = pd.merge(stop_loss_order, open_order, left_on='strategy_id', right_on='order_id', how='left', suffixes=('', '_open'))
    stop_loss_df = pd.merge(stop_loss_df, filled_order, on='order_id', how='left', suffixes=('', '_filled'))

    # 所有订单追单次数分布:
    print('信号追单次数分布: ', (signal_open_df['order_id_idx'] - signal_open_df['strategy_id_idx']).astype(int).describe(np.arange(0,1,0.05)), file=f)
    
    # 成交订单追单次数分布:
    strategy_type_df_complete_chase = signal_open_df[~signal_open_df.filled_price.isnull()]
    print('信号成交订单中的追单次数分布: ', (strategy_type_df_complete_chase['order_id_idx'] - strategy_type_df_complete_chase['strategy_id_idx']).astype(int).describe(np.arange(0,1,0.05)), file=f)
    strategy_type_df_complete_chase = stop_loss_df[~stop_loss_df.filled_price.isnull()]
    print('止损成交订单中的追单次数分布: ', (strategy_type_df_complete_chase['order_id_idx'] - strategy_type_df_complete_chase['strategy_id_idx']).astype(int).describe(np.arange(0,1,0.05)), file=f)
    
    # 查看成交订单中，价格与预期下单价格差异分布
    strategy_type_df_filled = signal_open_df[~signal_open_df.filled_price.isnull()]
    
    strategy_type_df_filled.loc[strategy_type_df_filled['side'] == 'BUY', 'diff_price'] = strategy_type_df_filled['open_price'] / strategy_type_df_filled['filled_price'] - 1
    strategy_type_df_filled.loc[strategy_type_df_filled['side'] == 'SELL', 'diff_price'] = strategy_type_df_filled['filled_price'] / strategy_type_df_filled['open_price'] - 1
    
    # print(f"信号开单时，追单带来的亏损分布: \n", strategy_type_df_filled.query('strategy_type == "signal_open"')['diff_price'].describe(np.arange(0,1,0.05)), file=f)
    print(f"信号开单时，追单带来的亏损分布: \n", strategy_type_df_filled['diff_price'].describe(np.arange(0,1,0.05)), file=f)

    strategy_type_df_filled = stop_loss_df[~stop_loss_df.filled_price.isnull()]
    strategy_type_df_filled.loc[strategy_type_df_filled['side'] == 'BUY', 'diff_price'] = strategy_type_df_filled['open_price'] / strategy_type_df_filled['filled_price'] - 1
    strategy_type_df_filled.loc[strategy_type_df_filled['side'] == 'SELL', 'diff_price'] = strategy_type_df_filled['filled_price'] / strategy_type_df_filled['open_price'] - 1
    # print(f"信号止损时，追单带来的亏损分布: \n", strategy_type_df_filled.query('strategy_type == "stop_loss"')['diff_price'].describe(np.arange(0,1,0.05)), file=f)
    # print(f"信号止损时，造成亏损小于1.8bp的总价格损益比例", strategy_type_df_filled.query('strategy_type == "stop_loss" and diff_price < -0.00018')['diff_price'].sum(), file=f)
    # print(f"信号止损时，造成亏损大于1.8bp的总价格损益比例", strategy_type_df_filled.query('strategy_type == "stop_loss" and diff_price > -0.00018')['diff_price'].sum(), file=f)
    print(f"信号止损时，追单带来的亏损分布: \n", strategy_type_df_filled['diff_price'].describe(np.arange(0,1,0.05)), file=f)
    print(f"信号止损时，造成亏损小于1.8bp的总价格损益比例", strategy_type_df_filled.query('diff_price < -0.00018')['diff_price'].sum(), file=f)
    print(f"信号止损时，造成亏损大于1.8bp的总价格损益比例", strategy_type_df_filled.query('diff_price > -0.00018')['diff_price'].sum(), file=f)

def get_stat_analysis(cache_path: str, trading_days: float, ts_period=3):
    order_file = np.load(f'{cache_path}/orders.npz')
    order_df = pd.DataFrame({key: order_file[key] for key in order_file.keys()})
        
    strategy_type_file = np.load(f'{cache_path}/order_strategy_type.npz', allow_pickle=True)
    strategy_type_df = pd.DataFrame({key: strategy_type_file[key] for key in strategy_type_file.keys()})
    
    strategy_type_df['order_id'] = strategy_type_df['order_id'].astype(str)
    strategy_type_df['strategy_id'] = strategy_type_df['strategy_id'].astype(str)

    strategy_type_df['order_id_idx'] = strategy_type_df.order_id.apply(lambda x: int(x.split('-')[-1]))
    strategy_type_df['strategy_id_idx'] = strategy_type_df.strategy_id.apply(lambda x: int(x.split('-')[-1]))
    strategy_type_df_complete = strategy_type_df.drop_duplicates(subset=['strategy_id'], keep='last')

    cols = ['order_id', 'price', 'ts', 'side']
    open_order = order_df[(order_df['status'] == 'OPENED')][cols].rename(columns={'price': 'open_price'})
    cancel_order = order_df[(order_df['status'] == 'CANCELED') | (order_df['status'] == 'EXPIRED')][cols].rename(columns={'price': 'cancel_price'})
    filled_order = order_df[(order_df['status'] == 'FILLED')][cols].rename(columns={'price': 'filled_price'})

    signal_open_order = strategy_type_df.query("strategy_type == 'signal_open'")
    stop_loss_order = strategy_type_df.query("strategy_type == 'stop_loss'")

    f = open(f"{cache_path}/stat_analysis_info.text", "a")

    get_hour_returns_plot(cache_path)

    get_order_event_count(f, trading_days, open_order, cancel_order, filled_order)

    get_signal_filled_stat(f, open_order, filled_order, signal_open_order, stop_loss_order)

    get_filled_period_stat(f, open_order, filled_order, signal_open_order, stop_loss_order, ts_period)
    
    get_signal_diff_price_stat(f, open_order, filled_order, signal_open_order, stop_loss_order)
    