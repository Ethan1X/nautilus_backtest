


import os
import sys
import tomli
import tomli_w
import pandas  as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
from .stat_data import *

if '.' not in sys.path:
    sys.path.append('./dtanalyse/')
from util.load_s3_data import LoadS3Data
from datetime import datetime, timedelta


def convert_ts(ts, adjust_time_zone=8):
    ts_ = ts_to_second(ts) + adjust_time_zone * 3600
    return datetime.fromtimestamp(ts_)

def draw_price(fig, market_prices, prices_list, interval=1):
    # ticker data用s3的数据
    vectorized_function = np.vectorize(convert_ts)
    price_ts = vectorized_function(market_prices['ts'])[::interval]
    for price in prices_list:
        # 价格变化
        fig.add_trace(
            go.Scatter(x=price_ts, y=market_prices[price][::interval], mode='lines', name=price, line=dict(shape='hv')),
            secondary_y=False,
        )
    return fig

def draw_net_value(fig, net_value_list, interval=1):
    vectorized_function = np.vectorize(convert_ts)
    net_value_ts = vectorized_function(net_value_list['ts'])[::interval]
    net_value = (net_value_list['net_value'] - net_value_list['net_value'][0])[::interval]
    
    fig.add_trace(
        go.Scatter(x=net_value_ts, y=net_value, mode='lines', name='Net Worth Curve', line=dict(color='blue')),
        secondary_y=True,
    )
    # # 添加最低点最高点
    # max_net_value_fidx = np.argmax(net_value)
    # min_net_value_fidx = np.argmin(net_value)
    # fidx_list = [0, max_net_value_fidx, min_net_value_fidx, len(net_value)-1]
    # for i in fidx_list:
    #     fig.add_annotation(
    #         x=net_value_ts[i], y=net_value[i],
    #         text=f'{net_value[i]:,.3f}',
    #         showarrow=True, arrowhead=1, ax=0, ay=-30,
    #         font=dict(size=10),
    #     )
    return fig

def draw_position(fig, balance_list):
    vectorized_function = np.vectorize(convert_ts)
    balance_ts = vectorized_function(balance_list['ts'])
    balance_value = balance_list['position']

    # 仓位变化
    fig.add_trace(
        go.Scatter(x=balance_ts, y=balance_value, mode='lines', name='Position', line=dict(color='green', dash='dash', shape='hv')),
        secondary_y=True,
    )
    return fig

def draw_hedge(fig, hedge_list, type_list):
    # 开平仓信号点
    amount_open = hedge_list['amount_open']
    open_side = hedge_list['open_side']
    ts_create = hedge_list['ts_create']
    ts_finish = hedge_list['ts_finish']
    avg_price_open = hedge_list['avg_price_open']
    avg_price_close = hedge_list['avg_price_close']
    # print()
    
    if type_list is not None:
        colors = {"buy_to_open": "red", "sell_to_close": "blue", "sell_to_open": "green", "buy_to_close": "orange"}
        order_ts = defaultdict(list)
        order_price = defaultdict(list)
        for fidx in range(len(ts_finish)):
            if abs(amount_open[fidx]) < ZERO_AMOUNT_THRESHOLD:
                continue
            if open_side[fidx] == 'BUY':
                order_ts["buy_to_open"].append(convert_ts(ts_create[fidx]))
                order_price["buy_to_open"].append(avg_price_open[fidx])
                if ts_finish[fidx] != 0:
                    order_ts["sell_to_close"].append(convert_ts(ts_finish[fidx]))
                    order_price["sell_to_close"].append(avg_price_close[fidx])
            elif open_side[fidx] == 'SELL':
                order_ts["sell_to_open"].append(convert_ts(ts_create[fidx]))
                order_price["sell_to_open"].append(avg_price_open[fidx])
                if ts_finish[fidx] != 0:
                    order_ts["buy_to_close"].append(convert_ts(ts_finish[fidx]))
                    order_price["buy_to_close"].append(avg_price_close[fidx])
    
        for type in type_list:
            fig.add_trace(
                go.Scatter(x=order_ts[type], y=order_price[type], mode='markers', name=type, marker=dict(color=colors[type], size=6)),
                secondary_y=False,
            )
    return fig

def draw_orders(fig, order_list, status_list, mode='markers'):
    orders_df = pd.DataFrame({key: order_list[key] for key in order_list.keys()})
    orders_df['ts'] = orders_df['ts'].apply(lambda x: convert_ts(x))
    # status_list = ['INVALID_STATUS', 'CANCELED', 'FILLED', 'REJECTED']
    
    for status in status_list:
        _df = orders_df[orders_df['status'] == status] if status != 'OPENED' else orders_df[orders_df['status'] == 'INVALID_STATUS']
        fig.add_trace(
            go.Scatter(x=_df['ts'], y=_df['price'], mode=mode, name=status, marker=dict(size=6)),
            secondary_y=False
        )
    return fig

def draw_stat_metrics(fig, stat_info, symbol, start_token, end_token, start_quote, end_quote):
    metrics = {
        # balance
        "Starting Balance(U)": f'{symbol["token"]}: {start_token:.3f}; {symbol["quote"]}: {start_quote:.3f}',
        "Ending Balance(U)": f'{symbol["token"]}: {end_token:.3f}; {symbol["quote"]}: {end_quote:.3f}',
        # holding time
        'Average Holding Time(s)': stat_info["average_holding_time"],
        'Total Trading Counts': stat_info["trading_counts"],
        'Daily Trading Counts': stat_info["daily_trading_counts"],
        'Trading Days': stat_info["trading_days"],
        'Turnover Rate(%)': stat_info["turnover_rate"],
        'Daily Turnover Rate(%)': stat_info["daily_turnover_rate"],
        'Total Trading Value(U)': stat_info["total_trading_value"],
    }
    _metrics = []
    for key, value in metrics.items():
        if isinstance(value, float):
            _metrics.append(f'{key}: {value:.5f}')
        else:
            _metrics.append(f'{key}: {value}')
    fig.add_annotation(
                # width=800,
                # height=100,
                text='<br>'.join(_metrics),
                xref='paper', yref='paper',
                x=0.0, y=-0.4, showarrow=False,
                font=dict(size=13),
                align="left",
    )
    metrics = {
        # win and loss indicator
        'Without Commissions': '',
        'Win Percentage(%)': stat_info["win_percentage_without_commission"],
        'Win Counts': stat_info["win_counts_without_commission"],
        'Lose Counts': stat_info["loss_counts_without_commission"],
        'Average Win Amount(U)': stat_info["average_win_amount_without_commission"],
        'Average Loss Amount(U)': stat_info["average_loss_amount_without_commission"],
        'Average Win Percentage(%)': stat_info["average_win_percentage_without_commission"],
        'Average Loss Percentage(%)': stat_info["average_loss_percentage_without_commission"],
        'Average Returns Percentage(%)': stat_info["average_returns_without_commission"],
    }
    _metrics = []
    for key, value in metrics.items():
        if isinstance(value, float):
            _metrics.append(f'{key}: {value:.5f}')
        else:
            _metrics.append(f'{key}: {value}')
    fig.add_annotation(
                # width=800,
                # height=100,
                text='<br>'.join(_metrics),
                xref='paper', yref='paper',
                x=0.25, y=-0.4, showarrow=False,
                font=dict(size=13),
                align="left",
    )
    metrics = {
        # win and loss indicator
        'With Commissions': '',
        'Win Percentage(%)': stat_info["win_percentage_with_commission_without_zero"],
        'Win Counts': stat_info["win_counts_with_commission_without_zero"],
        'Lose Counts': stat_info["loss_counts_with_commission_with_zero"],
        'Average Win Amount(U)': stat_info["average_win_amount_with_commission_without_zero"],
        'Average Loss Amount(U)': stat_info["average_loss_amount_with_commission_without_zero"],
        'Average Win Percentage(%)': stat_info["average_win_percentage_with_commission_without_zero"],
        'Average Loss Percentage(%)': stat_info["average_loss_percentage_with_commission_with_zero"],
        'Average Returns Percentage(%)': stat_info["average_returns_with_commission_without_zero"],
    }
    # metrics = {
    #     # win and loss indicator
    #     'Without Commissions(win with zero)': '',
    #     'Win Percentage(%)': stat_info["win_percentage_with_commission_with_zero"],
    #     'Win Counts': stat_info["win_counts_with_commission_with_zero"],
    #     'Lose Counts': stat_info["loss_counts_with_commission_without_zero"],
    #     'Average Win Amount(U)': stat_info["average_win_amount_with_commission_with_zero"],
    #     'Average Loss Amount(U)': stat_info["average_loss_amount_with_commission_without_zero"],
    #     'Average Win Percentage(%)': stat_info["average_win_percentage_with_commission_with_zero"],
    #     'Average Loss Percentage(%)': stat_info["average_loss_percentage_with_commission_without_zero"],
    #     'Average Returns Percentage(%)': stat_info["average_returns_with_commission_without_zero"],
    # }
    _metrics = []
    for key, value in metrics.items():
        if isinstance(value, float):
            _metrics.append(f'{key}: {value:.5f}')
        else:
            _metrics.append(f'{key}: {value}')
    fig.add_annotation(
                # width=800,
                # height=100,
                text='<br>'.join(_metrics),
                xref='paper', yref='paper',
                x=0.6, y=-0.4, showarrow=False,
                font=dict(size=13),
                align="left",
    )
    metrics = {
        # net worth value indicator
        'Total Returns(%)': stat_info["total_returns_rate"],
        'Annual Returns(%)': stat_info["annual_returns"],
        'Sharpe Ratio(%)': stat_info["sharpe_ratio"],
        'Maximum Drawdown(U)': stat_info["maxdrawdown"],
        'Max Drawdown Rate(%)': stat_info["maxdrawdown_rate"],
        'Drawdown Interval(h)': stat_info["drawdown_interval"],
        # turnover and profit
        'Total Gain/Loss without Commissions(U)': stat_info["total_returns"],
        'Total Commissions(U)': stat_info["total_commissions"],
        'Total Gain/Loss(U)': end_quote - start_quote,
    }
    _metrics = []
    for key, value in metrics.items():
        if isinstance(value, float):
            _metrics.append(f'{key}: {value:.5f}')
        else:
            _metrics.append(f'{key}: {value}')
    fig.add_annotation(
                # width=800,
                # height=100,
                text='<br>'.join(_metrics),
                xref='paper', yref='paper',
                x=1, y=-0.4, showarrow=False,
                font=dict(size=13),
                align="left",
    )
    return fig

def draw_stat_plot_html(symbol_info, stat_info, market_price_list, balance_list, net_value_list, hedge_list, order_list, config: dict, run_dir: str, period: str, plot_title='trade_pnl_anlyse', interval=1):
    data = {
        "stat_info": stat_info,
        "market_prices": market_price_list,
        "balance_list": balance_list,
        "net_value_list": net_value_list,
        "hedge_list": hedge_list,
        "order_list": order_list,
    }
    fig = make_subplots(
        rows=1, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.1,
        # subplot_titles=('Net Worth Curve and Price'), 
        specs=[[{"secondary_y": True}]]
    )
    fig.update_layout(
        title_text=f'{symbol_info["token"]} Trading Strategy Performance',
        showlegend=True,
        # margin=dict(t=100, b=300),
        margin=dict(b=250),
        # height=800,
        # width=1400,
        # xaxis=dict(
        #     rangeslider=dict(
        #         visible=True,
        #         bgcolor='#c5c5c5',
        #         bordercolor='#888888',
        #         thickness=0.1,
        #     ),
        # ),
    )
    if config.get("market_prices", False):
        fig = draw_price(fig, data['market_prices'], config["market_prices"].get("prices_list"), interval=1)
    if config.get("net_value", False):
        fig = draw_net_value(fig, data['net_value_list'], interval=1)
    if config.get("position", False):
        fig = draw_position(fig, data['balance_list'])
    if config.get("hedge", False):
        fig = draw_hedge(fig, data['hedge_list'], config["hedge"].get("type_list"))
    if config.get("orders", False):
            fig = draw_orders(fig, data['order_list'], config["orders"].get("status_list"))
    if config.get("stat_metrics", False):
        fig = draw_stat_metrics(fig, data['stat_info'], symbol_info, balance_list["token_capital"][0], balance_list["token_capital"][-1],
                               balance_list["quote_capital"][0], balance_list["quote_capital"][-1])

    # todo：保存文件或者显示
    # 保留现有，单独 html 保存
    # html 和 png 两种模式
    # png 的报告模式
    if plot_title is None:
        plot_title = 'trade_pnl_anlyse'
    html_filename = os.path.join(run_dir, f"{plot_title}_{period}.html")
    # html_filename = "Plotly_Net_Worth_Curve.html"
    fig.write_html(html_filename)
    print(f"Plotly plot saved to {html_filename}")

    return

def draw_stat_plot_png(symbol_info, stat_info, market_price_list, balance_list, net_value_list, hedge_list, order_list, config: dict, run_dir: str, period: str, plot_title='trade_pnl_anlyse', interval=1):
    data = {
        "stat_info": stat_info,
        "market_prices": market_price_list,
        "balance_list": balance_list,
        "net_value_list": net_value_list,
        "hedge_list": hedge_list,
        "order_list": order_list,
    }
    fig = make_subplots(
        rows=1, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.1,
        # subplot_titles=('Net Worth Curve and Price'), 
        specs=[[{"secondary_y": True}]]
    )
    fig.update_layout(
        title_text=f'{symbol_info["token"]} Trading Strategy Performance',
        showlegend=True,
        # margin=dict(t=100, b=300),
        margin=dict(b=250),
        # height=800,
        # width=1400,
        # xaxis=dict(
        #     rangeslider=dict(
        #         visible=True,
        #         bgcolor='#c5c5c5',
        #         bordercolor='#888888',
        #         thickness=0.1,
        #     ),
        # ),
    )
    if config.get("market_prices", False):
        fig = draw_price(fig, data['market_prices'], config["market_prices"].get("prices_list"), interval=1)
    if config.get("net_value", False):
        fig = draw_net_value(fig, data['net_value_list'], interval=1)
    if config.get("position", False):
        fig = draw_position(fig, data['balance_list'])
    if config.get("hedge", False):
        fig = draw_hedge(fig, data['hedge_list'], config["hedge"].get("type_list"))
    if config.get("orders", False):
            fig = draw_orders(fig, data['order_list'], config["orders"].get("status_list"))
    if config.get("stat_metrics", False):
        fig = draw_stat_metrics(fig, data['stat_info'], symbol_info, balance_list["token_capital"][0], balance_list["token_capital"][-1],
                               balance_list["quote_capital"][0], balance_list["quote_capital"][-1])

    # todo：保存文件或者显示
    # 保留现有，单独 html 保存
    # html 和 png 两种模式
    # png 的报告模式
    if plot_title is None:
        plot_title = 'trade_pnl_anlyse'
    html_filename = os.path.join(run_dir, f"{plot_title}_{period}.html")
    # html_filename = "Plotly_Net_Worth_Curve.html"
    fig.write_html(html_filename)
    print(f"Plotly plot saved to {html_filename}")

    return


def price_from_s3(start_time, end_time, exchange, symbol):
    # begin_time = datetime.datetime(2024, 4, 19, 14, 30, tzinfo=TZ_8)
    # end_time = datetime.datetime(2024, 4, 19, 15, tzinfo=TZ_8)

    cex_ticker_data = LoadS3Data.get_cex_ticker(start_time, end_time, symbol, exchange, plot_interval_us=None, is_adj_time=True)
    
    ticker_data = pd.DataFrame.from_records(cex_ticker_data)

    ticker_data['mid_price'] = (ticker_data.ap + ticker_data.bp) / 2
    ticker_data['ap_diff'] = ticker_data.ap.diff()
    ticker_data['bp_diff'] = ticker_data.bp.diff()
    ticker_data.iloc[-1, -1] = np.nan
    ticker_data = ticker_data[(ticker_data.ap_diff != 0) | (ticker_data.bp_diff != 0)]
    
    ticker_data['datetime'] = pd.to_datetime(ticker_data['tp'], unit='ms')
    ticker_data = ticker_data.rename(columns={'tp': 'ts'})
    ticker_data = ticker_data.sort_values('datetime')

    price_df = ticker_data[['ts', 'ap', 'bp', 'mid_price']]
    price_df.columns = ['ts', "Ask Price", "Bid Price", 'Mid Price']
    price_list = {col: price_df[col].values for col in price_df.columns}
    return price_list
    

def plot_func(plot_config, npz_path, start_time, end_time, run_dir=None, plot_title=None, interval=1):
    period = f'{start_time.strftime("%Y%m%d")}_{end_time.strftime("%Y%m%d")}'
    if run_dir is None:
        run_dir = npz_path
    with open(f"{npz_path}/symbol_info.toml", "rb") as f:
        symbol_info = tomli.load(f)
    with open(f"{npz_path}/stat_info.toml", "rb") as f:
        stat_info = tomli.load(f)
    balance_npz = np.load(f"{npz_path}/balance.npz", allow_pickle=True)
    net_value_npz = np.load(f"{npz_path}/net_value.npz", allow_pickle=True)
    hedge_npz = np.load(f"{npz_path}/hedge.npz", allow_pickle=True)
    order_npz = np.load(f"{npz_path}/orders.npz", allow_pickle=True)
    price_list = price_from_s3(start_time, end_time, symbol_info['market_info']['exchange'].lower(), f'{symbol_info["token"]}_{symbol_info["quote"]}'.lower())

    # draw_stat_plot_html(symbol_info, stat_info, price_list, balance_npz, net_value_npz, hedge_npz, order_npz, plot_config, run_dir, period, plot_title, interval=1)

    draw_stat_plot_png(symbol_info, stat_info, price_list, balance_npz, net_value_npz, hedge_npz, order_npz, plot_config, run_dir, period, plot_title, interval=1)