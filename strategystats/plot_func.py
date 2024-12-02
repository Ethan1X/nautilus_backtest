####################################################
# 绘图功能
####################################################
from collections import defaultdict
from datetime import datetime, timedelta
import pandas  as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
from .stat_data import *


def convert_ts(ts, adjust_time_zone=8):
    ts_ = ts_to_second(ts) + adjust_time_zone * 3600
    return datetime.fromtimestamp(ts_)

def draw_position(fig, market_prices, balance_in_time, interval=1):
    vectorized_function = np.vectorize(convert_ts)
    # price_ts = vectorized_function(market_prices['ts'])[::interval]
    # price_value = market_prices['mid_price'][::interval]
    price_ts = vectorized_function(market_prices['ts'])
    price_value = market_prices['mid_price']
    balance_ts = vectorized_function(balance_in_time['ts'])[::interval]
    balance_value = balance_in_time['position'][::interval]
    
    # 价格变化
    fig.add_trace(
        go.Scatter(x=price_ts, y=price_value, mode='lines', name='Price', line=dict(color='purple', shape='hv')),
        secondary_y=False,
    )
    # 仓位变化
    fig.add_trace(
        go.Scatter(x=balance_ts, y=balance_value, mode='lines', name='Position', line=dict(color='green', dash='dash', shape='hv')),
        secondary_y=True,
    )

    return fig

def draw_order_mark(fig, hedge_list, type_list):
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


def draw_signal(fig, signal_list, mode='markers'):
    name_list = signal_list['name']
    price_list = signal_list['price']
    ts_list = signal_list['ts']
    name_set = set(name_list)
    signal_ts = defaultdict(list)
    signal_price = defaultdict(list)
    for fidx in range(len(ts_list)):
        signal_ts[name_list[fidx]].append(convert_ts(ts_list[fidx]))
        signal_price[name_list[fidx]].append(price_list[fidx])
    
    for name in name_set:
        fig.add_trace(
            go.Scatter(x=signal_ts[name], y=signal_price[name], mode=mode, name=name, marker=dict(size=6)),
            secondary_y=True
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

def convert_balance_in_time(balance_list):
    # 对齐balance和price的时间戳
    # 先把两个list按时间戳排序
    balance_in_time_list = {
        'ts': [],
        'position': []
    }
    _ts_list = balance_list['ts']
    _position_list = balance_list['position']

    for fidx in range(len(_ts_list)):
        ts = _ts_list[fidx]
        ts_f = ts - 1
        ts_b = ts + 1

        if fidx != 0:
            balance_in_time_list['ts'].append(ts_f)
            balance_in_time_list['position'].append(_position_list[fidx - 1])

        balance_in_time_list['ts'].append(ts)
        balance_in_time_list['position'].append(_position_list[fidx])

        balance_in_time_list['ts'].append(ts_b)
        balance_in_time_list['position'].append(_position_list[fidx])
    
    return balance_in_time_list

def draw_stat_plot(symbol_info, stat_info, market_price_list, balance_list, net_value_list, hedge_list, signal_list, config: dict, run_dir: str, period='all', plot_title='trade_pnl_anlyse', interval=1):
    data = {
        "stat_info": stat_info,
        "market_prices": market_price_list,
        "balance_in_time": balance_list,
        "net_value_list": net_value_list,
        "hedge_list": hedge_list,
        "signal_list": signal_list,
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
    if config.get("net_value", False):
        fig = draw_net_value(fig, data['net_value_list'], interval)

    if config.get("position", False):
        fig = draw_position(fig, data['market_prices'], data['balance_in_time'], interval)

    if config.get("hedge", False):
        fig = draw_order_mark(fig, data['hedge_list'], config["hedge"].get("type_list"))

    if config.get("stat_metrics", False):
        fig = draw_stat_metrics(fig, data['stat_info'], symbol_info, balance_list["token_capital"][0], balance_list["token_capital"][-1],
                               balance_list["quote_capital"][0], balance_list["quote_capital"][-1])
        
    if config.get("signal", False):
        fig = draw_signal(fig, data['signal_list'])

    # todo：
    # 保留 interval 功能，拆分成为单独html文件
    # 单独文件可选择，单独html 或者单独 png
    # png考虑成为一个整个页面
    if plot_title is None:
        plot_title = 'trade_pnl_analyse'
    html_filename = os.path.join(run_dir, f"{plot_title}_{period}.html")
    # html_filename = "Plotly_Net_Worth_Curve.html"
    fig.write_html(html_filename)
    print(f"Plotly plot saved to {html_filename}")

    return


if __name__ == "__main__":
    # 单元测试不通过
    test_symbol = SymbolInfo("USDT", "USDT", SymbolType.SPOT_NORMAL, Market.SPOT, "OKEX", -0.00005, 0.00015)
    test_balance_list, test_hedge_list, test_price_list, test_net_value_list = [], [], [], []
    # tp = datetime.now()

    tp = 0
    capitals = {"BTC": 10, "USDT": 10}
    print("balance:")
    for i in range(1, 10):
        # tp = tp + timedelta(seconds=5)
        tp += 1
        balance = SymbolBalance(test_symbol, capitals, tp)
        balance.capitals["USDT"] = i * 1000
        balance.position = i * 0.5
        balance.holding_price = i * 100
        test_balance_list.append(balance)
        print(balance)
    # tp = datetime.now()
    tp = 0
    print("hedge:")
    for i in range(1, 5):
        # tp = tp + timedelta(seconds=2)
        # tp_ = tp + timedelta(seconds=1)
        tp = tp + 2
        tp_ = tp + 1
        hedge = HedgeInfo(TradeSide.BUY)
        hedge.total_value = i * 10
        hedge.total_returns = i * 5
        hedge.commissions = i * 1
        hedge.ts_create = tp
        hedge.ts_finish = tp_
        test_hedge_list.append(hedge)
        print(hedge)
    # tp = datetime.now()
    tp = 0
    print("price:")
    for i in range(1, 20):
        # tp = tp + timedelta(seconds=1)
        tp = tp + 1
        price = MarketPriceInfo(ask_price=i * 100, ask_amount=0.0, bid_price=i * 99, bid_amount=0.0, mid_price=i * 100, tp=tp)
        # price.price = i * 100
        # price.ts_event = i * 5
        test_price_list.append(price)
        print(price)
    # tp = datetime.now()
    tp = 0
    print("net:")
    for i in range(1, 20):
        # tp = tp + timedelta(seconds=1)
        tp = tp + 1
        net = NetValue(test_symbol, test_balance_list[0], test_price_list[i-1])
        test_net_value_list.append(net)
        print(net)

    stat_info = StatisticInfo(test_balance_list[0].trading_symbol, test_balance_list[0])

    # plot_config = {
    #     "net_value": {"plot": True},
    #     "position": {"plot": True},
    #     "stat_metrics": {"plot": True},
    #     "order_mark": {"plot": False, "type_list": None},
    #     "holding_spot_value": {"plot": True},
    # }
    plot_config = {
        "net_value": True,
        "position": True,
        "stat_metrics": True,
        "order_mark": {
            "type_list": ["buy_to_open", "sell_to_close", "sell_to_open", "buy_to_close"]
        },
        "holding_spot_value": True,
    }

    run_dir = "strategystats"

    draw_stat_plot(stat_info, test_price_list, test_balance_list, test_net_value_list, test_hedge_list, plot_config, run_dir)

            
