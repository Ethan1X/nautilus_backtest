import os, time, sys, datetime, heapq, logging
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from util.load_s3_data import LoadS3Data
from util.time_method import *

class AggressorSide:
    BUYER = 'buy'
    SELLER = 'sell'

class QuoteTick(object):
    
    def __init__(self, ask_price: float, ask_size: float, bid_price: float, bid_size: float, ts_event: int) -> None:

        self.ask_price = ask_price
        self.ask_size = ask_size
        self.bid_price = bid_price
        self.bid_size = bid_size
        self.ts_event = ts_event    # 单位 纳秒

    def __str__(self):
        return f'{self.__dict__}'


class TradeTick:

    def __init__(self, aggressor_side: str, price: float, size: float, ts_event: int) -> None:

        self.aggressor_side = aggressor_side
        self.price = price
        self.size = size
        self.ts_event = ts_event # 单位 纳秒

    def __str__(self):
        return f'{self.__dict__}'


class OrderBook:

    def __init__(self, ts_event: int, _asks: list, _bids: list) -> None:
        self.ts_event = ts_event
        self._asks = _asks
        self._bids = _bids

    def asks(self):
        return self._asks
    def bids(self):
        return self._bids


def trans_backtest_ticker(ticker) -> QuoteTick:
    return QuoteTick(float(ticker.ask_price), float(ticker.ask_size), float(ticker.bid_price), float(ticker.bid_size), ticker.ts_event)

def trans_backtest_trade(trade) -> TradeTick:
    return TradeTick('buy' if str(trade.aggressor_side) == '1' else 'sell', float(trade.price), float(trade.size), trade.ts_event)

def trans_backtest_depth(depth) -> OrderBook:
    _asks = depth.asks()
    _bids = depth.bids()
    new_asks, new_bids = [], []
    for i in range(min(20, len(_bids), len(_asks))):
        pb, sb, pa, sa = float(_bids[i].price), float(_bids[i].orders()[0].size), float(_asks[i].price), float(_asks[i].orders()[0].size)
        new_asks.append({'price': pa, 'size': sa})
        new_bids.append({'price': pb, 'size': sb})
    return OrderBook(depth.ts_event, new_asks, new_bids)

def tmp_trans_s3_backtest_ticker(ticker) -> QuoteTick:
    '''
        转换S3格式的ticker为回测系统类型
    '''
    return QuoteTick(ticker['ap'], ticker['aa'], ticker['bp'], ticker['ba'], ticker['tp']*10**6)

def tmp_trans_s3_backtest_depth(depth) -> OrderBook:

    return OrderBook(depth['tp']*10**6, depth['asks'], depth['bids'])

def tmp_trans_s3_backtest_trade(trade) -> TradeTick:
    side = AggressorSide.SELLER if trade['m'] == 'SELL' else AggressorSide.BUYER
    return TradeTick(side, trade['p'], trade['q'], trade['tp']*10**6)


# def get_depth_level(orderbook: OrderBook, level: int, type:str):
#     if type == 'asks':
#         _asks = orderbook.asks()
#         if _asks is None or len(_asks) < level+1:
#             return None, None
#         return orderbook.asks()[level]['p'], orderbook.asks()[level]['s']
#     else:
#         _bids = orderbook.bids()
#         if _bids is None or len(_bids) < level+1:
#             return None, None
#         return orderbook.bids()[level]['p'], orderbook.bids()[level]['s']


def get_depth_level(orderbook: OrderBook, level, type):
    if type == 'asks':
        _asks = orderbook.asks()
        if _asks is None or len(_asks) <= level:
            return None, None
        return float(_asks[level].price), float(_asks[level].orders()[0].size)
    else:
        _bids = orderbook.bids()
        if _bids is None or len(_bids) <= level:
            return None, None
        return float(_bids[level].price), float(_bids[level].orders()[0].size)



def get_data_type(data_list, data_type):
    for row in data_list:
        row['type'] = data_type
    return data_list

def get_sort_v(d):
    return d['tp']



def get_data_generator(begin_time, end_time, exchange, symbol):
    '''
        对接回测系统生成流式数据
    '''
    logging.info(f"开始生成{exchange}:{symbol} {begin_time}~{end_time}的数据")
    t1 = time.time()
    ticker_data = LoadS3Data.get_cex_ticker(begin_time, end_time, symbol, exchange)
    # trade_data = LoadS3Data.get_cex_trade(begin_time, end_time, exchange, symbol)
    # depth_data = LoadS3Data.get_cex_depth_online(begin_time, end_time, exchange, f"{symbol}_nats")
    # depth_data = LoadS3Data.get_cex_depth(begin_time, end_time, exchange, symbol)
    print(time.time() - t1)
    print(len(ticker_data))    # , len(trade_data), len(depth_data))

    all_data_iterator = ticker_data
    # all_data = [get_data_type(ticker_data, 'ticker'), get_data_type(trade_data, 'trade'), get_data_type(depth_data, 'depth')]
    # print(time.time() - t1)

    # all_data_iterator = heapq.merge(*all_data, key=get_sort_v)
    print(time.time() - t1)
    logging.info(f"数据组装完成 共耗时: {time.time() - t1}s")
    
    # return all_data_iterator
    for row in all_data_iterator:
        if 'type' not in row.keys() or row['type'] == 'ticker':
            yield tmp_trans_s3_backtest_ticker(row), 'ticker'
        elif 'type' in row.keys() and row['type'] == 'depth':
            yield tmp_trans_s3_backtest_depth(row), 'depth'
        else:
            yield tmp_trans_s3_backtest_trade(row), 'trade'



if __name__ == "__main__":
    begin_time = datetime.datetime(2024, 3, 27, 21,tzinfo=TZ_8)
    end_time = datetime.datetime(2024, 3, 27, 21, 10,tzinfo=TZ_8)
    exchange = 'binance'
    symbol = 'btc_usdt'

    for idx, row in enumerate(get_data_generator(begin_time, end_time, exchange, symbol)):
        print(idx, row)

        if idx >= 100:
            break
