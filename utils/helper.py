from datetime import datetime, timezone, timedelta

from nautilus_trader.model.book import OrderBook


def convert_timestamp(ts_event_ms, adjust_time_zone=0):  # ts_event_ms是毫秒
    ts_event_ms = ts_event_ms + adjust_time_zone * 3600 * 1000
    event_time = datetime.fromtimestamp(ts_event_ms / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    return event_time


def get_depth_level(orderbook: OrderBook, level, type):
    if type == 'asks':
        _asks = orderbook.asks()
        if _asks is None or len(_asks) == 0:
            return None, None
        return float(orderbook.asks()[level].price), float(orderbook.asks()[level].orders()[0].size)
    else:
        _bids = orderbook.bids()
        if _bids is None or len(_bids) == 0:
            return None, None
        return float(orderbook.bids()[level].price), float(orderbook.bids()[level].orders()[0].size)


def get_depth_all(orderbook: OrderBook, type, deep=10):
    if type == 'asks':
        _asks = orderbook.asks()
        if _asks is None or len(_asks) == 0:
            return None, None
        return [(float(ask.price), float(ask.orders()[0].size)) for ask in _asks[:deep]]
    else:
        _bids = orderbook.bids()
        if _bids is None or len(_bids) == 0:
            return None, None
        return [(float(bid.price), float(bid.orders()[0].size)) for bid in _bids[:deep]]


def iter_tool(data):
    for data in data.itertuples():
        yield data
