from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.events import OrderEvent,OrderFilled,OrderCanceled,OrderExpired,OrderSubmitted, OrderRejected
from nautilus_trader.model.enums import LiquiditySide, OrderSide
from nautilus_trader.model.objects import Money
from nautilus_trader.model.orders import Order, LimitOrder, MarketOrder
from ..stat_data import *
from copy import deepcopy

import time

####################################################
# Nautilus Trder所需的一些转换
####################################################

ORDER_MAP: dict[str, OrderInfo] = {}

ORDER_ORIGINAL_MAP: dict[str, Order] = {}

# def initial_balance(symbol: SymbolInfo, init_balances: list[str], start_ts) -> SymbolBalance:
#     return SymbolBalance(symbol, get_capital_list(init_balances), start_ts)


def register_order(symbol: SymbolInfo, order: Order):
    # symbol在初始化时作为参数传入
    id = str(order.client_order_id)
    ORDER_ORIGINAL_MAP[id] = order

    if order.side == OrderSide.BUY:
        trade_side = TradeSide.BUY
    elif order.side == OrderSide.SELL:
        trade_side = TradeSide.SELL
    else:
        print(f"Currently not supporting orders of side '{order.side}'!")
        trade_side = TradeSide.INVALID_SIDE

    price = 0.0
    if isinstance(order, LimitOrder):
        order_type = OrderType.LIMIT
        price = float(order.price)
    elif isinstance(order, MarketOrder):
        order_type = OrderType.MARKET
    else:
        print(f"Currently not supporting orders of type '{type(order)}'!")
        order_type = OrderType.INVALID_ORDER
    
    ORDER_MAP[id] = OrderInfo(
        order_id = id,
        symbol_info = symbol,
        price = price,
        trade_side = trade_side,
        order_type = order_type,
        amount = float(order.quantity),
        filled_price = 0.0,
        filled_amount = 0.0,
        tp = 0,
        status = OrderStatus.INVALID_STATUS,
    )

def from_quote_tick(quote_tick: QuoteTick) -> MarketPriceInfo:
    market_price_info = MarketPriceInfo(ask_price=float(quote_tick.ask_price),\
                                        ask_amount=float(quote_tick.ask_size),\
                                        bid_price=float(quote_tick.bid_price),\
                                        bid_amount=float(quote_tick.bid_size),\
                                        tp=quote_tick.ts_event,\
                                       )
    return market_price_info

def from_order_event(event: OrderEvent) -> OrderInfo:
    id = str(event.client_order_id)
    if not id in ORDER_MAP:
        print(f"Unregistered order '{id}'!")
        return None
    order_info = ORDER_MAP[id]
    order = ORDER_ORIGINAL_MAP[id]
    
    if isinstance(event, OrderSubmitted):
        order_info.status = OrderStatus.OPENED
        order_info.ts = ts_to_millisecond(event.ts_event)
        # order_info.ts = event.ts_event
    elif isinstance(event, OrderFilled):
        order_info.ts = ts_to_millisecond(event.ts_event)
        # order_info.ts = event.ts_event
        price = float(event.last_px)
        amount = float(event.last_qty)
        order_info.filled_price = (order_info.filled_price * order_info.filled_amount + price * amount) / (order_info.filled_amount + amount)
        # calc_filled_price = (order_info.filled_price * order_info.filled_amount + price * amount) / (order_info.filled_amount + amount)
        # calc_filled_amount = order_info.filled_amount + amount
        order_info.filled_amount += amount
        # order_info.filled_price = float(order.avg_px)
        # order_info.filled_amount = float(order.filled_qty)
        # if calc_filled_price != order_info.filled_price:
        #     print(f"Calculated filled price: {calc_filled_price}, Nautilus avg price: {float(order.avg_px)}")
        #     print(f"Calculated filled amount: {calc_filled_amount}, Nautilus filled amount: {float(order.filled_qty)}")
            
        if order_info.filled_amount == order_info.amount:
            order_info.status = OrderStatus.FILLED
        else:
            order_info.status = OrderStatus.PARTLIFILLED
        # TODO: order.commissions()是不同币种计价的相同的累计commission，目前只有USDT计价的commission，所以还没出问题
        order_info.commission = float(sum(order.commissions())) # 假定每次fill的commission用相同Currency
    elif isinstance(event, OrderCanceled):
        order_info.status = OrderStatus.CANCELED
        order_info.ts = ts_to_millisecond(event.ts_event)
    elif isinstance(event, OrderExpired):
        order_info.status = OrderStatus.EXPIRED
        order_info.ts = ts_to_millisecond(event.ts_event)
    elif isinstance(event, OrderRejected):
        order_info.status = OrderStatus.REJECTED
        order_info.ts = ts_to_millisecond(event.ts_event)
    else:
        print(f"Currently not supporting order events of type '{type(event)}'!")

    if order.liquidity_side == LiquiditySide.TAKER:
        trade_type = TradeType.TAKER
    elif order.liquidity_side == LiquiditySide.MAKER:
        trade_type = TradeType.MAKER
    else:
        if order_info.status == OrderStatus.INVALID_STATUS:
            print(f"Currently not supporting orders of liquidity side '{order.liquidity_side}'!")
        trade_type = TradeType.INVALID_TRADE
    order_info.trade_type = trade_type

    return duplicate_data(order_info)

def from_on_stop(instrument_id, symbol: SymbolInfo, close_amount: float, close_price: float, close_side: TradeSide, close_ts: int, commissions_rate=0.0):
    id = f"on_stop_close_all_order_{ts_to_millisecond(close_ts)}"
    order_info = OrderInfo(
        order_id = str(id),
        symbol_info = symbol,
        price = close_price,
        trade_side = TradeSide.BUY if close_side == OrderSide.BUY else TradeSide.SELL,
        order_type = OrderType.MARKET,
        amount = close_amount,
        filled_price = close_price,
        filled_amount = close_amount,
        tp = ts_to_millisecond(close_ts),
        commission = commissions_rate * close_amount * close_price,
        # tp=close_ts,
        status = OrderStatus.FILLED,
    )

    # order = self.order_factory.limit(
    #         instrument_id=instrument_id,
    #         order_side=close_side,
    #         price=Price(close_price, precision=self.instrument.price_precision),
    #         quantity=self.instrument.make_qty(close_amount),
    #         reduce_only=reduce_only,
    #         time_in_force=TimeInForce.GTC,  # Use Fill or Kill for market orders
    #         exec_algorithm_id=None,
    #         exec_algorithm_params=None
    #     )

    return duplicate_data(order_info)

    
