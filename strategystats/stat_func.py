from collections import deque
import pandas as pd
import numpy as np
import os
import tomli
from .stat_data import *
from tqdm import tqdm

# 统计计算功能

####################################################
# Utility
####################################################

# MODULE_PATH = os.path.dirname(stat_data.__file__)
# CONFIG_PATH = os.path.join(MODULE_PATH, "conf/")
CONFIG_PATH = "conf/"

def load_symbol_info(token: str, quote: str, symbol_type: SymbolType, exchange: str, path=CONFIG_PATH) -> SymbolInfo:
    filepath = f"{token}{quote}.{exchange}".upper()
    with open(f"{path}/{filepath}.toml", "rb") as f:
        config = tomli.load(f, parse_float=float)
    trade_config = config["trade"].get(symbol_type.name.lower())
    if trade_config is None:
        print(f"Corrupted config: '{path}/{filepath}.toml'")
    market = Market[trade_config["market"]]
    commission_rate_maker = trade_config["commission_rate_maker"]
    commission_rate_taker = trade_config["commission_rate_taker"]
    symbol = SymbolInfo(token, quote, symbol_type, market, exchange, commission_rate_maker, commission_rate_taker)
    return symbol

def _match_one_order(order, order_list, hedged_amount):
    # 假定order_list非空，其中的订单都与order有相反的side并且已经按时间排序
    # 遍历order_list，直到order被填满，或者order_list被遍历完
    # 返回order的hedge_info和hedge之后的order_list以及list中被hedge的最后一个订单被hedge的量

    # TODO：按着文博的方法，平仓永远只有一个单
    open_order_list = order_list
    open_side = TradeSide.BUY if order.side == TradeSide.SELL else TradeSide.SELL
    hedge_info = HedgeInfo(open_side)
    hedge_info.add_close_order_list([order])
    hedge_info.amount_close = order.filled_amount # TODO：是用amount做还是filled_amount做？
    hedge_info.avg_price_close = order.filled_price # TODO：是用price做还是filled_price做？
    # hedge_info.commissions = order.commission

    hedge_info.ts_finish = order.ts

    # 未被hedge的order_list
    unhedged_order_list = []
    new_hedged_amount = 0.0
    
    # 遍历order_list
    symbol_type = order.symbol_info.symbol_type
    open_order_list = []
    for idx, _order in enumerate(order_list):
        _hedge_amount = _order.filled_amount # TODO：是用amount做还是filled_amount做？
        _hedge_price = _order.filled_price # TODO：是用price做还是filled_price做？
        if idx == 0:
            # 第一个订单只需要hedge剩下的量即可
            _hedge_amount -= hedged_amount
        # if _hedge_amount == 0.0: # TEMP FIX
        #     continue
        if abs(_hedge_amount) < ZERO_AMOUNT_THRESHOLD: # TEMP FIX
            continue
        open_order_list.append(_order)
        if hedge_info.amount_open + _hedge_amount + ZERO_AMOUNT_THRESHOLD < hedge_info.amount_close:
            # 这个订单还不够hedge
            # 更新平仓价格
            hedge_info.avg_price_open = calc_avg_price(symbol_type, hedge_info.avg_price_open, hedge_info.amount_open, _hedge_price, _hedge_amount)
            # 把这个订单的量加入开仓总量
            hedge_info.amount_open += _hedge_amount
            # 把费用加进总费用
            # hedge_info.commissions += _order.commission
            hedge_info.commissions += _order.commission * _hedge_amount / _order.filled_amount
        else:
            # 这个订单够hedge
            # 计算这个订单hedge了多少
            order_hedged_amount = hedge_info.amount_close - hedge_info.amount_open
            if idx == 0:
                new_hedged_amount = hedged_amount
            else:
                new_hedged_amount = 0.0
            new_hedged_amount += order_hedged_amount
            # 更新平仓价格
            hedge_info.avg_price_open = calc_avg_price(symbol_type, hedge_info.avg_price_open, hedge_info.amount_open, _hedge_price, order_hedged_amount)
            # 把这个订单的量加入平仓总量
            hedge_info.amount_open += order_hedged_amount
            # 把费用加进总费用（按比例加）
            hedge_info.commissions += _order.commission * order_hedged_amount / _order.filled_amount
            # 用被hedge的第一个订单的ts作ts_create
            hedge_info.ts_create = open_order_list[0].ts
            # hedge_info到这里填完了，之后是hedged_order_list
            
            # 计算新order_list从哪开始
            next_idx = idx
            if order_hedged_amount == _hedge_amount:
                # 如果被hedge满了就到下一个订单
                next_idx = idx + 1
                new_hedged_amount = 0.0
            if next_idx < len(order_list):
                # 如果都被hedge了那就还是[]
                unhedged_order_list = order_list[next_idx:]

            break

    hedge_info.add_open_order_list(open_order_list)
    
    # 最后把时间戳填一下
    if len(open_order_list) != 0:
        # 如果不为[]，那么一定在上面处理过了
        # 如果order_list不够hedge或者正好hedge完使得unhedged_order_list为[]
        # 那么都可以直接取最后一个order来填hedge_info的ts_finish
        hedge_info.ts_create = open_order_list[0].ts

    hedge_info.commissions += order.commission * hedge_info.amount_open / order.filled_amount
    # if hedge_info.commissions == 0:
    #     print(order.commission)

    return hedge_info, unhedged_order_list, new_hedged_amount
    
def _calculate_hedge_metrics(symbol_type, hedge_info):
    hedge_info.total_value = hedge_info.avg_price_close * hedge_info.amount_close + hedge_info.avg_price_open * hedge_info.amount_open
    sign = 1 if hedge_info.open_side == TradeSide.BUY else -1
    hedge_info.total_returns = calc_profit(symbol_type, hedge_info.avg_price_open, hedge_info.avg_price_close, sign * hedge_info.amount_close)
    
    return hedge_info

    
####################################################
# 订单相关
####################################################

def select_terminated_orders(order_list: OrderList, start_time=None, end_time=None):
    # appeared = set()
    # terminated_orders_index = deque()
    # # for order in reversed(order_list):
    # data_size = len(order_list_array["order_id"])
    # for index in tqdm(range(data_size-1, -1, -1)):
    #     # if end_time is not None and order_list_array["ts"][index] > end_time:
    #     #     continue
    #     # if start_time is not None and order_list_array["ts"][index] < start_time:
    #     #     break
    #     if not order_list_array["order_id"][index] in appeared:
    #         appeared.add(order_list_array["order_id"][index])
    #         terminated_orders_index.appendleft(index)

    # odd_count = np.array([1 for i in terminated_orders_index if order_list_array["filled_amount"][i] > 0]).sum()
    # print(len(terminated_orders_index), len(appeared), odd_count)
    # return list(terminated_orders_index)
    return list(order_list.index.values())         

def matching_orders(order_index_list, order_data_list, symbol_info):
    # NOTE: 假定order_list中每个订单均处于终态

    if len(order_index_list) == 0:
        return []
    
    _hedge_list: List(HedgeInfo) = []
    # todo: 根据成交量，在open与close订单之间进行匹配
    # 当前采用的是文博的方法
    # 按照时间顺序按量依次匹配

    # 对order_list按ts_finish排序
    # _order_list = sorted(order_list, key=lambda order: order.ts)

    # 维护两个队列，unhedged_buy和unhedged_sell
    # 其中只有一个队列不为空
    unhedged_buy = []
    unhedged_sell = []

    # 记录位于队列最前方的那个order的hedged amount
    # 其中只有一个不为0
    hedged_amount_buy = 0.0
    hedged_amount_sell = 0.0

    symbol_type = symbol_info.symbol_type
    # order = OrderInfo(
    #     order_id="", symbol_info=symbol_info, price=0.0, amount=0.0, filled_price=0.0, filled_amount=0.0, 
    #     trade_side=TradeSide.INVALID_SIDE, order_type=OrderType.INVALID_ORDER, tp=0, 
    #     commission = 0.0, trade_type=TradeType.INVALID_TRADE, status=OrderStatus.OPENED)
    for idx in order_index_list:
        # print(idx, order_data_list["filled_amount"][idx])
        # if order_data_list["filled_amount"][idx] <= 0.0:
        #     # 未成交订单不用管
        #     continue
        
        # order = OrderInfo(
        #     order_id=order_data_list["order_id"][idx], symbol_info=symbol_info, price=order_data_list["price"][idx], amount=order_data_list["amount"][idx], 
        #     filled_price=order_data_list["filled_price"][idx], filled_amount=order_data_list["filled_amount"][idx], 
        #     trade_side=TradeSide[order_data_list["side"][idx]], order_type=OrderType[order_data_list["order_type"][idx]], tp=order_data_list["ts"][idx], 
        #     commission=order_data_list["commission"][idx], trade_type=TradeType[order_data_list["liquidity_side"][idx]], status=OrderStatus[order_data_list["status"][idx]].value
        # )

        if order_data_list.filled_amount[idx] <= 0.0:
            # 未成交订单不用管
            continue
        
        order = OrderInfo(
            order_id=order_data_list.order_id[idx], symbol_info=symbol_info, price=order_data_list.price[idx], amount=order_data_list.amount[idx], 
            filled_price=order_data_list.filled_price[idx], filled_amount=order_data_list.filled_amount[idx], 
            trade_side=TradeSide[order_data_list.side[idx]], order_type=OrderType[order_data_list.order_type[idx]], tp=order_data_list.ts[idx], 
            commission=order_data_list.commission[idx], trade_type=TradeType[order_data_list.trade_type[idx]], status=OrderStatus[order_data_list.status[idx]].value
        )

        # print(len(unhedged_buy), len(unhedged_sell), idx, order_data_list["side"][idx], TradeSide.SELL.name)
        if len(unhedged_buy) != 0 and order_data_list.side[idx] == TradeSide.SELL.name:
            hedge_info, unhedged_buy, hedged_amount_buy = _match_one_order(order, unhedged_buy, hedged_amount_buy)
            if len(unhedged_buy) > 0:
                if abs(hedged_amount_buy - unhedged_buy[0].filled_amount) < ZERO_AMOUNT_THRESHOLD:
                    del unhedged_buy[0]
                    hedged_amount_buy = 0.0
            if hedge_info.amount_open + ZERO_AMOUNT_THRESHOLD < hedge_info.amount_close:
                hedged_amount_sell = hedge_info.amount_open
                hedge_info.amount_close = hedge_info.amount_open
                unhedged_sell = [order]
            else:
                hedged_amount_sell = 0.0
                unhedged_sell = []
            hedge_info = _calculate_hedge_metrics(symbol_type, hedge_info)
            _hedge_list.append(hedge_info)
        elif len(unhedged_sell) != 0 and order_data_list.side[idx] == TradeSide.BUY.name:
            hedge_info, unhedged_sell, hedged_amount_sell = _match_one_order(order, unhedged_sell, hedged_amount_sell)
            if len(unhedged_sell) > 0:
                if abs(hedged_amount_sell - unhedged_sell[0].filled_amount) < ZERO_AMOUNT_THRESHOLD:
                    del unhedged_sell[0]
                    hedged_amount_sell = 0.0
            if hedge_info.amount_open + ZERO_AMOUNT_THRESHOLD < hedge_info.amount_close:
                hedged_amount_buy = hedge_info.amount_open
                hedge_info.amount_close = hedge_info.amount_open
                unhedged_buy = [order]
            else:
                hedged_amount_buy = 0.0
                unhedged_buy = []
            hedge_info = _calculate_hedge_metrics(symbol_type, hedge_info)
            _hedge_list.append(hedge_info)
        elif order.side == TradeSide.SELL:
            unhedged_sell.append(order)
        else:
            unhedged_buy.append(order)

    # 把最后剩余的订单单独搞一块
    assert(not(len(unhedged_buy) != 0 and len(unhedged_sell) != 0))
    if len(unhedged_buy) != 0 or len(unhedged_sell) != 0:
        (unhedged, hedged_amount, open_side) = (unhedged_buy, hedged_amount_buy, TradeSide.BUY) if len(unhedged_buy) != 0 else (unhedged_sell, hedged_amount_sell, TradeSide.SELL)
        _hedge_info_left = HedgeInfo(open_side)
        _hedge_info_left.add_open_order_list(unhedged)
        _hedge_info_left.avg_price_open = 0.0
        _hedge_info_left.amount_open = 0.0
        for idx, order in enumerate(unhedged):
            amount = order.filled_amount
            if idx == 0:
                amount -= hedged_amount
            _hedge_info_left.avg_price_open = calc_avg_price(symbol_type, _hedge_info_left.avg_price_open, _hedge_info_left.amount_open, order.filled_price, amount)
            _hedge_info_left.amount_open += amount
        _hedge_info_left.commissions = sum(order.commission for order in unhedged)
        _hedge_info_left.commissions -= unhedged[0].commission * hedged_amount / unhedged[0].filled_amount
        _hedge_info_left.ts_create = unhedged[0].ts
        # 没平掉，暂时不填ts_finish
    
        _hedge_info_left = _calculate_hedge_metrics(symbol_type, _hedge_info_left)
        _hedge_list.append(_hedge_info_left)

    
    print(f'matching_orders _hedge_list: {_hedge_list}')
    
    return _hedge_list


def estimate_returns_for_orders(order_index_list, order_data_list, symbol_info):
    _hedge_list = matching_orders(order_index_list, order_data_list, symbol_info)
    _returns_list = [hedge_info.total_returns for hedge_info in _hedge_list]
    _total_returns = sum(_returns_list)

    return _total_returns, _returns_list, _hedge_list

    
####################################################
# 资产相关
####################################################

def calculate_net_value(balance_list, price_list, start_ts=None, end_ts=None):
    # 假设输入的balance_list和price_list有相同的symbol
    assert len(balance_list.ts_event) != 0 and len(price_list["ts"]) != 0, "Balance list or price list is empty."
    # assert price_list[0].ts_event < balance_list[0].ts_event, "Not enough price to calculate net value"

    # 先把两个list按时间戳排序
    # _balance_list = sorted(balance_list, key=lambda balance: balance.ts_event)
    # _price_list = sorted(price_list, key=lambda price_info: price_info.ts_event)

    # 从balance_list第一个balance之后开始
    if start_ts is None:
        start_ts = balance_list.ts_event[0]
    if end_ts is None:
        end_ts = balance_list.ts_event[-1]

    # 找到在start_ts之后距离最近的price_info作为开始
    # start_idx = min(idx for idx, price_info in enumerate(_price_list) if price_info.ts_event >= start_ts)
    start_idx = min(idx for idx, ts in enumerate(price_list["ts"]) if ts >= start_ts)
    data_size = len(price_list["ts"])
    balance_size = len(balance_list.ts_event)

    # print(f'before calc net value: {start_ts}, {end_ts}, {price_list["ts"][0]}; {data_size}, {balance_size}, {start_idx}')
    
    # start_idx = 0
    # while start_idx < data_size:
    #     if start_idx % 100 == 0:
    #         print(start_idx, price_list["ts"][start_idx])
    #     if price_list["ts"][start_idx] < start_ts:
    #         start_idx += 1
    #     else:
    #         break
    if start_idx >= data_size:
        # 没有时间范围内的价格数据
        return None
        
    # _price_list = _price_list[start_idx:]

    # 同时遍历两个list
    i = 0
    j = start_idx
    price_info = PriceInfo(price=price_list["mid_price"][j], tp=price_list["ts"][j])
    net_value = NetValue(balance_list.trading_symbol, balance_list.get_at(0), price_info)
    # 要返回的净值序列
    net_value_list = NetValueList(balance_list.trading_symbol, balance_list.get_at(0), price_info)

    # print(net_value_list.ts_event)

    now_ts = start_ts
    print(start_idx, data_size, balance_size)
    while (i < balance_size) and (j < data_size):
        balance = balance_list.get_at(i)
        price_info.set_data(price=price_list["mid_price"][j], tp=price_list["ts"][j])
        net_value.update(balance, price_info)
        net_value_list.update(net_value)
        now_ts = net_value.ts

        if now_ts > end_ts:
            break

        # if i % 1000 == 0 or (j > 0 and j % 1000 == 0):
        #     print(i, j, now_ts)
            
        if i + 1 == balance_size:
            # 最后一个balance了，更新j即可
            j += 1
            continue
        if j + 1 == data_size:
            # 最后一个price了，更新i即可
            i += 1
            continue
        
        # 获取下一个balance和price_info的时间
        next_ts_balance = balance_list.ts_event[i + 1]
        next_ts_price = price_list["ts"][j + 1]
        if next_ts_balance > next_ts_price:
            # 下一个balance还没到
            # 先更新price
            j += 1
        elif next_ts_balance < next_ts_price:
            # 下一个price还没到
            # 先更新balance
            i += 1
        else:
            # 两者同时到达
            # 一起更新
            i += 1
            j += 1
    
    return net_value_list

def close_all_positions(current_balance: SymbolBalance, current_price: PriceInfo):
    symbol_type = current_balance.trading_symbol.symbol_type
    if symbol_type == SymbolType.SPOT_NORMAL:
        return close_all_positions_spot(current_balance, current_price)
    elif symbol_type == SymbolType.SWAP_FOREVER:
        return close_all_positions_uswap(current_balance, current_price)
    elif symbol_type == SymbolType.SWAP_COIN_FOREVER:
        return close_all_positions_cswap(current_balance, current_price)
    else:
        print(f"close_all_positions: Not supporting balances for {symbol_type}")
        return None, None

def close_all_positions_spot(current_balance: SymbolBalance, current_price: PriceInfo):
    _balance = duplicate_data(current_balance)
    symbol = current_balance.trading_symbol
    token = symbol.token
    quote = symbol.quote
    position = current_balance.position
    _profit = calc_profit(SymbolType.SPOT_NORMAL, current_balance.holding_price, current_price.price, position)
    _balance.position = 0.0
    _balance.update_capital(token, -position)
    _balance.update_capital(quote, position * current_price.price)
    return _balance, _profit

def close_all_positions_uswap(current_balance: SymbolBalance, current_price: PriceInfo):
    _balance = duplicate_data(current_balance)
    symbol = current_balance.trading_symbol
    token = symbol.token
    quote = symbol.quote
    position = abs(current_balance.position)
    _profit = calc_profit(SymbolType.SWAP_FOREVER, current_balance.holding_price, current_price.price, position)
    _balance.position = 0.0
    _balance.update_capital(quote, position * current_price.price / symbol.lever_rate)
    # _balance.update_capital(quote, position * current_balance.holding_price / symbol.lever_rate)
    return _balance, _profit

def close_all_positions_cswap(current_balance: SymbolBalance, current_price: PriceInfo):
    _balance = duplicate_data(current_balance)
    symbol = current_balance.trading_symbol
    token = symbol.token
    quote = symbol.quote
    position = abs(current_balance.position)
    _profit = calc_profit(SymbolType.SWAP_COIN_FOREVER, current_balance.holding_price, current_price.price, position)
    _balance.position = 0.0
    _balance.update_capital(token, position / current_price.price / symbol.lever_rate)
    # _balance.update_capital(quote, position * current_balance.holding_price / symbol.lever_rate)
    return _balance, _profit

def adjust_balance(initial_balance: SymbolBalance, order_index_list, order_data_list):
    # 假定balance的symbol就是order_list中每个order的symbol
    symbol_info = initial_balance.trading_symbol
    balance_list = BalanceList(initial_balance.trading_symbol, initial_balance.capitals, initial_balance.ts_event)
    # print(f"adjust balance: init {balance_list.ts_event[0]}; {balance_list.capitals['BTC'][0]}; {balance_list.capitals['USDT'][0]}; {len(balance_list.ts_event)}; {len(order_index_list)}; {len(order_data_list.price)}; {len(order_data_list.trade_type)}")
    # for order in order_list:
    order = OrderInfo(
        order_id="", symbol_info=symbol_info, price=0.0, amount=0.0, filled_price=0.0, filled_amount=0.0, 
        trade_side=TradeSide.INVALID_SIDE, order_type=OrderType.INVALID_ORDER, tp=0, 
        commission = 0.0, trade_type=TradeType.INVALID_TRADE, status=OrderStatus.OPENED.value
    )
    cur_balance = duplicate_data(initial_balance)
    counter = 0
    
    # odd_count = np.array([1 for i in order_index_list if order_data_list["filled_amount"][i] > 0]).sum()
    # print(f'before adjust: {odd_count}')

    for idx in order_index_list:
        # if order_data_list["filled_amount"][idx] > 0:
        #     order.set_data(
        #         order_id=order_data_list["order_id"][idx], price=order_data_list["price"][idx], amount=order_data_list["amount"][idx], 
        #         filled_price=order_data_list["filled_price"][idx], filled_amount=order_data_list["filled_amount"][idx], 
        #         trade_side=TradeSide[order_data_list["side"][idx]], order_type=OrderType[order_data_list["order_type"][idx]], tp=order_data_list["ts"][idx], 
        #         commission=order_data_list["commission"][idx], trade_type=TradeType[order_data_list["liquidity_side"][idx]], status=OrderStatus[order_data_list["status"][idx]].value
        #     )        
        if order_data_list.filled_amount[idx] > 0:
            if counter < 300:
                last_balance = balance_list.get_at(-1)
                # print(f'adjust balance: {counter}, {idx} {order_data_list.status[idx]} {order_data_list.ts[idx]}\n{cur_balance.capitals["BTC"]} btc + {cur_balance.capitals["USDT"]} vs {last_balance.capitals["BTC"]} btc + {last_balance.capitals["USDT"]}')
            counter += 1

            order.set_data(
                order_id=order_data_list.order_id[idx], price=order_data_list.price[idx], amount=order_data_list.amount[idx], 
                filled_price=order_data_list.filled_price[idx], 
                filled_amount=order_data_list.filled_amount[idx], 
                trade_side=TradeSide[order_data_list.side[idx]], 
                order_type=OrderType[order_data_list.order_type[idx]], 
                tp=order_data_list.ts[idx], 
                commission=order_data_list.commission[idx], trade_type=TradeType[order_data_list.trade_type[idx]], status=OrderStatus[order_data_list.status[idx]].value
            )

            cur_balance = adjust_balance_by_order(cur_balance, order)

        balance_list.update(cur_balance.capitals, cur_balance.position, cur_balance.holding_price, cur_balance.ts_event)
    return balance_list

def adjust_balance_by_order(balance: SymbolBalance, order: OrderInfo):
    # todo: 根据实际定义，填写symbol_type
    _symbol_type = order.symbol_info.symbol_type
    _balance = balance
    if _symbol_type == SymbolType.SPOT_NORMAL:
        _balence = adjust_balance_for_SPOT(_balance, order)
    elif _symbol_type == SymbolType.SWAP_FOREVER:
        _balance = adjust_balance_for_USWAP(_balance, order)
    elif _symbol_type == SymbolType.SWAP_COIN_FOREVER:
        _balance = adjust_balance_for_CSWAP(_balance, order)
    return _balance


def adjust_balance_for_SPOT(balance: SymbolBalance, order: OrderInfo):
    # 检查order和balance是否有相同的SymbolInfo
    assert order.symbol_info == balance.trading_symbol, \
        f"Can't update balance of '{balance.trading_symbol}' using orders of '{order.symbol_info}'"
    
    # 调整现货资产
    _balance = balance

    # 更新时间戳
    if order.ts == 0:
        print(f"Order with ts == 0: {order}")
    _balance.ts_event = order.ts
    
    # 先看看这个订单在不在balance的统计范围里
    token = order.symbol_info.token
    quote = order.symbol_info.quote
    if not token in _balance.capitals or not quote in _balance.capitals:
        # 如果订单某一方不在balance里
        # 那么就不统计这个订单
        not_found = token if not token in _balance.capitals else quote
        print(f"Not adjusting balance for order {order.order_id}, since `{not_found}` is not found.")
        return _balance

    # 未成交订单不改变balance
    if order.filled_amount == 0:
        return _balance

    # 获取订单方向
    if order.side == TradeSide.INVALID_SIDE:
        # 如果订单TradeSide无效，则不更新balance
        print(f"Not adjusting balance for order {order.order_id}, since its side is invalid.")
        return _balance
    sign = +1 if order.side == TradeSide.BUY else -1

    # 更新平均持仓价格
    # 现货，所以只有买才会改变平均持仓价格
    if order.side == TradeSide.BUY:
        _balance.holding_price = calc_avg_price(SymbolType.SPOT_NORMAL, _balance.holding_price, _balance.position, order.filled_price, order.filled_amount)
    
    # 更新仓位
    _balance.position += sign * order.filled_amount

    if abs(_balance.position) < ZERO_AMOUNT_THRESHOLD:
        _balance.position = 0.0
        _balance.holding_price = 0.0
    
    # TODO: 如果order.commission == 0，根据费率计算；需要设定commission的币种

    # 更新token和quote的持有量
    _fee = calc_commission(order)
    order.commission = _fee
    # order.commission = 0
    _balance.update_capital(token, sign * order.filled_amount)
    _balance.update_capital(quote, - sign * order.filled_amount * order.filled_price - _fee)
    # _balance.update_capital(quote, - sign * order.filled_amount * order.filled_price)
    
    return _balance


def adjust_balance_for_USWAP(balance: SymbolBalance, order: OrderInfo):
    # 检查order和balance是否有相同的SymbolInfo
    assert order.symbol_info == balance.trading_symbol, \
        f"Can't update balance of '{balance.trading_symbol}' using orders of '{order.symbol_info}'"
    
    # 调整U本位永续合约资产
    _balance = balance
    
    # 更新时间戳
    if order.ts == 0:
        print(f"Order with ts == 0: {order}")
    _balance.ts_event = order.ts

    # 先看看这个订单在不在balance的统计范围里
    token = order.symbol_info.token
    quote = order.symbol_info.quote
    if not token in _balance.capitals or not quote in _balance.capitals:
        # 如果订单某一方不在balance里
        # 那么就不统计这个订单
        not_found = token if not token in _balance.capitals else quote
        print(f"Not adjusting balance for order {order.order_id}, since `{not_found}` is not found.")
        return _balance

    # 未成交订单不改变balance
    if order.filled_amount == 0:
        return _balance

    # 判断是加仓还是平仓
    # 只要order与balance的多空方向一致，那就是加仓，反之为平仓
    sign = +1 if order.side == TradeSide.BUY else -1
    is_close = _balance.position * sign < 0

    _fee = calc_commission(order)
    order.commission = _fee # 这样之后不用再算了
    _balance.update_capital(quote, -_fee)
    
    if is_close: # 先看是否平仓
        _amount = min(abs(_balance.position), order.filled_amount) # 平掉的量
        _balance.position += sign * _amount
        _init_margin = _amount * _balance.holding_price / _balance.trading_symbol.lever_rate # 初始保证金
        # _profit = sign * _amount * (order.filled_price - _balance.holding_price) # 利润
        _profit = calc_profit(SymbolType.SWAP_FOREVER, _balance.holding_price, order.filled_price, -sign * _amount)
        # _profit = -sign * _amount * (order.filled_price - _balance.holding_price) # 利润
        _balance.update_capital(quote, _init_margin + _profit)
        _amount_open = order.filled_amount - _amount
    else:
        _amount_open = order.filled_amount

    if abs(_balance.position) < ZERO_AMOUNT_THRESHOLD:
        _balance.position = 0.0
        _balance.holding_price = 0.0
    
    # 处理开仓
    if _amount_open > 0:
        _init_margin = _amount_open * order.filled_price / _balance.trading_symbol.lever_rate # 初始保证金
        _balance.holding_price = calc_avg_price(SymbolType.SWAP_FOREVER, _balance.holding_price, abs(_balance.position), order.filled_price, _amount_open)
        _balance.update_capital(quote, -_init_margin)
        _balance.position += sign * _amount_open
    
    return _balance

def adjust_balance_for_CSWAP(balance: SymbolBalance, order: OrderInfo):
    # 检查order和balance是否有相同的SymbolInfo
    assert order.symbol_info == balance.trading_symbol, \
        f"Can't update balance of '{balance.trading_symbol}' using orders of '{order.symbol_info}'"
    
    # 调整币本位永续合约资产
    _balance = balance
    
    # 更新时间戳
    if order.ts == 0:
        print(f"Order with ts == 0: {order}")
    _balance.ts_event = order.ts

    # 先看看这个订单在不在balance的统计范围里
    token = order.symbol_info.token
    quote = order.symbol_info.quote
    if not token in _balance.capitals or not quote in _balance.capitals:
        # 如果订单某一方不在balance里
        # 那么就不统计这个订单
        not_found = token if not token in _balance.capitals else quote
        print(f"Not adjusting balance for order {order.order_id}, since `{not_found}` is not found.")
        return _balance

    # 未成交订单不改变balance
    if order.filled_amount == 0:
        return _balance

    # 判断是加仓还是平仓
    # 只要order与balance的多空方向一致，那就是加仓，反之为平仓
    sign = +1 if order.side == TradeSide.BUY else -1
    is_close = _balance.position * sign < 0

    _fee = calc_commission(order)
    order.commission = _fee # 这样之后不用再算了
    _balance.update_capital(token, -_fee)
    
    if is_close: # 先看是否平仓
        _amount = min(abs(_balance.position), order.filled_amount) # 平掉的量
        _balance.position += sign * _amount
        _init_margin = _amount / _balance.holding_price / _balance.trading_symbol.lever_rate # 初始保证金
        # _profit = sign * _amount * (order.filled_price - _balance.holding_price) # 利润
        _profit = calc_profit(SymbolType.SWAP_COIN_FOREVER, _balance.holding_price, order.filled_price, -sign * _amount)
        # _profit = -sign * _amount * (1.0 / _balance.holding_price - 1.0 / order.filled_price) # 利润
        _balance.update_capital(token, _init_margin + _profit)
        _amount_open = order.filled_amount - _amount
    else:
        _amount_open = order.filled_amount

    if abs(_balance.position) < ZERO_AMOUNT_THRESHOLD:
        _balance.position = 0.0
        _balance.holding_price = 0.0
    
    # 处理开仓
    if _amount_open > 0:
        _init_margin = _amount_open / order.filled_price / _balance.trading_symbol.lever_rate # 初始保证金
        _balance.holding_price = calc_avg_price(SymbolType.SWAP_COIN_FOREVER, _balance.holding_price, abs(_balance.position), order.filled_price, _amount_open)
        _balance.update_capital(token, -_init_margin)
        _balance.position += sign * _amount_open
    
    return _balance

if __name__ == "__main__":
    # 测试ts_to_millisecond
    print(ts_to_millisecond(1721451450000000000))
    print(ts_to_millisecond(1721451450000000))
    print(ts_to_millisecond(1721451450000))
    print(ts_to_millisecond(1721451450))
