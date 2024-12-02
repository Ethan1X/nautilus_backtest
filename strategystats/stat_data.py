from enum import Enum
from copy import deepcopy
import pickle

from collections import OrderedDict

ZERO_AMOUNT_THRESHOLD = 1e-10


def duplicate_data(data):
    # return deepcopy(data)
    return pickle.loads(pickle.dumps(data))


def ts_to_millisecond(ts):
    _ts = int(ts)
    if ts > 1e16: # 纳秒
        _ts = _ts // int(1e6)
    elif ts > 1e13: # 微秒
        _ts = _ts // int(1e3)
    elif ts < 1e11: # 秒
        _ts = _ts * int(1e3)
    return _ts


def ts_to_second(ts):
    _ts = int(ts)
    if ts > 1e16: # 纳秒
        _ts = _ts // int(1e9)
    elif ts > 1e13: # 微秒
        _ts = _ts // int(1e6)
    elif ts > 1e10: # 毫秒
        _ts = _ts // int(1e3)
    return _ts


def calc_commission(order) -> float:
    if order.commission != 0.0:
        return order.commission

    if order.filled_amount == 0.0: # 没成交，无手续费
        return 0.0
    
    commission = 0.0

    if order.symbol_info.symbol_type == SymbolType.SWAP_COIN_FOREVER:
        price = 1.0 / order.filled_price
    else: # 目前只支持现货、U本位永续和币本位永续
        price = order.filled_price
    
    if order.trade_type == TradeType.MAKER:
        commission = order.filled_amount * price * order.symbol_info.commission_rate_maker
    elif order.trade_type == TradeType.TAKER:
        commission = order.filled_amount * price * order.symbol_info.commission_rate_taker
    else:
        print(f"calc_commission: Order {order.order_id} has invalid trade type. ({order})")
        
    return commission


def calc_avg_price(symbol_type, price, amount, delta_price, delta_amount):
    avg_price = 0.0
    if symbol_type == SymbolType.SPOT_NORMAL or symbol_type == SymbolType.SWAP_FOREVER:
        if amount + delta_amount > 0:
            avg_price = (amount * price + delta_amount * delta_price) / (amount + delta_amount)
    elif symbol_type == SymbolType.SWAP_COIN_FOREVER:
        if amount + delta_amount > 0:
            if price > 0:
                avg_price = (amount + delta_amount) / (amount / price + delta_amount / delta_price)
            else:
                avg_price = delta_price
    else:
        print(f"calc_avg_price: Not supporting symbol type {symbol_type.name}")
    return avg_price


def calc_profit(symbol_type, holding_price, market_price, position):
    profit = 0.0
    if symbol_type == SymbolType.SPOT_NORMAL or symbol_type == SymbolType.SWAP_FOREVER:
        profit = position * (market_price - holding_price)
    elif symbol_type == SymbolType.SWAP_COIN_FOREVER:
        if market_price > 0 and holding_price > 0:
            profit = position * (1.0 / market_price - 1.0 / holding_price)
    else:
        print(f"calc_profit: Not supporting symbol type {symbol_type.name}")
    return profit


class SymbolType(Enum):
    INVALID_TYPE = 0
    SPOT_NORMAL = 1 # 现货

    SWAP_COIN_FOREVER = 1001 # 币本位永续合约
    SWAP_FOREVER = 1002 # U本位永续合约

    def __repr__(self):
        return self.name


class TradeSide(Enum):
    INVALID_SIDE = 0
    BUY = 1
    SELL = 2

    def __repr__(self):
        return self.name


class TradeType(Enum):
    INVALID_TRADE = 0
    TAKER = 1
    MAKER = 2

    def __repr__(self):
        return self.name


class OrderType(Enum):
    INVALID_ORDER = 0
    LIMIT = 1
    MARKET = 2

    def __repr__(self):
        return self.name

class OrderStatus(Enum):
    INVALID_STATUS = 0
    
    # NOTE: try to avoid conflict with proto
    SUBMITTED = 1
    DENIED = 2
    #####
    
    OPENED = 3
    PARTLIFILLED = 4
    FILLED = 7
    CANCELED = 8
    EXPIRED = 9
    REJECTED = 10

    
    def __repr__(self):
        return self.name

def convert_order_status(status):
    _status = OrderStatus.INVALID_STATUS

    _status_name = status.name
    if _status_name == "ACCEPTED":
        _status = OrderStatus.OPENED
    elif _status_name == "PARTIALLY_FILLED":
        _status = OrderStatus.PARTLIFILLED
    elif _status_name in ["DENIED", "SUBMITTED", "REJECTED", "CANCELED", "EXPIRED", "FILLED"]:
        _status = OrderStatus[status.name]
    
    return _status


class Market(Enum):
    INVALID_MARKET = 0
    SPOT = 1
    FUTURE = 2
    SWAP = 3
    MARGIN = 4
    OPTION = 5
    OTC = 6
    WALLET = 7
    FUTURE_COIN = 8
    SWAP_COIN = 9

    def __repr__(self):
        return self.name


####################################################
# 定义统计中必要的数据结构
####################################################


####################################################
# 资产数据
####################################################

# 市场信息
class MarketInfo(object):
    def __init__(self, market, exchange):
        self.market: Market = market
        self.exchange: str = exchange
    
    def to_dict(self):
        return {
            "market": self.market.name,
            "exchange": self.exchange,
        }
    
    def __eq__(self, other):
        if isinstance(other, MarketInfo):
            if self.market == other.market and self.exchange == other.exchange:
                return True
        return False
    
    def __repr__(self):
        return str(self.__dict__)


# 交易对信息
class SymbolInfo(object):
    def __init__(self, token, quote, symbol_type, market, exchange, commission_rate_maker, commission_rate_taker, lever_rate=1.0):
        self.token: str = token    # 基础币
        self.quote: str = quote    # 计价币
        self.symbol_type: SymbolType = symbol_type    # 交易品种
        self.market_info: MarketInfo = MarketInfo(market, exchange)
        self.commission_rate_maker: float = commission_rate_maker     # maker费率
        self.commission_rate_taker: float = commission_rate_taker     # taker费率
        self.lever_rate: float = 1.0 # 杠杆率

    def get_symbol_name(self):
        return f'{self.token}_{self.quote}'

    @staticmethod
    def from_dict(d: dict):
        return SymbolInfo(
            token=d["token"],
            quote=d["quote"],
            symbol_type=SymbolType[d["symbol_type"]],
            market=Market[d["market_info"]["market"]],
            exchange=d["market_info"]["exchange"],
            commission_rate_maker=d["commission_rate_maker"],
            commission_rate_taker=d["commission_rate_taker"],
            lever_rate=d.get("lever_rate", 1.0),
        )
    
    def to_dict(self):
        return {
            "token": self.token,
            "quote": self.quote,
            "symbol_type": self.symbol_type.name,
            "market_info": self.market_info.to_dict(),
            "commission_rate_maker": self.commission_rate_maker,
            "commission_rate_taker": self.commission_rate_taker,
            "lever_rate": self.lever_rate,
        }
    
    def __eq__(self, other):
        if isinstance(other, SymbolInfo):
            if self.token == other.token and \
                self.quote == other.quote and \
                self.symbol_type == other.symbol_type and \
                self.market_info == other.market_info and \
                self.commission_rate_maker == other.commission_rate_maker and \
                self.commission_rate_taker == other.commission_rate_taker and \
                self.lever_rate == other.lever_rate:
                return True
        return False
            
    def __repr__(self):
        return str(self.__dict__)
        

# 资产信息
class SymbolBalance(object):
    def __init__(self, symbol: SymbolInfo, capitals, tp):
        self.ts_event = ts_to_millisecond(tp)
        
        self.capitals = duplicate_data(capitals)   # {token: 资金持有量}
        self.trading_symbol: SymbolInfo = symbol    # 交易对信息
        self.position: float = 0.0   # 持仓
        self.holding_price: float = 0.0   # 平均持仓价格

    def update_capital(self, token, amount):
        if token in self.capitals:
            self.capitals[token] += amount
    
    def set_capital(self, token, amount):
        if len(token) > 0:
            _amount = amount if amount > 0 else 0.0
            self.capitals[token] = _amount

    def __repr__(self):
        return str(self.__dict__)


class BalanceList(object):
    def __init__(self, symbol: SymbolInfo, capitals, tp):
        self.trading_symbol: SymbolInfo = symbol
        self.ts_event = [ts_to_millisecond(tp)]
        self.capitals = {k:[v] for k,v in capitals.items()}
        self.position = [0.0]
        self.holding_price = [0.0]

    def update(self, capitals, position, holding_price, tp):
        self.ts_event.append(tp)
        for k in self.capitals.keys():
            if k in capitals.keys():
                self.capitals[k].append(capitals[k])
            else:
                self.capitals[k].append(0.0)
        self.position.append(position)
        self.holding_price.append(holding_price)

    def get_at(self, idx):
        _capitals = {k:self.capitals[k][idx] for k in self.capitals.keys()}
        _balance = SymbolBalance(self.trading_symbol, _capitals, self.ts_event[idx])
        _balance.position = self.position[idx]
        _balance.holding_price = self.holding_price[idx]
        return _balance


# 订单数据
class OrderInfo(object):
    ts: int = 0.0

    def __init__(self, order_id, symbol_info, price, amount, filled_price, filled_amount, trade_side, order_type, tp, commission = 0.0, trade_type=TradeType.INVALID_TRADE, status=OrderStatus.OPENED):
        self.order_id: str = order_id    # 订单id
        self.client_order_id: str = order_id    # 订单id
        self.symbol_info: SymbolInfo = symbol_info    # 交易对信息
    
        self.status: OrderStatus = status    # 订单状态
        
        self.price: float = price    # 订单报价
        self.amount: float = amount  # 订单量
        self.side: TradeSide = trade_side   # 方向：买/卖
        self.trade_type: TradeType = trade_type    # Taker/Maker
        self.order_type: OrderType = order_type    # Limit/Market
        self.filled_price: float = filled_price    # 成交价格
        self.filled_amount: float = filled_amount  # 成交量
        self.commission: float = commission   # 手续费：暂用quote值 todo：如无，可根据费率自行计算

        self.ts = ts_to_millisecond(tp)       # 订单时间戳：转换成毫秒进行保留 

    def set_data(self, order_id=None, symbol_info=None, price=None, amount=None, filled_price=None, filled_amount=None, 
                 trade_side=None, order_type=None, tp=None, commission=None, trade_type=None, status=None):
        if order_id:
            self.order_id = order_id    # 订单id
        if symbol_info:
            self.symbol_info = symbol_info    # 交易对信息

        if status:
            self.status = status    # 订单状态

        if price:
            self.price = price    # 订单报价
        if amount:
            self.amount = amount  # 订单量
        if trade_side:
            self.side = trade_side   # 方向：买/卖
        if trade_type:
            self.trade_type = trade_type    # Taker/Maker
        if order_type:
            self.order_type = order_type    # Limit/Market
        if filled_price:
            self.filled_price = filled_price    # 成交价格
        if filled_amount:
            self.filled_amount = filled_amount  # 成交量
        if commission:
            self.commission = commission   # 手续费：暂用quote值 todo：如无，可根据费率自行计算

        if tp:
            self.ts = ts_to_millisecond(tp)       # 订单时间戳：转换成毫秒进行保留 

    def __repr__(self):
        return str(self.__dict__)


class OrderList(object):
    def __init__(self, symbol_info: SymbolInfo=None):
        self.order_id = []    # 订单id
        self.symbol_info = symbol_info    # 交易对信息
    
        self.status = []    # 订单状态
        
        self.price = []    # 订单报价
        self.amount = []  # 订单量
        self.side = []   # 方向：买/卖
        # self.liquidity_side = []
        self.trade_type = []    # Taker/Maker
        self.order_type = []    # Limit/Market
        self.filled_price = []    # 成交价格
        self.filled_amount = []  # 成交量
        self.commission = []   # 手续费：暂用quote值 todo：如无，可根据费率自行计算
        self.index: OrderedDict = OrderedDict()

        self.ts = []       # 订单时间戳：转换成毫秒进行保留 

    def load_npz(self, order_npz):
        self.order_id = list(order_npz["order_id"])
        self.status = list(order_npz["status"])
        self.price = list(order_npz["price"])
        self.amount = list(order_npz["amount"])
        self.side = list(order_npz["side"])
        # self.liquidity_side = list(order_npz.get("liquidity_side",[]))
        self.trade_type = list(order_npz.get("trade_type", []))
        self.order_type = list(order_npz["order_type"])
        self.filled_price = list(order_npz["filled_price"])
        self.filled_amount = list(order_npz["filled_amount"])
        self.commission = list(order_npz["commission"])
        _unique_order = order_npz.get("unique_order")
        _unique_index = order_npz.get("unique_index")
        if _unique_order is not None and _unique_index is not None:
            self.index = OrderedDict(zip(_unique_order, _unique_index))
        else:
            self.index = OrderedDict()
            self.build_index()
        self.ts = order_npz["ts"]

    def build_index(self):
        # self.index = OrderedDict((k,i) for i, k in enumerate(self.order_id))
        for i, k in enumerate(self.order_id):
            if k in self.index.keys():
                del self.index[k]
            self.index[k] = i

    # def register(self, order, symbol_info, is_maker=True):
    #     if order is not None and (symbol_info is None or self.symbol_info is None or symbol_info == self.symbol_info):
    #         if order.side == OrderSide.BUY:
    #         if int(order.side) == 1:
    #             trade_side = TradeSide.BUY
    #         elif order.side == OrderSide.SELL:
    #             trade_side = TradeSide.SELL
    #         else:
    #             print(f"Currently not supporting orders of side '{order.side}'!")
    #             trade_side = TradeSide.INVALID_SIDE

    #         price = 0.0
    #         if isinstance(order, LimitOrder):
    #             order_type = OrderType.LIMIT
    #             price = float(order.price)
    #         elif isinstance(order, MarketOrder):
    #             order_type = OrderType.MARKET
    #         else:
    #             print(f"Currently not supporting orders of type '{type(order)}'!")
    #             order_type = OrderType.INVALID_ORDER

    #         if is_maker is None:
    #             trade_type = TradeType.INVALID_TRADE
    #         else:
    #             trade_type = TradeType.MAKER if is_maker else TradeType.TAKER

    #         self._append(order_id=order.client_order_id, status=order.status, price=float(order.price), amount=float(order.quantity), 
    #                      side=trade_side, trade_type=trade_type, order_type=order_type, 
    #                      filled_price=0.0, filled_amount=0.0, commission=0.0, ts=order.ts_init)
    #         self.index[order_id] = len(self.order_id) - 1

    # def update(self, event):
    #     id = str(event.client_order_id)
    #     if id not in self.index:
    #         print(f"Unregistered order '{id}'!")
    #         return None

    #     idx = self.index.get(id)
    #     status = OrderStatus[self.status[idx]]
    #     ts = ts_to_millisecond(event.ts_event)
    #     trade_type = self.trade_type[idx]
    #     filled_price = self.filled_price[idx]
    #     filled_amount = self.filled_amount[idx]
    #     commission = self.commission[idx]
        
    #     if isinstance(event, OrderSubmitted):
    #         status = OrderStatus.OPENED
    #     elif isinstance(event, OrderFilled):
    #         price = float(event.last_px)
    #         amount = float(event.last_qty)
    #         filled_price = (filled_price * filled_amount + price * amount) / (filled_amount + amount)
    #         # calc_filled_price = (order_info.filled_price * order_info.filled_amount + price * amount) / (order_info.filled_amount + amount)
    #         # calc_filled_amount = order_info.filled_amount + amount
    #         filled_amount += amount
    #         # order_info.filled_price = float(order.avg_px)
    #         # order_info.filled_amount = float(order.filled_qty)
    #         # if calc_filled_price != order_info.filled_price:
    #         #     print(f"Calculated filled price: {calc_filled_price}, Nautilus avg price: {float(order.avg_px)}")
    #         #     print(f"Calculated filled amount: {calc_filled_amount}, Nautilus filled amount: {float(order.filled_qty)}")
                
    #         if filled_amount == self.amount[idx]:
    #             status = OrderStatus.FILLED
    #         else:
    #             status = OrderStatus.PARTLIFILLED
    #         # TODO: order.commissions()是不同币种计价的相同的累计commission，目前只有USDT计价的commission，所以还没出问题
    #         commission = float(sum(order.commissions())) # 假定每次fill的commission用相同Currency
    #     elif isinstance(event, OrderCanceled):
    #         status = OrderStatus.CANCELED
    #     elif isinstance(event, OrderExpired):
    #         status = OrderStatus.EXPIRED
    #     else:
    #         print(f"Currently not supporting order events of type '{type(event)}'!")

    #     self.append(order_id=order_id, status=status, price=self.price[idx], amount=self.amount[idx], 
    #                      side=TradeSide[self.trade_side[idx]], trade_type=trade_type, order_type=self.order_type[idx], 
    #                      filled_price=filled_price, filled_amount=filled_amount, commission=commission, ts=ts)

    def append(self, order_id, status, price, amount, side, trade_type, order_type, filled_price, filled_amount, commission, ts):
        self._append(order_id, status, price, amount, side, trade_type, order_type, filled_price, filled_amount, commission, ts)
        
    def _append(self, order_id, status, price, amount, side, trade_type, order_type, filled_price, filled_amount, commission, ts):
            self.order_id.append(order_id)
            self.status.append(status.name)
            self.price.append(price)
            self.amount.append(amount)
            self.side.append(side.name)
            self.trade_type.append(trade_type.name)
            self.order_type.append(order_type.name)
            self.filled_price.append(filled_price)
            self.filled_amount.append(filled_amount)
            self.commission.append(commission)
            self.ts.append(ts_to_millisecond(ts))
            if order_id in self.index:
                del self.index[order_id]
            self.index[order_id] = len(self.order_id) - 1   

    def get(self, order_id):
        _idx = self.index.get(order_id)
        if _idx:
            return self.get_at(_idx)
        else:
            return None
        
    def get_at(self, idx):
        return OrderInfo(
                    order_id=self.order_id[idx], 
                    symbol_info=self.symbol_info, 
                    price=self.price[idx], 
                    amount=self.amount[idx], 
                    filled_price=self.filled_price[idx], 
                    filled_amount=self.filled_amount[idx], 
                    trade_side=TradeSide[self.side[idx]].value, 
                    order_type=OrderType[self.order_type[idx]].value, 
                    tp=self.ts[idx], 
                    commission=self.commission[idx], 
                    trade_type=TradeType[self.trade_type[idx]].value, 
                    status=OrderStatus[self.status[idx]].value
                )

    def to_dict(self):
        attr_dict = {k: v for k, v in vars(self).items() if k not in ['symbol_info', 'index', '__dict__']}
        attr_dict['unique_order'] = list(self.index.keys())
        attr_dict['unique_index'] = list(self.index.values())
        return attr_dict
        

# 订单匹配数据
class HedgeInfo():

    amount_open: float = 0.0   # 开仓成交量
    amount_close: float = 0.0   # 关仓成交量
    open_side: TradeSide         # 开仓方向：买/卖
    avg_price_open: float = 0.0    # 平均开仓成交价
    avg_price_close: float = 0.0   # 平均关仓成交价
    total_value: float = 0.0   # 总成交价值，用于计算换手率
    total_returns: float = 0.0  # 总收益
    commissions: float = 0.0    # 总费用

    ts_create: int = 0    # 开仓成功的时间戳（毫秒）
    ts_finish: int = 0    # 关仓时间时间戳（毫秒）
 
    def __init__(self, open_side: TradeSide):
        self.open_side = open_side
        self.open_order_id = []
        self.close_order_id = []
        
    def add_open_order_list(self, order_list):
        if order_list is not None and len(order_list) > 0:
            self.open_order_id = [order.order_id for order in order_list]
            
    def add_close_order_list(self, order_list):
        if order_list is not None and len(order_list) > 0:
            self.close_order_id = [order.order_id for order in order_list]

    def __repr__(self):
        return str(self.__dict__)


class SignalInfo():
    def __init__(self, ts: int, price: float, amount: float, name: str):
        self.ts = ts_to_millisecond(ts)
        self.price = float(price)
        self.amount = float(amount)
        self.name = name

####################################################
# 行情数据
####################################################

# 价格信息
class PriceInfo(object): 
    def __init__(self, price=0.0, amount=0.0, tp=0.0):
        self.price: float = price
        self.amount: float = amount
        self.ts_event = tp

    def set_data(self, price=None, amount=None, tp=None):
        if price:
            self.price = price
        if amount:
            self.amount = amount
        if tp:
            self.ts_event = tp
        
    def __repr__(self):
        return str(self.__dict__)


# 市场价格
# 调用update更新并获取最新的市场中间价
# 或者调用get_mid_price获取当前的市场中间价
class MarketPriceInfo(object):
    price_type: str = "ticker"   # "ticker" for now
    ts_event: int = 0

    def __init__(self, ask_price=0.0, ask_amount=0.0, bid_price=0.0, bid_amount=0.0, mid_price=0.0, tp=0):
        tp = ts_to_millisecond(tp)
        self.ask_info = PriceInfo(ask_price, ask_amount, tp)
        self.bid_info = PriceInfo(bid_price, bid_amount, tp)
        if mid_price != 0:
            self._mid_price = PriceInfo(mid_price, 0.0, tp)
        else:
            self._mid_price = PriceInfo(tp=tp)
        self.ts_event = tp
    
    def update(self, ask_price=0.0, ask_amount=0.0, bid_price=0.0, bid_amount=0.0, tp=0):
        tp = ts_to_millisecond(tp)
        if ask_price > 0:
            self.ask_info = self.set_ask(ask_price, ask_amount, tp)
        if bid_price > 0:
            self.bid_info = self.set_bid(bid_price, bid_amount, tp)
        self.ts_event = tp
        self._mid_price.ts_event = tp
        return duplicate_data(self._mid_price)

    @property
    def mid_price(self):
        if self._mid_price.price != 0:
            return duplicate_data(self._mid_price)
        if self.ask_info.price == 0 or self.bid_info.price == 0:
            _mid = self.ask_info.price + self.bid_info.price
        else:
            _mid = (self.ask_info.price + self.bid_info.price) / 2
        self._mid_price.price = _mid
        return duplicate_data(self._mid_price)

    # def set_ask(self, ask_price, ask_amount, tp):
    #     if ask_price > 0:
    #         self.ask_info.ask_price = ask_price
    #         self.ask_info.ask_amount = ask_amount
    #         self.ts_event = tp
    #         self.get_mid_price()

    # def set_bid(self, bid_price, bid_amount, tp):
    #     if bid_price > 0:
    #         self.bid_info.bid_price = bid_price
    #         self.bid_info.bid_amount = bid_amount
    #         self.ts_event = tp
    #         self.get_mid_price()

    def get_ask(self):
        return PriceInfo(self.ask_info.price, self.ask_info.amount, self.ask_info.ts_event)

    def get_bid(self):
        return PriceInfo(self.bid_info.price, self.bid_info.amount, self.bid_info.ts_event)
        
    def __repr__(self):
        return str(self.__dict__)


# 净值
class NetValue(object):
    def __init__(self, symbol: SymbolInfo, initial_balance: SymbolBalance, initial_price: PriceInfo):
        self.symbol = symbol
        self._net_value = 0.0
        self._ts = -1
        if self.symbol.symbol_type == SymbolType.SWAP_COIN_FOREVER:
            self.quote = self.symbol.token # 对币本位永续暂时先这么记
        else: # 目前只支持现货、U本位永续和币本位永续
            self.quote = self.symbol.quote
        self.update(initial_balance, initial_price)

    def update(self, balance, price_info):
        self._ts = max(balance.ts_event, price_info.ts_event)
        if self.symbol.symbol_type == SymbolType.SPOT_NORMAL:
            self._update_spot(balance, price_info)
        elif self.symbol.symbol_type == SymbolType.SWAP_FOREVER:
            self._update_uswap(balance, price_info)
        elif self.symbol.symbol_type == SymbolType.SWAP_COIN_FOREVER:
            self._update_cswap(balance, price_info)
        else:
            print(f"NetValue.update: Not supporting updating balance for {self.symbol.symbol_type}")

    def _update_spot(self, balance, price_info):
        self._net_value = balance.position * price_info.price + balance.capitals[self.quote] # 假定quote是USDT

    def _update_uswap(self, balance, price_info):
        # self._net_value = abs(balance.position) * price_info.mid_price.price + balance.capitals[self.symbol.quote] # 假定quote是USDT
        net_value = abs(balance.position) * balance.holding_price / self.symbol.lever_rate + balance.capitals[self.quote] # 假定quote是USDT
        self._net_value = net_value + calc_profit(SymbolType.SWAP_FOREVER, balance.holding_price, price_info.price, balance.position)

    def _update_cswap(self, balance, price_info):
        net_value = balance.capitals[self.quote]
        if balance.holding_price > 0:
            net_value += abs(balance.position) / balance.holding_price / self.symbol.lever_rate
        self._net_value = net_value + calc_profit(SymbolType.SWAP_COIN_FOREVER, balance.holding_price, price_info.price, balance.position)
    
    @property
    def net_value(self):
        return self._net_value

    @property
    def ts(self):
        return self._ts

    def __repr__(self):
        return str(self.__dict__)


class NetValueList(object):
    def __init__(self, symbol: SymbolInfo, initial_balance: SymbolBalance, initial_price: PriceInfo):
        self.symbol = symbol
        self.net_values = []
        self.ts_event = []
        if self.symbol.symbol_type == SymbolType.SWAP_COIN_FOREVER:
            self.quote = self.symbol.token # 对币本位永续暂时先这么记
        else: # 目前只支持现货、U本位永续和币本位永续
            self.quote = self.symbol.quote
        _net_value = NetValue(symbol, initial_balance, initial_price)
        self.update(_net_value)

    def update(self, net_value: NetValue):
        self.net_values.append(net_value.net_value)
        self.ts_event.append(net_value.ts)

    def get_at(self, idx):
        return self.net_values[idx], self.ts_event[idx]


####################################################
# 统计数据
####################################################

class StatisticInfo(object):
    def __init__(self, symbol_info, balance):
        # self.ts_event: int = 0
        
        # self.symbol_info: SymbolInfo = symbol_info
    
        # self.balance: SymbolBalance = balance
        
        self.total_returns: float = 0.0 # 总收益-USDT-不含手续费
        self.total_returns_rate: float = 0.0 # 不含手续费的收益率-%
        self.total_commissions: float = 0.0 # 总手续费-USDT
        self.total_trading_amount: float = 0.0 # 
        self.total_trading_value: float = 0.0 # 总交易额-quote
        self.trading_days: float = 0.0
        self.trading_counts: float = 0.0  # 总交易次数-非对冲组
        self.daily_trading_counts: float = 0.0  # 日度交易次数  
        self.daily_win_rate: float = 0.0  # 日度胜率
    
        # todo: 常用统计数据
        self.turnover_rate: float = 0.0   # 换手率
        self.daily_turnover_rate: float = 0.0   # 日度换手率
        self.annual_returns: float = 0.0  # 年化收益率
        self.sortino_ratio: float = 0.0  # 下行风险比率
        self.sharpe_ratio: float = 0.0  # 夏普比率
        self.calmar_ratio: float = 0.0  # 卡尔玛比率
        self.maxdrawdown: float = 0.0  # 最大回撤
        self.maxdrawdown_rate: float = 0.0  # 最大回撤比例
        self.drawdown_interval: float = 0.0  # 最大回撤时间:小时 #TODO

        # 不计入手续费
        self.win_counts_without_commission: float = 0.0  # 盈利次数
        self.average_win_amount_without_commission: float = 0.0  # 单次盈利均值
        self.average_win_percentage_without_commission: float = 0.0  # 单次盈利率
        self.loss_counts_without_commission: float = 0.0  # 亏损次数 #TODO 正数
        self.average_loss_amount_without_commission: float = 0.0  # 单次亏损均值
        self.average_loss_percentage_without_commission: float = 0.0  # 单次亏损率
        self.win_loss_rate_without_commission: float = 0.0  # 盈亏比
        self.win_percentage_without_commission: float = 0.0  # 胜率
        self.average_returns_without_commission: float = 0.0  # 总体单次盈利率
        self.average_holding_time: float = 0.0  # 平均持仓时间
        # self.trading_counts: float = 0.0  # 交易次数-非hedge组
        # self.daily_trading_counts: float = 0.0  # 日度交易次数

        # 计入手续费（盈利不包含0）
        self.win_counts_with_commission_without_zero: float = 0.0  # 盈利次数
        self.average_win_amount_with_commission_without_zero: float = 0.0  # 单次盈利均值
        self.average_win_percentage_with_commission_without_zero: float = 0.0  # 单次盈利率
        self.loss_counts_with_commission_with_zero: float = 0.0  # 亏损次数
        self.average_loss_amount_with_commission_with_zero: float = 0.0  # 单次亏损均值
        self.average_loss_percentage_with_commission_with_zero: float = 0.0  # 单次亏损率
        self.win_loss_rate_with_commission_without_zero: float = 0.0  # 盈亏比
        self.win_percentage_with_commission_without_zero: float = 0.0  # 胜率
        self.average_returns_with_commission_without_zero: float = 0.0  # 总体单次盈利率

        # 计入手续费（盈利包含0）
        self.win_counts_with_commission_with_zero: float = 0.0  # 盈利次数
        self.average_win_amount_with_commission_with_zero: float = 0.0  # 单次盈利均值
        self.average_win_percentage_with_commission_with_zero: float = 0.0  # 单次盈利率
        self.loss_counts_with_commission_without_zero: float = 0.0  # 亏损次数
        self.average_loss_amount_with_commission_without_zero: float = 0.0  # 单次亏损均值
        self.average_loss_percentage_with_commission_without_zero: float = 0.0  # 单次亏损率
        self.win_loss_rate_with_commission_with_zero: float = 0.0  # 盈亏比
        self.win_percentage_with_commission_with_zero: float = 0.0  # 胜率

    
    def __repr__(self):
        return str(self.__dict__)



    
