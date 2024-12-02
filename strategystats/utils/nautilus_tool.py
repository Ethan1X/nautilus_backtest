from nautilus_trader.model.events import OrderEvent,OrderFilled,OrderCanceled,OrderExpired,OrderSubmitted, OrderRejected
from nautilus_trader.model.enums import LiquiditySide, OrderSide
from nautilus_trader.model.objects import Money
from nautilus_trader.model.orders import Order, LimitOrder, MarketOrder
from ..stat_data import *


# NOTE：在回测程序中，用此类替代OrderList即可
class NautilusOrderList(object):

    def __init__(self, symbol_info: SymbolInfo=None):
        self._order_list = OrderList(symbol_info)

    def load_npz(self, order_npz):
        self._order_list.load_npz(order_npz)

    def build_index(self):
        self._order_list.build_index()

    def register(self, order, symbol_info, is_maker=True):
        if order is not None and (symbol_info is None or self._order_list.symbol_info is None or symbol_info == self._order_list.symbol_info):
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

            if is_maker is None:
                trade_type = TradeType.INVALID_TRADE
            else:
                trade_type = TradeType.MAKER if is_maker else TradeType.TAKER
            
            id = str(order.client_order_id)
            order_status = convert_order_status(order.status)
            price = float(order.price) if order_type == OrderType.LIMIT else 0.0
            
            self._order_list._append(order_id=id, status=order_status, price=price, amount=float(order.quantity), 
                         side=trade_side, trade_type=trade_type, order_type=order_type, 
                         filled_price=0.0, filled_amount=0.0, commission=0.0, ts=order.ts_init)
            self._order_list.index[id] = len(self._order_list.order_id) - 1

    def update(self, event):
        id = str(event.client_order_id)
        if id not in self._order_list.index:
            print(f"Unregistered order '{id}'!")
            return None

        idx = self._order_list.index.get(id)
        status = OrderStatus[self._order_list.status[idx]]
        ts = ts_to_millisecond(event.ts_event)
        trade_type = self._order_list.trade_type[idx]
        filled_price = self._order_list.filled_price[idx]
        filled_amount = self._order_list.filled_amount[idx]
        commission = self._order_list.commission[idx]
        
        if isinstance(event, OrderSubmitted):
            status = OrderStatus.OPENED
        elif isinstance(event, OrderFilled):
            price = float(event.last_px)
            amount = float(event.last_qty)
            filled_price = (filled_price * filled_amount + price * amount) / (filled_amount + amount)
            # calc_filled_price = (order_info.filled_price * order_info.filled_amount + price * amount) / (order_info.filled_amount + amount)
            # calc_filled_amount = order_info.filled_amount + amount
            filled_amount += amount
            # order_info.filled_price = float(order.avg_px)
            # order_info.filled_amount = float(order.filled_qty)
            # if calc_filled_price != order_info.filled_price:
            #     print(f"Calculated filled price: {calc_filled_price}, Nautilus avg price: {float(order.avg_px)}")
            #     print(f"Calculated filled amount: {calc_filled_amount}, Nautilus filled amount: {float(order.filled_qty)}")
                
            if filled_amount == self._order_list.amount[idx]:
                status = OrderStatus.FILLED
            else:
                status = OrderStatus.PARTLIFILLED
            # TODO: order.commissions()是不同币种计价的相同的累计commission，目前只有USDT计价的commission，所以还没出问题
            commission += float(event.commission) # 假定每次fill的commission用相同Currency
        elif isinstance(event, OrderCanceled):
            status = OrderStatus.CANCELED
        elif isinstance(event, OrderExpired):
            status = OrderStatus.EXPIRED
        elif isinstance(event, OrderRejected):
            status = OrderStatus.REJECTED
        else:
            print(f"Currently not supporting order events of type '{type(event)}'!")

        self._order_list.append(order_id=id, status=status, price=self._order_list.price[idx], amount=self._order_list.amount[idx], 
                         side=TradeSide[self._order_list.side[idx]], trade_type=TradeType[self._order_list.trade_type[idx]], order_type=OrderType[self._order_list.order_type[idx]], 
                         filled_price=filled_price, filled_amount=filled_amount, commission=commission, ts=ts)

    def get(self, order_id):
        return self._order_list.get(order_id)

    def get_at(self, idx):
        return self._order_list.get_at(idx)

    def to_dict(self):
        return self._order_list.to_dict()

    def append(self, order_id, status, price, amount, side, trade_type, order_type, filled_price, filled_amount, commission, ts):
        self._order_list.append(order_id, status, price, amount, side, trade_type, order_type, filled_price, filled_amount, commission, ts)

    @property
    def order_id(self):
        return self._order_list.order_id

    @property
    def symbol_info(self):
        return self._order_list.symbol_info

    @property
    def status(self):
        return self._order_list.status

    @property
    def price(self):
        return self._order_list.price

    @property
    def amount(self):
        return self._order_list.amount

    @property
    def side(self):
        return self._order_list.side

    @property
    def liquidity_side(self):
        return self._order_list.liquidity_side

    @property
    def trade_type(self):
        return self._order_list.trade_type

    @property
    def order_type(self):
        return self._order_list.order_type

    @property
    def filled_price(self):
        return self._order_list.filled_price

    @property
    def filled_amount(self):
        return self._order_list.filled_amount

    @property
    def commission(self):
        return self._order_list.commission

    @property
    def index(self):
        return self._order_list.index

    @property
    def ts(self):
        return self._order_list.ts
        




