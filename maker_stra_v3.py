import os, time, logging, _ctypes
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

import warnings
warnings.filterwarnings("ignore")

from strategystats.stat_data import SymbolInfo
from strategystats.stra_stat import StrategyStatsRecord, plot_from_npz
from strategystats.stat_load import initial_balance, get_capital_list
from strategystats.utils.nautilus import *
from strategystats.utils.nautilus_tool import *

from utils.helper import iter_tool
from utils.order_manager import OPENED, CANCELING, OrderManager

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.message import Event
from nautilus_trader.model.book import OrderBook
from nautilus_trader.common.component import TimeEvent
from nautilus_trader.model.data import OrderBookDeltas, QuoteTick, TradeTick, BarType, Bar, BarSpecification
from nautilus_trader.model.enums import BookType, PriceType, AggregationSource
from nautilus_trader.model.events import PositionChanged, PositionClosed, PositionOpened, OrderFilled, OrderCanceled, OrderExpired, OrderRejected, OrderSubmitted, OrderDenied
from nautilus_trader.model.identifiers import InstrumentId, ClientOrderId, StrategyId, AccountId, Venue
from nautilus_trader.model.instruments import Instrument 
from nautilus_trader.model.functions import position_side_to_str 
from nautilus_trader.model.objects import Price, Quantity, Currency
from nautilus_trader.trading.strategy import Strategy 
from nautilus_trader.model.enums import OrderSide, AggressorSide, PositionSide, TimeInForce, TriggerType


EPSILON = 1e-8
FEATURE_TIME_COEFFICIENT = 1  # 以ms为基准的feature timestamp转化系数
FEATURE_LATENCY = 4  # 信号延迟时间 ms单位

# -------------------------------strategy module--------------------------------- #


class MyStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    starting_balance: list
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    save_path: str
    factor_type: str
    factors: list
    threshold: dict
    factor_path: str
    threshold_data: pd.DataFrame
    holding_time: float
    order_amount: float
    plot_config: dict
    symbol_info: int
    stop_loss_rate: float
    stop_win_rate: float
    price_interval: float
    stop_loss_taker_threshold: float
    stop_loss_type: str
    feature_interval: float


class MyStrategy(Strategy):
    def __init__(self, config: MyStrategyConfig):
        super().__init__(config)
        
        self.start_time = config.start_time                                         # 开始时间
        self.end_time = config.end_time                                             # 结束时间

        # for record & plot
        self.symbol_info = _ctypes.PyObj_FromPtr(config.symbol_info)
        capital = get_capital_list(config.starting_balance)
        init_balance = initial_balance(self.symbol_info, capital, self.start_time.value)
        self.stats_record = StrategyStatsRecord(init_balance)
        self.stats_record.order_list = NautilusOrderList(self.symbol_info)
        self.plot_config = config.plot_config

        self.instrument_id = config.instrument_id                                  # 交易对
        self.quote_token = self.symbol_info.token
        self.base_token = self.symbol_info.quote
        self.init_pos = init_balance.capitals[self.symbol_info.token]
        self.order_amount = config.order_amount                                     # 下单数量
        self.save_path = config.save_path                                           # 存储目录
        self.instrument: Instrument | None = None                                   # 交易对
        self.account_id: AccountId | None = None                                    # 账户
        
        self.initial_date = self.start_time - pd.Timedelta(days=1)                  # 初始日期
        self.venue = Venue(config.instrument_id.venue.value)                                               # 交易所
        self.date_range = pd.date_range(start=self.start_time - timedelta(days=1), end=self.end_time + timedelta(days=1), freq='D')
        self.current_date_index = list(self.date_range).index(pd.Timestamp(self.start_time))
        
        self.pos = 0
        self.curr_signal_row = None
        self.order_manager = OrderManager()
        self.max_pos = config.order_amount * 1

        self.last_ask_price = np.inf
        self.last_bid_price = -np.inf
        self.ask_price = None
        self.bid_price = None

        self.on_signal_get_feature_value_time = 0
        self.on_signal_time = 0
        self.on_quote_tick_time = 0
        self.on_order_filled_time = 0
        self.limit_order_time = 0

        self.price_interval = config.price_interval
        
        self.factor_type = config.factor_type
        self.factors = config.factors  # 因子名称
        self.date_range =  pd.date_range(start=self.start_time-timedelta(days=1), end=self.end_time+timedelta(days=1), freq='D')
        self.current_date_index = list(self.date_range).index(pd.Timestamp(self.start_time))
        self.feature_data = None
        self.feature_data_max_timestamp = 0
        self.feature_interval = config.feature_interval if self.factor_type != 'LobImbalance' else 101
        # self.load_initial_feature_data()
        self.feature_percentiles = {}
        self.threshold = config.threshold
        self.factor_path = config.factor_path
        self.curr_row = None
        # threshold_data = config.threshold_data[config.threshold_data['factor']==self.factor].set_index('time')
        # self.threshold_data = iter_tool(threshold_data)
        self.threshold_data = config.threshold_data
        # self.threshold_data_max_date = threshold_data.index[-1]
        self.last_feature_value = 0
        self.backward_factors = ['LobImbalance_0_5', 'NI_mean_0_5_5', 'LobImbalance_5_10', 'CNVPL_ask_mean_0_5_5']
        self.forward_factors = ['TFI_5', 'CRSI_5', 'MultiLevelOrderFlowImbalance_0_10_1', 'CNVPL_bid_mean_0_5_5', 'oir_avg_0_5']

    def on_start(self):
        self.run_start_time = time.time()
        
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            logging.error(f"Could not find instrument for {self.instrument_id}")
            self.stop()
            return

        self.account_id = self.cache.account_id(self.venue)
        print('-------------', self.venue)
        print('------------', self.account_id)
        print('------------', self.instrument_id)
        print('------------', self.instrument)
        print(self.symbol_info)

        self.clock.set_timer(name='calculte_signal', interval=pd.Timedelta(milliseconds=self.feature_interval), start_time=self.start_time+timedelta(milliseconds=FEATURE_LATENCY), callback=lambda e: self.on_signal(e))
        self.clock.set_timer(name='update', interval=pd.Timedelta(milliseconds=1), start_time=self.start_time, callback=lambda e: self.on_update_ms(e))
        self.subscribe_quote_ticks(self.instrument_id)
    
    def on_update_ms(self, event: TimeEvent):
        # 加上链路延迟后，订单接收只有在有新的事件触发后才会触发接收，所以加上每ms这个timer事件，触发订单接收
        return

    def on_stop(self) -> None:
        """
        Actions to be performed when the strategy is stopped.
        """
        # _account = self.portfolio.account(self.instrument_id.venue)
        # _cap = _account.balance(Currency.from_str(self.base_token))
        # _free = float(_cap.free)
        # print(_free)
        
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)

        if self.pos != 0:
            close_price = float(self.bid_price) if self.pos > 0 else float(self.ask_price)
            close_amount = abs(self.pos)
            side = OrderSide.BUY if self.pos < 0 else OrderSide.SELL
            close_ts = self.end_time.timestamp() * 1e9
            order_id = f"on_stop_close_all_order_{ts_to_millisecond(close_ts)}"
            self.stats_record.order_list.append(order_id,
                                                status=OrderStatus.FILLED,
                                                price=close_price,
                                                amount=close_amount,
                                                side=side,
                                                trade_type=TradeType.TAKER,
                                                order_type=OrderType.MARKET,
                                                filled_price=close_price,
                                                filled_amount=close_amount,
                                                commission=float(self.instrument.taker_fee) * close_amount * close_price,
                                                ts=ts_to_millisecond(close_ts),
                                               )

        self.run_end_time = time.time()
        logging.info(f"回测结束{self.start_time}~{self.end_time}")
        logging.info(f"time needed in mins: {(self.run_end_time - self.run_start_time) / 60}")
        logging.info(f"on_signal time needed in mins: {(self.on_signal_time) / 60}")
        logging.info(f"on_quote_tick time needed in mins: {(self.on_quote_tick_time) / 60}")
        logging.info(f"on_order_filled time needed in mins: {(self.on_order_filled_time) / 60}")
        logging.info(f"on_signal_get_feature_value time needed in mins: {(self.on_signal_get_feature_value_time) / 60}")
        logging.info(f"limit_order time needed in mins: {(self.limit_order_time) / 60}")

        self.stats_record.to_npz(self.save_path)
        logging.warning(f"save npz and stat plot time needed in mins: {(time.time()- self.run_end_time) / 60}")

    def on_signal(self, event: TimeEvent):
        t1 = time.time()
        UTC_time = event.ts_event / 1e6
        if self.ask_price is None or self.bid_price is None:
            return
        
        # 若风控导致追盘口时没有挂关仓单，需要补上
        if self.pos > EPSILON:
            place_amount = self.pos - self.order_manager.get_open_order_amount(OrderSide.SELL)
            side = OrderSide.SELL
        elif self.pos < -EPSILON:
            place_amount = -self.pos - self.order_manager.get_open_order_amount(OrderSide.BUY)
            side = OrderSide.BUY
        else:
            place_amount = 0

        if place_amount > EPSILON:
            check_ret, msg = self.check_risk(side)
            if check_ret:
                place_price = self.get_order_place_price(side)
                order_id = self.limit_order(side=side, volume=place_amount, price=place_price,
                                            post_only=True, reduce_only=False, expire_time=None, life_time=None)
                logging.debug(f"关仓单挂单完成: s:{side} p:{place_price} amt:{place_amount} id:{order_id} pos:{self.pos}")
            else:
                logging.debug(f"不发送关仓单: {msg}")
            
        """strategy calculate"""

        feature_value = self.get_feature_value(UTC_time)
        self.on_signal_get_feature_value_time += time.time() - t1

        signal_buy, signal_sell = False, False
        if feature_value is not None:
            self.signal_value_buy = getattr(feature_value, self.factors[0])
            self.signal_value_sell = getattr(feature_value, self.factors[1])
            if getattr(feature_value, self.factors[0]) > self.threshold['above']:
                signal_buy = True
            if getattr(feature_value, self.factors[1]) > self.threshold['below']:
                signal_sell = True

            logging.debug(f"on_signal 计算: {UTC_time} buy:{signal_buy}, sell:{signal_sell} {self.signal_value_buy}:{self.signal_value_sell}")
        else:
            logging.debug(f"on_signal 计算: {UTC_time} 未获取到信号数据 ")
            return

        # 有信号，查看当前有无同方向挂单：没挂单>挂单
        # 无信号，查看pos，如果pos和信号方向相反，不撤单，否则撤单
        if signal_buy:
            order_info_list = self.order_manager.get_open_order_info(OrderSide.BUY)
            if len(order_info_list) == 0:
                check_ret, msg = self.check_risk(OrderSide.BUY)
                if check_ret:
                    price = self.get_order_place_price(OrderSide.BUY)
                    order_id = self.limit_order(side=OrderSide.BUY, volume=self.order_amount, price=price)
                    logging.debug(f"发送buy信号单: {order_id} {event.ts_event}")
                else:
                    logging.debug(f"不发送buy信号单: {msg}")
            else:
                logging.debug(f"有buy信号, 但不需要下单")
        else:
            if self.pos >= -EPSILON:  # 无信号且有该方向仓位要撤同向挂单
                order_info_list = self.order_manager.get_open_order_info(OrderSide.BUY)
                for order_info in order_info_list:
                    self.order_manager.update_order_status(order_info['order_id'], 'cancel')  # todo: 接受的时候判断一下状态撤掉
                    order = self.cache.order(order_info['order_id'])
                    if order.is_open:  # 不在路上才能撤
                        self.cancel_order(order)
                        logging.debug(f"撤销订单: {order_info['side']} {order_info['order_id']}")
                    else:
                        logging.debug(f"信号触发撤单，但暂时先不撤单: {order_info['side']} {order_info['order_id']}")

        if signal_sell:
            order_info_list = self.order_manager.get_open_order_info(OrderSide.SELL)
            if len(order_info_list) == 0:
                check_ret, msg = self.check_risk(OrderSide.SELL)
                if check_ret:
                    price = self.get_order_place_price(OrderSide.SELL)
                    order_id = self.limit_order(side=OrderSide.SELL, volume=self.order_amount, price=price)
                    logging.debug(f"发送sell信号单: {order_id} {event.ts_event}")
                else:
                    logging.debug(f"不发送sell信号单: {msg}")
            else:
                logging.debug(f"有sell信号, 但不需要下单")
        else:
            if self.pos <= EPSILON:  # 无信号且有该方向仓位要撤同向挂单
                order_info_list = self.order_manager.get_open_order_info(OrderSide.SELL)
                for order_info in order_info_list:
                    self.order_manager.update_order_status(order_info['order_id'], 'cancel')
                    order = self.cache.order(order_info['order_id'])
                    if order.is_open:
                        self.cancel_order(order)
                        logging.debug(f"撤销订单: {order_info['side']} {order_info['order_id']}")
                    else:
                        logging.debug(f"信号触发撤单，但暂时先不撤单: {order_info['side']}{order_info['order_id']}")
        self.on_signal_time += time.time() - t1
        return
    
    def on_quote_tick(self, ticker: QuoteTick) -> None:
        t1 = time.time()
        logging.debug(f"收到tick数据: {ticker}")

        self.ask_price = float(ticker.ask_price)
        self.bid_price = float(ticker.bid_price)
        # UTC_time = ticker.ts_event / 1e6

        if abs(self.ask_price - self.last_ask_price) < EPSILON and abs(self.bid_price - self.last_bid_price) < EPSILON:
            return

        buy_amount = self.order_manager.get_open_order_amount(OrderSide.BUY)
        sell_amount = self.order_manager.get_open_order_amount(OrderSide.SELL)

        if buy_amount > EPSILON and sell_amount < EPSILON:
            # amount = 0
            price = self.get_order_place_price(OrderSide.BUY)
            # TODO：注意在判断价格是否相等时是否会遇到精度问题
            cancel_order_id_list = self.order_manager.get_cancel_order_ids(OrderSide.BUY, price)
            for order_id in cancel_order_id_list:
                self.order_manager.update_order_status(order_id, 'cancel')
                order = self.cache.order(order_id)
                if order.is_open:
                    self.cancel_order(order)
                    logging.debug(f"改单撤单: {order_id}")
                else:
                    logging.debug(f"触发改单撤单, 但先不执行: {order_id}")
                check_ret, msg = self.check_risk(OrderSide.BUY)
                if check_ret:
                    if not order.is_closed:
                        new_order_id = self.limit_order(side=OrderSide.BUY, volume=float(order.leaves_qty), price=price)
                        logging.debug(f"追单发单: {new_order_id}")
                    else:
                        logging.debug(f"触发追单发单 但这单已经close, 不发单")
                else:
                    logging.debug(f"触发追单 但风控不通过: {msg}")
            #     # 追单订单合并
            #     if not order.is_closed:
            #         amount += order.leaves_qty
            #     else:
            #         logging.debug(f"触发追单发单 但这单已经close, 不发单")
            # if amount > EPSILON:
            #     check_ret, msg = self.check_risk(OrderSide.BUY)
            #     if check_ret:
            #         new_order_id = self.limit_order(side=OrderSide.BUY, volume=float(amount), price=price)
            #         logging.debug(f"追单发单: {new_order_id}")
            #     else:
            #         logging.debug(f"触发追单 但风控不通过: {msg}")
        elif sell_amount > EPSILON and buy_amount < EPSILON:
            # amount = 0
            price = self.get_order_place_price(OrderSide.SELL)
            cancel_order_id_list = self.order_manager.get_cancel_order_ids(OrderSide.SELL, price)
            for order_id in cancel_order_id_list:
                self.order_manager.update_order_status(order_id, 'cancel')
                order = self.cache.order(order_id)
                if order.is_open:
                    self.cancel_order(order)
                    logging.debug(f"改单撤单: {order_id}")
                else:
                    logging.debug(f"触发改单撤单, 但先不执行: {order_id}")
                check_ret, msg = self.check_risk(OrderSide.SELL)
                if check_ret:
                    if not order.is_closed:
                        new_order_id = self.limit_order(side=OrderSide.SELL, volume=float(order.leaves_qty), price=price)
                        logging.debug(f"追单发单: {new_order_id}")
                    else:
                        logging.debug(f"触发追单发单 但这单已经close, 不发单")
                else:
                    logging.debug(f"触发追单 但风控不通过: {msg}")
            #     # 追单订单合并
            #     if not order.is_closed:
            #         amount += order.leaves_qty
            #     else:
            #         logging.debug(f"触发追单发单 但这单已经close, 不发单")
            # if amount > EPSILON:
            #     check_ret, msg = self.check_risk(OrderSide.SELL)
            #     if check_ret:
            #         new_order_id = self.limit_order(side=OrderSide.SELL, volume=float(amount), price=price)
            #         logging.debug(f"追单发单: {new_order_id}")
            #     else:
            #         logging.debug(f"触发追单 但风控不通过: {msg}")

        self.last_ask_price = ticker.ask_price
        self.last_bid_price = ticker.bid_price

        self.on_quote_tick_time += time.time() - t1

        if len(self.stats_record.price_list) == 0 or \
            (self.stats_record.price_list[-1].ts_event < ticker.ts_event // 1e6 - self.price_interval + 1 and \
             self.stats_record.price_list[-1].mid_price.price != (ticker.ask_price + ticker.bid_price) / 2):
            # self.stats_record.price_list[-1].ts_event < ticker.ts_event):
            self.stats_record.price_list.append(from_quote_tick(ticker))
        return
        
    def on_order_accepted(self, event: OrderSubmitted):
        logging.debug(f"订单接受:{event}")
        # TODO: 回测系统加入链路延时后，在路上的订单不允许撤单，如果撤单也加了链路延时呢？是不是就可以了
        if self.order_manager.orders[event.client_order_id]['status'] == CANCELING:
            order = self.cache.order(event.client_order_id)
            self.cancel_order(order)
            logging.debug(f"撤销订单: {event.client_order_id}")
            if not order.is_open:
                logging.error(f"遇到不符合预期的情况")

        # for record & plot
        # self.stats_record.order_list.update(event)
        return
    
    def on_order_rejected(self, event: OrderRejected):
        logging.debug(f"订单拒绝:{event}")
        self.stats_record.order_list.update(event)

        if self.order_manager.orders[event.client_order_id]['status'] == CANCELING:
            logging.debug(f"订单被拒后，不补单：订单已发送过撤销")
            self.order_manager.del_order(event.client_order_id)
            return

        self.order_manager.del_order(event.client_order_id)
        
        # 增加风控，同一个方向只能存在3个进行中订单
        
        order = self.cache.order(event.client_order_id)

        check_ret, msg = self.check_risk(order.side)
        if check_ret:
            place_price = self.get_order_place_price(order.side)
            order_id = self.limit_order(side=order.side, volume=float(order.leaves_qty), price=place_price)
            logging.debug(f"订单被拒后，重新补单: p: {place_price} {order_id}")
        else:
            logging.debug(f"订单被拒后，不补单: {msg}")
        
        # for record & plot
        # self.stats_record.order_list.update(event)
        return
    
    def on_order_filled(self, event: OrderFilled):
        t1 = time.time()
        # 响应事件“订单完成”
        logging.debug(f"订单成交:{event}")
        # self.log.warning(f"订单成交:{event}")
        self.stats_record.order_list.update(event)
        self.order_manager.del_order(event.client_order_id)

        filled_amount = float(event.last_qty)
        # filled_amount = event.last_qty
        # filled_price = float(event.last_px)
        # order_id = event.client_order_id
        # update_timestamp = event.ts_event / 1000 / 1000  # ts_event是纳秒,转换为毫秒

        sign = 1 if event.order_side == OrderSide.BUY else -1
        self.pos = float(round(sign * filled_amount + self.pos, self.instrument.size_precision))

        if self.pos > EPSILON:
            place_amount = self.pos - self.order_manager.get_open_order_amount(OrderSide.SELL)
            side = OrderSide.SELL
        elif self.pos < -EPSILON:
            place_amount = -self.pos - self.order_manager.get_open_order_amount(OrderSide.BUY)
            side = OrderSide.BUY
        else:
            place_amount = 0

        if place_amount > EPSILON:
            check_ret, msg = self.check_risk(side)
            if check_ret:
                place_price = self.get_order_place_price(side)
                order_id = self.limit_order(side=side, volume=place_amount, price=place_price,
                                            post_only=True, reduce_only=False, expire_time=None, life_time=None)
                logging.debug(f"关仓单挂单完成: s:{side} p:{place_price} amt:{place_amount} id:{order_id} pos:{self.pos}")
            else:
                logging.debug(f"不发送关仓单: {msg}")

        self.on_order_filled_time += time.time() - t1
        return
    
    def on_order_canceled(self, event: OrderCanceled):
        logging.debug(f"订单取消:{event}")
        self.order_manager.del_order(event.client_order_id)
        self.stats_record.order_list.update(event)
        return
    
    def on_order_expired(self, event: OrderExpired):
        logging.debug(f"订单超时:{event}")
        self.order_manager.del_order(event.client_order_id)
        self.stats_record.order_list.update(event)
        return

    def on_order_denied(self, event: OrderDenied):
        logging.debug(f"订单失败:{event}")
        print(f"订单失败:{event.client_order_id}")
        print(f"订单失败:{self.order_manager.orders.keys()}")
        print(f"订单失败:{event.client_order_id in self.order_manager.orders.keys()}")
        self.order_manager.del_order(event.client_order_id)
        self.stats_record.order_list.update(event)
        return
    
    def on_order_submitted(self, event: OrderSubmitted):
        # TODO: 有必要保存吗？
        # self.stats_record.order_list.update(event)
        return
    
    def get_order_place_price(self, side):
        '''
            获取关仓价格：按照仓位优先，信号优先
               当前方向有仓位：保守挂单
               当前方向信号弱：保守挂单，反之，激进贴盘口
        '''
        if self.pos > EPSILON:
            priority_side = OrderSide.SELL
        elif self.pos < -EPSILON:
            priority_side = OrderSide.BUY
        elif self.signal_value_buy > self.signal_value_sell:
            priority_side = OrderSide.BUY
        else:
            priority_side = OrderSide.SELL

        if side == priority_side and side == OrderSide.BUY:
            return self.ask_price - self.instrument.price_increment
        elif side == priority_side and side == OrderSide.SELL:
            return self.bid_price + self.instrument.price_increment
        elif side != priority_side and side == OrderSide.BUY:
            return self.bid_price
        else:
            return self.ask_price

    def check_risk(self, side):
        check_ret, msg = False, ''
        if not self.is_pos_control(side):
            return check_ret, f"仓位{self.pos} {side}超出最大仓位{self.max_pos}，不进行下单"
        
        not_stoped_orderids = self.order_manager.get_not_stoped_order_ids(side)
        if len(not_stoped_orderids) >= 3:
            return check_ret, f"s:{side} 方向未完成订单{len(not_stoped_orderids)}超过限制 停止挂单"
        
        return True, ''

    def is_pos_control(self, side):
        if (side == OrderSide.BUY and self.pos <= self.max_pos):
            return True
        elif (side == OrderSide.SELL and self.pos >= -self.max_pos):
            return True
        else:
            return False

    def limit_order(self, side, volume, price, post_only=True, reduce_only=False, expire_time=None, life_time=None):
        t1 = time.time()

        # 不考虑超时，挂GTC，手动撤单
        order = self.order_factory.limit(
            instrument_id=self.instrument_id,
            order_side=side,
            price=Price(price, precision=self.instrument.price_precision),
            quantity=self.instrument.make_qty(volume),
            post_only=post_only,
            reduce_only=reduce_only,
            time_in_force=TimeInForce.GTC,
            emulation_trigger=TriggerType["NO_TRIGGER"]
        )
        logging.debug(f"发送limit订单: {order}")
        # for record & plot
        # register_order(self.symbol_info, order)
        self.stats_record.order_list.register(order, self.symbol_info, is_maker=True)
        
        # NOTE: order id is useful for further processing, eg. cancel order
        _order_id = order.client_order_id
        self.submit_order(order)
        # self.order_list[_order_id] = order

        if not order.is_closed:
            self.order_manager.add_order(price, side, volume, _order_id)
            
        self.limit_order_time += time.time() - t1

        return _order_id

    # -------------------------------feature function--------------------------------- #

    def load_feature_data(self, path: str, date) -> pd.DataFrame:
        file_path = os.path.join(path, f"feature_{date.strftime('%Y%m%d')}.parquet")

        df = pd.read_parquet(file_path)
        df.columns = [row.replace('.','_') for row in df.columns]
        df['timestamp'] = df['timestamp'] / FEATURE_TIME_COEFFICIENT
        # df.set_index('timestamp', inplace=True)
        self.feature_data_max_timestamp = df['timestamp'].iloc[-1]

        iter_df = iter_tool(df)

        return iter_df

    def update_feature_data(self, timestamp):
        if self.feature_data is None or timestamp - self.feature_data_max_timestamp >= self.feature_interval:
            if 0 < self.current_date_index < len(self.date_range):
                next_date = self.date_range[self.current_date_index]
                logging.info(f"开始load: {next_date} 的信号")
                self.feature_data = self.load_feature_data(self.factor_path, next_date)
                last_date = self.date_range[self.current_date_index-1]
                if self.factor_type == 'LobImbalance':
                    self.feature_percentiles = self.calculate_feature_percentiles(last_date)
                self.current_date_index += 1
            else:
                self.feature_data = None  # No more data to load

    def get_feature_value(self, timestamp):

        self.update_feature_data(timestamp)
        
        if self.feature_data is not None:

            if self.curr_signal_row is None:
                self.curr_signal_row = next(self.feature_data)
                return None

            if timestamp - self.curr_signal_row.timestamp < 0:
                # print(f'curr_row: {self.curr_signal_row["timestamp"]}')
                # print(f'timestamp: {timestamp}')
                return None
                
            while (timestamp - self.curr_signal_row.timestamp >= self.feature_interval):
                try:
                    self.curr_signal_row = next(self.feature_data)
                except StopIteration:
                    print(f"No more items in the iterator. {self.curr_signal_row.timestamp}")

            # print(f'curr_row: {self.curr_signal_row.timestamp}')
            # print(f'timestamp: {timestamp}')
            return self.curr_signal_row
            
        return None

