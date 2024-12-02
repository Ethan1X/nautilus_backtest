import pandas as pd
import logging
from datetime import timedelta
import time
import os
from collections import defaultdict, deque
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from strategystats.stat_data import SymbolInfo
from strategystats.stra_stat import StrategyStatsRecord
from strategystats.stat_load import initial_balance, get_capital_list
from strategystats.utils.nautilus import *
from strategystats.utils.nautilus_tool import *

import _ctypes

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

from utils.helper import iter_tool
from utils.order_manager import OPENED, CANCELING, OrderManager

EPSILON = 1e-8
FEATURE_TIME_COEFFICIENT = 1  # 以ms为基准的feature timestamp转化系数
FEATURE_LATENCY = 4  # 信号延迟时间 ms单位

# -------------------------------strategy module--------------------------------- #

class MyStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId                                                   # id
    starting_balance: list                                                        # 初始balance
    start_time: pd.Timestamp                                                      # 回测开始时间
    end_time: pd.Timestamp                                                        # 回测结束时间
    save_path: str                                                                # 回测结果保存位置
    factor_type: str                                                              # 信号
    factors: list                                                                 # 信号列表
    threshold: dict                                                               # 信号阈值
    factor_path: str                                                               # 信号读取位置
    threshold_data: pd.DataFrame                                                  # 信号日阈值数据
    holding_time: float                                                          # 固定持仓时间
    order_amount: float                                                          # 固定下单量
    plot_config: dict                                                            # 回测画图config
    symbol_info: int                                                             # 回测symbol
    stop_loss_rate: float                                                        # 止损率
    stop_win_rate: float                                                         # 止盈率
    price_interval: float                                                        # 价格记录间隔
    stop_loss_taker_threshold: float                                             # taker止损阈值
    stop_loss_type: str                                                          # 止损方式
    feature_interval: float                                                      # 信号间隔

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
        self.price_interval = config.price_interval

        self.instrument_id = config.instrument_id                                  # 交易对
        self.quote_token = self.symbol_info.token                                  # 交易对token
        self.base_token = self.symbol_info.quote                                   # 交易对quote
        self.init_pos = init_balance.capitals[self.symbol_info.token]              # 初始仓位
        self.order_amount = config.order_amount                                     # 下单数量
        self.save_path = config.save_path                                           # 存储目录
        self.instrument: Instrument | None = None                                   # 交易对
        self.account_id: AccountId | None = None                                    # 账户
        
        self.initial_date = self.start_time - pd.Timedelta(days=1)                  # 初始日期
        self.venue = Venue(config.instrument_id.venue.value)                        # 交易所

        # 订单记录
        self.order_record = defaultdict(list)
        self.signal_orders = OrderManager()
        self.stop_win_orders = OrderManager()

        # 信号记录
        self.signal_value_buy = 0
        self.signal_value_sell = 0
        self.max_pos = self.order_amount * 5
        self.max_open_orders = 3

        # 最新盘口价
        self.ask_price = None
        self.bid_price = None

        self.holding_time = config.holding_time * 1000                                           # 最大持仓时间（ms）
        self.pos = 0
        self.open_price = None
        self.last_ask_price = np.inf
        self.last_bid_price = -np.inf

        # 止盈止损参数
        self.stop_win_rate = config.stop_win_rate
        self.stop_loss_rate = config.stop_loss_rate
        self.stop_loss_flag = False if config.stop_loss_type == 'NONE' else True
        self.stop_loss_type = config.stop_loss_type
        
        # 因子信号参数
        self.factor_type = config.factor_type
        self.factors = config.factors  # 因子名称
        # 按照天来load信号
        self.date_range =  pd.date_range(start=self.start_time-timedelta(days=1), end=self.end_time+timedelta(days=1), freq='D')
        self.current_date_index = list(self.date_range).index(pd.Timestamp(self.start_time))
        self.feature_data = None
        self.feature_data_max_timestamp = 0
        self.feature_interval = config.feature_interval if self.factor_type != 'LobImbalance' else 101
        self.feature_percentiles = {}
        self.threshold = config.threshold
        self.factor_path = config.factor_path
        self.curr_row = None
        self.threshold_data = config.threshold_data
        self.last_feature_value = 0
        self.backward_factors = ['LobImbalance_0_5', 'NI_mean_0_5_5', 'LobImbalance_5_10', 'CNVPL_ask_mean_0_5_5']
        self.forward_factors = ['TFI_5', 'CRSI_5', 'MultiLevelOrderFlowImbalance_0_10_1', 'CNVPL_bid_mean_0_5_5', 'oir_avg_0_5']

        # 回测时间记录
        self.run_start_time = None
        self.run_end_time = None
        self.on_signal_time = 0
        self.on_quote_tick_time = 0
        self.on_order_filled_time = 0
        self.on_signal_get_feature_value_time = 0
        self.limit_order_time = 0

        logging.info(f"回测开始 {self.start_time}~{self.end_time}..............................")

    # -------------------------------event function--------------------------------- #
    def on_start(self):
        self.run_start_time = time.time()
        
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            logging.error(f"Could not find instrument for {self.instrument_id}")
            self.stop()
            return

        self.account_id = self.cache.account_id(self.venue)
        print('-------------',self.venue)
        print('------------',self.account_id)
        print('------------',self.instrument_id)
        print('------------',self.instrument)
        print(self.symbol_info)

        self.clock.set_timer(name='calculte_signal', interval=pd.Timedelta(milliseconds=self.feature_interval), start_time=self.start_time+timedelta(milliseconds=FEATURE_LATENCY), callback=lambda e: self.on_signal(e))
        
        self.clock.set_timer(name='update', interval=pd.Timedelta(milliseconds=1), start_time=self.start_time, callback=lambda e: self.on_update_ms(e))
        
        # 这里只订阅盘口价来模拟回测，用trade没有考虑排队
        # bar_spec = BarSpecification.from_timedelta(self.time_interval, PriceType.MID)
        # bar_type = BarType(instrument_id=self.instrument_id, bar_spec=bar_spec, aggregation_source=AggregationSource.INTERNAL)
        # self.subscribe_bars(bar_type=bar_type)
        # self.subscribe_trade_ticks(self.instrument_id)
        self.subscribe_quote_ticks(self.instrument_id)
        # self.subscribe_order_book_at_interval(self.instrument_id, interval_ms=INTERVAL_MS)
    
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

        # for record & plot
        # 最后手动taker关仓单用作统计
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
                                                commission = float(self.instrument.taker_fee) * close_amount * close_price,
                                                ts = ts_to_millisecond(close_ts),
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

        np.savez_compressed(f"./{self.save_path}/order_strategy_type.npz", **self.order_record)
        logging.warning(f"save npz and stat plot time needed in mins: {(time.time()- self.run_end_time) / 60}")

    # 处理bar数据
    def on_bar(self, bar: Bar) -> None:
        pass


    def on_signal(self, event: TimeEvent):
        t1 = time.time()
        UTC_time = event.ts_event / 1000 / 1000
        if self.ask_price is None or self.bid_price is None:
            return
        """strategy calculate"""
        logging.debug(f"on_signal 计算: {UTC_time}")

        feature_value = self.get_feature_value(UTC_time)
        self.on_signal_get_feature_value_time += time.time() - t1

        signal_buy, signal_sell = False, False
        if feature_value is not None:
            signal_value_buy = getattr(feature_value, self.factors[0])
            signal_value_sell = getattr(feature_value, self.factors[1])
            if getattr(feature_value, self.factors[0]) > self.threshold['above']:
                signal_buy = True
            if getattr(feature_value, self.factors[1]) > self.threshold['below']:
                signal_sell = True
            logging.debug(f"on_signal 计算: {UTC_time} buy:{signal_buy}, sell:{signal_sell} {signal_value_buy}:{signal_value_sell}")
        else:
            logging.debug(f"on_signal 计算: {UTC_time} 未获取到信号数据 ")
            return

        buy_order_price =  self.ask_price-self.instrument.price_increment
        sell_order_price = self.bid_price+self.instrument.price_increment
        if signal_buy and signal_sell:
            if signal_value_buy < signal_value_sell:
                buy_order_price = self.bid_price
            else:
                sell_order_price = self.ask_price
        if feature_value is not None:
            self.signal_value_buy = signal_value_buy
            self.signal_value_sell = signal_value_sell

        logging.debug(f"信号: buy {self.signal_value_buy}; sell {self.signal_value_sell}")

        # 有信号，查看当前有无同方向挂单：没挂单>挂单
        # 无信号，查看pos，如果pos和信号方向相反，不撤单，否则撤单
        if signal_buy:
            order_info_list = self.signal_orders.get_open_order_info(OrderSide.BUY)
            if len(order_info_list) == 0:
                check_ret, msg = self.check_risk(OrderSide.BUY)
                if check_ret:
                    price = self.get_order_place_price(OrderSide.BUY)
                    order_id = self.limit_order(side=OrderSide.BUY, volume=self.order_amount, price=price, stra_type='signal')
                    logging.debug(f"发送buy信号单: {order_id} {event.ts_event}")
                else:
                    logging.debug(f"不发送buy信号单: {msg}")
            else:
                logging.debug(f"有buy信号, 但不需要下单")
        else:
            order_info_list = self.signal_orders.get_open_order_info(OrderSide.BUY)
            if len(order_info_list) > 0:
                for order_info in order_info_list:
                    self.signal_orders.update_order_status(order_info['order_id'], 'cancel')  # todo: 接受的时候判断一下状态撤掉
                    order = self.cache.order(order_info['order_id'])
                    if order.is_open:  # 不在路上才能撤
                        self.cancel_order(order)
                        logging.debug(f"撤销订单: {order_info['side']} {order_info['order_id']}")
                    else:
                        logging.debug(f"信号触发撤单，但暂时先不撤单: {order_info['side']} {order_info['order_id']}")

        if signal_sell:
            order_info_list = self.signal_orders.get_open_order_info(OrderSide.SELL)
            if len(order_info_list) == 0:
                check_ret, msg = self.check_risk(OrderSide.SELL)
                if check_ret:
                    price = self.get_order_place_price(OrderSide.SELL)
                    order_id = self.limit_order(side=OrderSide.SELL, volume=self.order_amount, price=price, stra_type='signal')
                    logging.debug(f"发送sell信号单: {order_id} {event.ts_event}")
                else:
                    logging.debug(f"不发送sell信号单: {msg}")
            else:
                logging.debug(f"有sell信号, 但不需要下单")
        else:
            order_info_list = self.signal_orders.get_open_order_info(OrderSide.SELL)
            if len(order_info_list) > 0:
                for order_info in order_info_list:
                    self.signal_orders.update_order_status(order_info['order_id'], 'cancel')
                    order = self.cache.order(order_info['order_id'])
                    if order.is_open:
                        self.cancel_order(order)
                        logging.debug(f"撤销订单: {order_info['side']} {order_info['order_id']}")
                    else:
                        logging.debug(f"信号触发撤单，但暂时先不撤单: {order_info['side']}{order_info['order_id']}")
  
        self.on_signal_time += time.time() - t1

        return

    def stop_loss(self):
        assert self.open_price > 0
        if self.quote_token == "BNB":
            if self.pos > 0 and self.bid_price - self.open_price <= -self.instrument.price_increment:
                return True
            elif self.pos < 0 and self.ask_price - self.open_price >= self.instrument.price_increment:
                return True
        else:
            if self.pos > 0 and self.bid_price / self.open_price - 1 < -self.stop_loss_rate:
                return True
            elif self.pos < 0 and self.ask_price / self.open_price - 1 > self.stop_loss_rate:
                return True
        
        return False

    # 处理DEPTH/Order_book/Snapshots数据
    # 这个是可以获取到深度数据的，并且深度数据是唯一可以设置时间戳间隔的数据，因此我们的因子测试如果是在相同时间间隔下测试的话，可以使用这个数据
    def on_order_book(self, orderbook: OrderBook) -> None:
        return
    
    # 处理QUOTE/TICK数据
    def on_quote_tick(self, ticker: QuoteTick) -> None:
        t1 = time.time()
        logging.debug(f"收到tick数据: {ticker}")

        self.ask_price = ticker.ask_price
        self.bid_price = ticker.bid_price
        UTC_time = ticker.ts_event // 1e6
        
        if abs(self.ask_price - self.last_ask_price) < EPSILON and abs(self.bid_price - self.last_bid_price) < EPSILON:
            return

        buy_amount = self.signal_orders.get_open_order_amount(OrderSide.BUY)
        sell_amount = self.signal_orders.get_open_order_amount(OrderSide.SELL)

        if buy_amount > EPSILON and sell_amount < EPSILON:
            # amount = 0
            price = self.get_order_place_price(OrderSide.BUY)
            # TODO：注意在判断价格是否相等时是否会遇到精度问题
            cancel_order_id_list = self.signal_orders.get_cancel_order_ids(OrderSide.BUY, price)
            for order_id in cancel_order_id_list:
                self.signal_orders.update_order_status(order_id, 'cancel')
                order = self.cache.order(order_id)
                if order.is_open:
                    self.cancel_order(order)
                    logging.debug(f"改单撤单: {order_id}")
                else:
                    logging.debug(f"触发改单撤单, 但先不执行: {order_id}")
                check_ret, msg = self.check_risk(OrderSide.BUY)
                if check_ret:
                    if not order.is_closed:
                        new_order_id = self.limit_order(side=OrderSide.BUY, volume=float(order.leaves_qty), price=price, stra_type='signal')
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
            cancel_order_id_list = self.signal_orders.get_cancel_order_ids(OrderSide.SELL, price)
            for order_id in cancel_order_id_list:
                self.signal_orders.update_order_status(order_id, 'cancel')
                order = self.cache.order(order_id)
                if order.is_open:
                    self.cancel_order(order)
                    logging.debug(f"改单撤单: {order_id}")
                else:
                    logging.debug(f"触发改单撤单, 但先不执行: {order_id}")
                check_ret, msg = self.check_risk(OrderSide.SELL)
                if check_ret:
                    if not order.is_closed:
                        new_order_id = self.limit_order(side=OrderSide.SELL, volume=float(order.leaves_qty), price=price, stra_type='signal')
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

        # for record & plot
        if len(self.stats_record.price_list) == 0 or (
            self.stats_record.price_list[-1].ts_event < ticker.ts_event // 1e6 - self.price_interval + 1 and 
            self.stats_record.price_list[-1].mid_price.price != (ticker.ask_price+ticker.bid_price)/2):
            # self.stats_record.price_list[-1].ts_event < ticker.ts_event):
            self.stats_record.price_list.append(from_quote_tick(ticker))

        self.last_ask_price = ticker.ask_price
        self.last_bid_price = ticker.bid_price

        self.on_quote_tick_time += time.time() - t1
        
        return

    # 处理TRADE数据
    def on_trade_tick(self, trade: TradeTick) -> None:
        return
        
    def on_order_accepted(self, event: OrderSubmitted):
        logging.debug(f"订单接受:{event}")
        # 如果发过撤单请求 则需要在接受后里立马撤单
        if event.client_order_id in self.signal_orders.orders.keys():
            if self.signal_orders.orders[event.client_order_id]['status'] == CANCELING:
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

        if event.client_order_id in self.signal_orders.orders.keys():
            if self.signal_orders.orders[event.client_order_id]['status'] == CANCELING:
                logging.debug(f"订单被拒后，不补单：订单已发送过撤销")
                self.signal_orders.del_order(event.client_order_id)
                return
            self.signal_orders.del_order(event.client_order_id)
            logging.debug(f"删除信号单: {event.client_order_id}")
        elif event.client_order_id in self.stop_win_orders.orders.keys():
            if self.stop_win_orders.orders[event.client_order_id]['status'] == CANCELING:
                logging.debug(f"订单被拒后，不补单：订单已发送过撤销")
                self.stop_win_orders.del_order(event.client_order_id)
                return
            self.stop_win_orders.del_order(event.client_order_id)
        
        # 增加风控，同一个方向只能存在3个进行中订单
        order = self.cache.order(event.client_order_id)
        check_ret, msg = self.check_risk(order.side)
        if check_ret:
            place_price = self.get_order_place_price(order.side)
            
            order_id = self.limit_order(side=order.side, volume=float(order.leaves_qty), price=place_price, stra_type='signal')
            logging.debug(f"订单被拒后，重新补单: p: {place_price} {order_id}")
        else:
            logging.debug(f"订单被拒后，不补单: {msg}")
        
        # for record & plot
        # self.stats_record.order_list.update(event)
          
        return
        
    
    def on_order_filled(self, event: OrderFilled):
        # 响应事件“订单完成”
        logging.debug(f"订单成交:{event}")
        # for record & plot
        self.stats_record.order_list.update(event)
        
        filled_amount = float(event.last_qty)
        filled_price = float(event.last_px)
        order_id = event.client_order_id
        # fee = event.commission
        update_timestamp = event.ts_event / 1000 / 1000 # ts_event是纳秒,转换为毫秒

        # 计算持仓均价
        sign = 1 if event.order_side == OrderSide.BUY else -1
        amount_sum = round(sign * filled_amount + self.pos, self.instrument.size_precision)
        if self.open_price is None:
            self.open_price = filled_price
        else:
            if abs(amount_sum) > 1e-10:
                self.open_price = (self.open_price * self.pos + filled_price * filled_amount * sign) / amount_sum
            else:
                self.open_price = None
        # 更新仓位
        last_pos = self.pos
        self.pos = amount_sum
            
        if order_id in self.signal_orders.orders.keys():
            # 判断是否要撤止盈单
            if (
                (last_pos > 0 and event.order_side == OrderSide.SELL)
                or (last_pos < 0 and event.order_side == OrderSide.BUY)
            ):
                _order_list = self.cache.orders_open()
                if len(_order_list) > 0:
                    _order_list = sorted(_order_list, key=lambda x: x.price)
                    # _order_list = list(filter(lambda x: x.side == event.order_side, _order_list))
                    # assert len(_order_list) != 0
                    if event.order_side == OrderSide.SELL:
                        self.cancel_order(_order_list[-1])
                        logging.debug(f"撤止盈单 {_order_list[-1]}")
                    elif event.order_side == OrderSide.BUY:
                        self.cancel_order(_order_list[0])
                        logging.debug(f"撤止盈单 {_order_list[0]}")
                self.signal_orders.del_order(event.client_order_id)
                logging.debug(f"删除信号单: {event.client_order_id}")
                return
            
            # 挂止盈单
            if event.order_side == OrderSide.SELL:
                if self.quote_token == "BNB":
                    price = filled_price - self.instrument.price_increment
                else:
                    price = filled_price *(1 - self.stop_win_rate)
                win_order_id = self.limit_order(side=OrderSide.BUY, volume=filled_amount, price=price, stra_type='stop_win')
                logging.debug(f"止盈挂单完成: BUY")

            else:
                if self.quote_token == "BNB":
                    price = filled_price + self.instrument.price_increment
                else:
                    price = filled_price *(1 + self.stop_win_rate)
                win_order_id = self.limit_order(side=OrderSide.SELL, volume=filled_amount, price=price, stra_type='stop_win')
                logging.debug(f"止盈挂单完成: SELL")

        if event.client_order_id in self.signal_orders.orders.keys():
            self.signal_orders.del_order(event.client_order_id)
            logging.debug(f"删除信号单: {event.client_order_id}")
        elif event.client_order_id in self.stop_win_orders.orders.keys():
            self.stop_win_orders.del_order(event.client_order_id)
        

    def on_order_canceled(self, event: OrderCanceled):
        # 响应事件“订单已取消”
        # for record & plot
        self.stats_record.order_list.update(event)

        if event.client_order_id in self.signal_orders.orders.keys():
            self.signal_orders.del_order(event.client_order_id)
            logging.debug(f"删除信号单: {event.client_order_id}")
            
        elif event.client_order_id in self.stop_win_orders.orders.keys():
            self.stop_win_orders.del_order(event.client_order_id)
        
        logging.debug(f"订单取消:{event}")

        return 
    
    def on_order_expired(self, event: OrderExpired):
        # 响应事件“订单超时”
        # for record & plot
        self.stats_record.order_list.update(event)

        if event.client_order_id in self.signal_orders.orders.keys():
            self.signal_orders.del_order(event.client_order_id)
        elif event.client_order_id in self.stop_win_orders.orders.keys():
            self.stop_win_orders.del_order(event.client_order_id)
        
        logging.debug(f"订单超时:{event}")

        return 

    def on_order_denied(self, event: OrderDenied):
        logging.debug(f"订单失败:{event}")
        # for record & plot
        # self.stats_record.order_list.update(event)
        
        if event.client_order_id in self.signal_orders.orders.keys():
            self.signal_orders.del_order(event.client_order_id)
            logging.debug(f"订单失败::{event.client_order_id}")
        elif event.client_order_id in self.stop_win_orders.orders.keys():
            self.stop_win_orders.del_order(event.client_order_id)
        return

    def on_order_submitted(self, event: OrderSubmitted):
        # for record & plot
        # self.stats_record.order_list.update(event)
        
        return 

    def get_order_place_price(self, side):
        '''
            获取关仓价格：信号优先
               当前方向信号弱：保守挂单，反之，激进贴盘口
        '''
        if self.signal_value_buy > self.signal_value_sell:
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
        
        not_stoped_orderids = self.signal_orders.get_not_stoped_order_ids(side)
        
        if len(not_stoped_orderids) >= self.max_open_orders:
            return check_ret, f"s:{side} 方向未完成订单{not_stoped_orderids};{len(not_stoped_orderids)}超过限制 停止挂单"
        
        return True, ''

    def is_pos_control(self, side):
        if (side == OrderSide.BUY and self.pos <= self.max_pos):
            return True
        elif (side == OrderSide.SELL and self.pos >= -self.max_pos):
            return True
        else:
            return False

    # -------------------------------order function--------------------------------- #

    def limit_order(self, side, volume, price, post_only=True, reduce_only=False, expire_time=None, life_time=None, stra_type=None):
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
        self.stats_record.order_list.register(order, self.symbol_info, is_maker=True)
        
        # NOTE: order id is useful for further processing, eg. cancel order
        _order_id = order.client_order_id
        self.submit_order(order)

        if not order.is_closed:
            if stra_type == 'signal':
                self.signal_orders.add_order(price, side, volume, _order_id)
            elif stra_type == 'stop_win':
                self.stop_win_orders.add_order(price, side, volume, _order_id)

        self.limit_order_time += time.time() - t1

        return _order_id

    def market_order(self, side, volume, reduce_only=False):
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=self.instrument.make_qty(volume),
            reduce_only=reduce_only,
            time_in_force=TimeInForce.GTC,  # Use Fill or Kill for market orders
            exec_algorithm_id=None,
            exec_algorithm_params=None
        )
        logging.debug(f"发送market订单: {order}")
        
        # for record & plot
        # register_order(self.symbol_info, order)
        self.stats_record.order_list.register(order, self.symbol_info, is_maker=False)
        
        # NOTE: order id is useful for further processing, eg. cancel order
        _order_id = order.client_order_id
        self.submit_order(order)

        return _order_id
        

    # -------------------------------feature function--------------------------------- #

    def load_feature_data(self, path: str, date) -> pd.DataFrame:
        file_path = os.path.join(path, f"feature_{date.strftime('%Y%m%d')}.parquet")

        df = pd.read_parquet(file_path)
        df.columns = [row.replace('.','_') for row in df.columns]

        df['timestamp'] = df['timestamp'] / FEATURE_TIME_COEFFICIENT  # 注意 默认feature data default ms unit
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

            if self.curr_row is None:
                self.curr_row = next(self.feature_data)
                return None

            if timestamp - self.curr_row.timestamp < 0:
                # print(f'curr_row: {self.curr_row["timestamp"]}')
                # print(f'timestamp: {timestamp}')
                return None
                
            while (timestamp - self.curr_row.timestamp >= self.feature_interval):
                self.curr_row = next(self.feature_data)

            # print(f'curr_row: {self.curr_row.timestamp}')
            # print(f'timestamp: {timestamp}')
            return self.curr_row
            
        return None

    def calculate_feature_percentiles(self, date: str):
        date = int(date.strftime('%Y%m%d'))

        percentiles = {}

        for factor in self.factors:
            threshold_row = self.threshold_data[(self.threshold_data['factor']==factor) & (
                self.threshold_data['time']==date)]
                
            percentiles[factor] = {
                "90%": threshold_row['90%'].iloc[0],
                "95%": threshold_row['95%'].iloc[0],
                "97%": threshold_row['97%'].iloc[0],
                "98%": threshold_row['98%'].iloc[0],
                "99%": threshold_row['99%'].iloc[0],
                "10%": threshold_row['10%'].iloc[0],
                "5%": threshold_row['5%'].iloc[0],
                "3%": threshold_row['3%'].iloc[0],
                "2%": threshold_row['2%'].iloc[0],
                "1%": threshold_row['1%'].iloc[0],
                "mean": threshold_row['mean'].iloc[0],
                "std": threshold_row['std'].iloc[0],
                "1sigma": threshold_row['1sigma'].iloc[0],
                "-1sigma": threshold_row['-1sigma'].iloc[0],
                "2sigma": threshold_row['2sigma'].iloc[0],
                "-2sigma": threshold_row['-2sigma'].iloc[0],
                "3sigma": threshold_row['3sigma'].iloc[0],
                "-3sigma": threshold_row['-3sigma'].iloc[0],
            }
        # print(date)
        # print(threshold_row.name)
        # print(percentiles)
        return percentiles


