import pickle
import pandas as pd
import json
import logging
from typing import Union
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import time
import pytz
import tomli
import random
import os
from collections import defaultdict, deque
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore")

from strategystats.stat_data import SymbolInfo
from strategystats.stra_stat import StrategyStatsRecord, plot_from_npz
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
from nautilus_trader.model.events import PositionChanged, PositionClosed, PositionOpened, OrderFilled, OrderCanceled, OrderExpired, OrderRejected, OrderSubmitted
from nautilus_trader.model.identifiers import InstrumentId, ClientOrderId, StrategyId, AccountId, Venue
from nautilus_trader.model.instruments import Instrument 
from nautilus_trader.model.functions import position_side_to_str 
from nautilus_trader.model.objects import Price, Quantity, Currency
from nautilus_trader.trading.strategy import Strategy 
from nautilus_trader.model.enums import OrderSide, AggressorSide, PositionSide, TimeInForce, TriggerType 


# -------------------------------utils function--------------------------------- #
def convert_timestamp(ts_event_ms, adjust_time_zone = 0): # ts_event_ms是毫秒
    ts_event_ms = ts_event_ms  + adjust_time_zone * 3600 * 1000
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
        # init_balance = initial_balance(self.symbol_info, config.starting_balance, self.start_time.value)
        # self.stats_record = StrategyStatsRecord(init_balance)
        self.plot_config = config.plot_config

        self.instrument_id = config.instrument_id                                   # 交易对
        self.quote_token = self.symbol_info.token                                   # eg: BTC
        self.base_token = self.symbol_info.quote
        self.init_pos = init_balance.capitals[self.symbol_info.token]
        self.order_amount = config.order_amount                                     # 下单数量
        self.save_path = config.save_path                                           # 存储目录
        self.instrument: Instrument | None = None                                   # 交易对
        self.account_id: AccountId | None = None                                    # 账户
        
        self.initial_date = self.start_time - pd.Timedelta(days=1)                  # 初始日期
        self.venue = Venue(config.instrument_id.venue.value)                        # 交易所

        self.orders = {}                                                            # 记录所有的订单    
        self.open_positions = {}                                                    # 持仓订单
        self.open_to_close_order_map = {}                                           # 开仓订单及对应的平仓订单
        self.close_to_open_order_map = {}                                           # 平仓订单及对应的开仓订单 for record
        self.submit_open_orders = {}
        self.submit_close_orders = {}
        self.filled_open_orders = {}
        self.filled_close_orders = {}
        self.unfilled_orders = {}
        self.order_list = {}
        self.open_win_map = {}
        self.last_time = 0
        self.buy_signal = 0
        self.sell_signal = 0
        self.signal_amounts = 0
        self.signal_filled_amounts = 0
        self.submit_num = 0
        self.cancel_num = 0
        self.signal_num = 0
        self.signal_submit_num = 0
        self.order_record = defaultdict(list)
        self.signal_info_records = []
        self.curr_signal_info = {}

        self.price_interval = config.price_interval
        

        self.holding_time = config.holding_time * 1000                                           # 最大持仓时间（ms）
        self.pos = 0
        self.first_signal = 0
        self.last_signal = np.inf
        self.max_price = None
        self.min_price = None
        self.open_price = None
        self.last_ask_price = np.inf
        self.last_bid_price = -np.inf
        self.order_expire_time = {}
        self.not_stop_win_order = {None}
        self.open_orders = {None}

        self.stop_win_rate = config.stop_win_rate
        self.stop_loss_rate = config.stop_loss_rate

        self.factor_type = config.factor_type
        self.factors = config.factors  # 因子名称
        self.date_range =  pd.date_range(start=self.start_time-timedelta(days=1), end=self.end_time+timedelta(days=1), freq='D')
        self.current_date_index = list(self.date_range).index(pd.Timestamp(self.start_time))
        self.feature_data = None
        self.feature_data_max_timestamp = 0
        self.feature_interval = 100 if self.factor_type != 'LobImbalance' else 101
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

        # store the excution data
        self.latest_data = {
            "quote": {},
            "trade": {},
            "orderbook": {},
        }
        self.recent_quote_data = deque(maxlen=60)

        self.run_start_time = None
        self.run_end_time = None

    # -------------------------------event function--------------------------------- #
    def on_start(self):
        self.run_start_time = time.time()
        
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.instrument_id}")
            self.stop()
            return

        self.account_id = self.cache.account_id(self.venue)
        print('------------', self.venue)
        print('------------', self.account_id)
        print('------------', self.instrument_id)

        self.clock.set_timer(name='calculte_signal', interval=pd.Timedelta(milliseconds=100), callback=lambda e: self.on_signal(e))
        
        # bar_spec = BarSpecification.from_timedelta(self.time_interval, PriceType.MID)
        # bar_type = BarType(instrument_id=self.instrument_id, bar_spec=bar_spec, aggregation_source=AggregationSource.INTERNAL)
        # self.subscribe_bars(bar_type=bar_type)
        # self.subscribe_trade_ticks(self.instrument_id)
        self.subscribe_quote_ticks(self.instrument_id)
        # self.subscribe_order_book_at_interval(self.instrument_id, interval_ms=INTERVAL_MS)

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

        _account = self.portfolio.account(self.instrument_id.venue)
        _pos = _account.balance(Currency.from_str(self.quote_token))
        _pos_total = float(_pos.total) - self.init_pos
        if _pos_total != 0:
            # self.stats_record.order_list.register(from_on_stop(
            # # self.stats_record.order_list.append(from_on_stop(
            #     instrument_id = None,
            #     symbol=self.symbol_info, 
            #     close_amount=abs(_pos_total),
            #     close_price=float(self.bid_price) if _pos_total > 0 else float(self.ask_price),
            #     close_side=OrderSide.BUY if _pos_total < 0 else OrderSide.SELL,
            #     close_ts=self.end_time.timestamp() * 1e9,
            #     commissions_rate=float(self.instrument.taker_fee),
            # ), self.symbol_info, is_maker=True)
        
            close_side = OrderSide.BUY if _pos_total < 0 else OrderSide.SELL
            close_amount = abs(_pos_total)
            close_ts = self.end_time.timestamp() * 1e9
            close_price = float(self.bid_price) if _pos_total > 0 else float(self.ask_price)
            order_id = f"on_stop_close_all_order_{ts_to_millisecond(close_ts)}"
            self.stats_record.order_list.append(order_id,
                                                status=OrderStatus.FILLED,
                                                price=close_price,
                                                amount=close_amount,
                                                side=close_side,
                                                trade_type=TradeType.TAKER,
                                                order_type=OrderType.MARKET,
                                                filled_price=close_price,
                                                filled_amount=close_amount,
                                                commission=float(self.instrument.taker_fee) * close_amount * close_price,
                                                ts=ts_to_millisecond(close_ts),
                                                )

        self.run_end_time = time.time()
        self.log.warning(f"time needed in mins: {(self.run_end_time - self.run_start_time) / 60}")

        # for record & plot
        # self.stats_record.plot(self.plot_config, self.save_path)
        # print(f'signal filled prob: {round(float(self.signal_filled_amounts)/float(self.signal_amounts), 7)*100}%')
        # print(f'submit num: {self.submit_num}')
        # print(f'cancel num: {self.cancel_num}')
        self.stats_record.to_npz(self.save_path)
        # plot_from_npz(self.plot_config, self.save_path)
        # run_end_time_ = time.time()
        # print((run_end_time_ - self.run_end_time) / 60)

        # self.report_plot()

    # 处理bar数据
    def on_bar(self, bar: Bar) -> None:
        pass

    def on_signal(self, event: TimeEvent):
        UTC_time = event.ts_event / 1000 / 1000
        
        """strategy calculate"""

        # trading_flag = False
        feature_value = self.get_feature_value(UTC_time)

        # print(self.last_feature_value)
        _account = self.portfolio.account(self.instrument_id.venue)
        _pos = _account.balance(Currency.from_str(self.quote_token))
        _pos_total = float(_pos.total) - self.init_pos
        self.pos = np.round(_pos_total, 5)

        signal_buy, signal_sell = False, False
        if feature_value is not None:
            if self.factor_type == 'vinci_reg':
                if feature_value.reg_preds_t3_2s_d16 > self.threshold['above']:
                    signal_value_buy = feature_value.reg_preds_t3_2s_d16
                    signal_buy = True
                elif feature_value.reg_preds_t3_2s_d15 < self.threshold['below']:
                    signal_value_sell = feature_value.reg_preds_t3_2s_d15
                    signal_sell = True
            elif self.factor_type == 'vinci_cls':
                if feature_value.cls_preds_t3_2s_d0 > self.threshold['above']:
                    signal_value_buy = feature_value.cls_preds_t3_2s_d0
                    signal_buy = True
                elif feature_value.cls_preds_t3_2s_d1 > self.threshold['below']:
                    signal_value_sell = feature_value.cls_preds_t3_2s_d1
                    signal_sell = True
            elif self.factor_type == 'LobImbalance':
                if feature_value.LobImbalance_0_5 < self.feature_percentiles['LobImbalance_0_5'][self.threshold['below']]:
                    signal_value_buy = feature_value.LobImbalance_0_5
                    signal_buy = True
                elif feature_value.LobImbalance_0_5 > self.feature_percentiles['LobImbalance_0_5'][self.threshold['above']]:
                    signal_value_sell = feature_value.LobImbalance_0_5
                    signal_sell = True
            elif self.factor_type == 'vinci_maker_label':
                if feature_value.maker_cls_up_d3 > self.threshold['above']:
                    signal_value_buy = feature_value.maker_cls_up_d3
                    signal_buy = True
                if feature_value.maker_cls_down_d3 > self.threshold['below']:
                    signal_value_sell = feature_value.maker_cls_down_d3
                    signal_sell = True
            elif self.factor_type == 'vinci_mixer_reg':
                if feature_value.reg_preds_t12_8s_d0 > self.threshold['above']:
                    signal_value_buy = feature_value.reg_preds_t12_8s_d0
                    signal_buy = True
                elif feature_value.reg_preds_t12_8s_d1 < self.threshold['below']:
                    signal_value_sell = feature_value.reg_preds_t12_8s_d1
                    signal_sell = True
            elif self.factor_type == 'vinci_mixer_cls':
                # self.log.warning(f'feature: {feature_value.cls_preds_t12_8s_d0} {feature_value.cls_preds_t12_8s_d1}')
                if feature_value.cls_preds_t12_8s_d0 > self.threshold['above']:
                    signal_value_buy = feature_value.cls_preds_t12_8s_d0
                    signal_buy = True
                elif feature_value.cls_preds_t12_8s_d1 > self.threshold['below']:
                    signal_value_sell = feature_value.cls_preds_t12_8s_d1
                    signal_sell = True

            elif self.factor_type == 'xuefeng':
                if getattr(feature_value, f'{self.quote_token.lower()}_pred') > self.threshold['above']:
                    signal_value_buy = getattr(feature_value, f'{self.quote_token.lower()}_pred')
                    signal_buy = True
                elif getattr(feature_value, f'{self.quote_token.lower()}_pred') < self.threshold['below']:
                    signal_value_sell = getattr(feature_value, f'{self.quote_token.lower()}_pred')
                    signal_sell = True
            elif self.factor_type == 'vinci_tsmixer_reg':
                if feature_value.reg_preds_t3_2s_d16 > self.threshold['above']:
                    signal_value_buy = feature_value.reg_preds_t3_2s_d16
                    signal_buy = True
                elif feature_value.reg_preds_t3_2s_d15 < self.threshold['below']:
                    signal_value_sell = feature_value.reg_preds_t3_2s_d15
                    signal_sell = True
            elif self.factor_type == 'vinci_tsmixer_cls':
                if feature_value.pred_cls_t3_2s_d0 > self.threshold['above']:
                    signal_value_buy = feature_value.pred_cls_t3_2s_d0
                    signal_buy = True
                elif feature_value.pred_cls_t3_2s_d1 > self.threshold['below']:
                    signal_value_sell = feature_value.pred_cls_t3_2s_d1
                    signal_sell = True
            elif self.factor_type == 'xuefeng_0926':
                if feature_value.SignalModel01 > self.threshold['above']:
                    signal_value_buy = feature_value.SignalModel01
                    signal_buy = True
                elif feature_value.SignalModel01 < self.threshold['below']:
                    signal_value_sell = feature_value.SignalModel01
                    signal_sell = True


        # 满足开仓条件
        if signal_buy:
            open_side = "buy"
            order_amount = self.order_amount
            open_order_id = self.market_order(side=open_side, volume=order_amount, reduce_only=False)
            self.open_to_close_order_map[open_order_id] = None
            self.submit_open_orders[open_order_id] = {'timestamp': UTC_time, 'side': open_side}
            # self.log.warning(f'[submit order{open_order_id}]: time: {UTC_time}; amount: {order_amount}; side: {open_side}, direction: open')
            
        if signal_sell:
            open_side = "sell"
            order_amount = self.order_amount
            open_order_id = self.market_order(side=open_side, volume=order_amount, reduce_only=False)
            self.open_to_close_order_map[open_order_id] = None
            self.submit_open_orders[open_order_id] = {'timestamp': UTC_time, 'side': open_side}
            # self.log.warning(f'[submit order{open_order_id}]: time: {UTC_time}; amount: {order_amount}; side: {open_side}, direction: open')
            

        # 检查是否满足平仓条件
        for open_order_id, info in list(self.open_positions.items()):
            open_price = info['filled_price']
            open_order_clock = info["timestamp"]
            order_amount = info["filled_amount"]
            open_side = info["side"]
            if (
                UTC_time - open_order_clock > self.holding_time
            ):
                trading_flag = True
                close_side = "buy" if open_side == "sell" else "sell"
                close_order_id = self.market_order(side=close_side, volume=order_amount, reduce_only=False)
                self.close_to_open_order_map[close_order_id] = open_order_id
                del self.open_positions[open_order_id]
                self.submit_close_orders[close_order_id] = {'timestamp': UTC_time, 'side': close_side}
                # self.log.warning(f'[submit order{close_order_id}]: time: {UTC_time}; amount: {order_amount}; side: {close_side}, direction: close')
                

        return

    # 处理DEPTH/Order_book/Snapshots数据
    # 这个是可以获取到深度数据的，并且深度数据是唯一可以设置时间戳间隔的数据，因此我们的因子测试如果是在相同时间间隔下测试的话，可以使用这个数据
    def on_order_book(self, orderbook: OrderBook) -> None:
        # UTC_time = orderbook.ts_event / 1000 / 1000   
        
        # ask_price, ask_size = get_depth_level(orderbook, 0, "asks")
        # bid_price, bid_size = get_depth_level(orderbook, 0, "bids")
        # ask_data = get_depth_all(orderbook, "asks")
        # bid_data = get_depth_all(orderbook, "bids")
        # mid_price = round((ask_price + bid_price) / 2, 2)  
        
        return
    
    # 处理QUOTE/TICK数据
    def on_quote_tick(self, ticker: QuoteTick) -> None:
        self.ask_price = ticker.ask_price
        self.bid_price = ticker.bid_price
        # for record & plot
        if len(self.stats_record.price_list) == 0 or (
            self.stats_record.price_list[-1].ts_event < ticker.ts_event // 1e6 - self.price_interval + 1 and 
            self.stats_record.price_list[-1].mid_price.price != (ticker.ask_price+ticker.bid_price)/2):
            # self.stats_record.price_list[-1].ts_event < ticker.ts_event):
            self.stats_record.price_list.append(from_quote_tick(ticker))

        return
    
    # 处理TRADE数据
    def on_trade_tick(self, trade: TradeTick) -> None:
        return

    def on_order_filled(self, event: OrderFilled):
        # 响应事件“订单完成”
        side = 'buy' if event.order_side == OrderSide.BUY else 'sell' if event.order_side == OrderSide.SELL else None
        filled_amount = event.last_qty.as_decimal()
        filled_price = event.last_px.as_decimal()
        order_id = event.client_order_id
        fee = event.commission.as_decimal()
        update_timestamp = event.ts_event /1000 / 1000 # ts_event是纳秒,转换为毫秒
        # readable_time = convert_timestamp(update_timestamp, 8)
        direction = 'open' if order_id in self.submit_open_orders else 'close' if order_id in self.submit_close_orders else None

        if direction == 'open':
            if order_id in self.filled_open_orders:
                amount_sum = filled_amount + self.filled_open_orders[order_id]["filled_amount"]
                price_avg = (filled_price * filled_amount + self.filled_open_orders[order_id]["filled_price"] * self.filled_open_orders[order_id]["filled_amount"]) / amount_sum
            else:
                amount_sum = filled_amount
                price_avg = filled_price
            self.filled_open_orders[order_id] = {"timestamp": update_timestamp, "filled_price": price_avg, "filled_amount": amount_sum, "side": side}
            self.open_positions[order_id] = {"timestamp": update_timestamp, "filled_price": price_avg, "filled_amount": amount_sum, "side": side}
        if direction == 'close':
            open_id = self.close_to_open_order_map[order_id]
            self.open_to_close_order_map[open_id] = order_id
            if order_id in self.filled_close_orders:
                amount_sum = filled_amount + self.filled_close_orders[order_id]["filled_amount"]
                price_avg = (filled_price * filled_amount + self.filled_close_orders[order_id]["filled_price"] * self.filled_close_orders[order_id]["filled_amount"]) / amount_sum
            else:
                amount_sum = filled_amount
                price_avg = filled_price
            self.filled_close_orders[order_id] = {"timestamp": update_timestamp, "filled_price": price_avg, "filled_amount": amount_sum, "side": side}
    
        # self.log.warning(f'[filled order{order_id}]: time: {update_timestamp}; price: {filled_price}; amount: {filled_amount}; side: {side}, direction: {direction}')

        # for record & plot
        # order_info = from_order_event(event)
        # self.stats_record.order_list.append(order_info)
        self.stats_record.order_list.update(event)
        


    def on_order_rejected(self, event: OrderRejected):
        # 响应事件订单已拒绝
        return


    def on_order_canceled(self, event: OrderCanceled):
        # 响应事件“订单已取消”
        self.cancel_num += 1
        # for record & plot
        # order_info = from_order_event(event)
        # self.stats_record.order_list.append(order_info)
        self.stats_record.order_list.update(event)
        

        return 
    
    def on_order_expired(self, event: OrderExpired):
        # 响应事件“订单超时”
        # win_order_id = self.open_win_map.get(event.client_order_id)
        # if win_order_id is not None:
        #     self.cancel_order(self.order_list.get(win_order_id))
        self.cancel_num += 1
        self.stats_record.order_list.update(event)
        
        return 

    def on_order_submitted(self, event: OrderSubmitted):
        self.submit_num += 1
        # for record & plot
        # order_info = from_order_event(event)
        # self.stats_record.order_list.append(order_info)
        self.stats_record.order_list.update(event)
        
    

    # -------------------------------order function--------------------------------- #

    # TODO：优化写法

    def limit_order(self, side, volume, price, post_only=True, reduce_only=False, expire_time=None, life_time=None):
        trade_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
        # order = self.order_factory.limit(
        #         instrument_id=self.instrument_id,
        #         order_side=trade_side,
        #         price=Price(price, precision=self.instrument.price_precision),
        #         quantity=self.instrument.make_qty(volume),
        #         post_only=post_only,
        #         reduce_only=reduce_only,
        #         time_in_force=TimeInForce.GTC,
        #         emulation_trigger=TriggerType["NO_TRIGGER"]
        # )

        if expire_time is None:
            expire_time = self.clock.utc_now() + pd.Timedelta(milliseconds=life_time)

        order = self.order_factory.limit(
            instrument_id=self.instrument_id,
            order_side=trade_side,
            price=Price(price, precision=self.instrument.price_precision),
            quantity=self.instrument.make_qty(volume),
            post_only=post_only,
            reduce_only=reduce_only,
            time_in_force=TimeInForce.GTD,
            expire_time=expire_time,
            emulation_trigger=TriggerType["NO_TRIGGER"]
        )
        
        # for record & plot
        # register_order(self.symbol_info, order)
        self.stats_record.order_list.register(order, self.symbol_info, is_maker=True)
        
        
        # NOTE: order id is useful for further processing, eg. cancel order
        _order_id = order.client_order_id
        self.submit_order(order)
        self.order_list[_order_id] = order

        self.order_expire_time[_order_id] = expire_time

        return _order_id

    def market_order(self, side, volume, reduce_only=False):
        trade_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=trade_side,
            quantity=self.instrument.make_qty(volume),
            reduce_only=reduce_only,
            time_in_force=TimeInForce.GTC,  # Use Fill or Kill for market orders
            exec_algorithm_id=None,
            exec_algorithm_params=None
        )
        # for record & plot
        # register_order(self.symbol_info, order)
        self.stats_record.order_list.register(order, self.symbol_info, is_maker=False)
        
        
        # NOTE: order id is useful for further processing, eg. cancel order
        _order_id = order.client_order_id
        self.submit_order(order)
        self.order_list[_order_id] = order

        return _order_id
        

    # -------------------------------feature function--------------------------------- #
    
    # Load the feature data
    def load_initial_feature_data(self):
        initial_date = self.initial_date
        self.feature_data = self.load_feature_data(self.factor_path, initial_date)
        # self.feature_percentiles = self.calculate_feature_percentiles(initial_date)

    def load_feature_data(self, path: str, date) -> pd.DataFrame:
        file_path = os.path.join(path, f"feature_{date.strftime('%Y%m%d')}.parquet")

        df = pd.read_parquet(file_path)
        df.columns = [row.replace('.','_') for row in df.columns]
        df['timestamp'] = df['timestamp'] / 1000 / 1000
        # df.set_index('timestamp', inplace=True)
        self.feature_data_max_timestamp = df['timestamp'].iloc[-1]

        iter_df = iter_tool(df)

        return iter_df

    def update_feature_data(self, timestamp):
        if self.feature_data is None or timestamp - self.feature_data_max_timestamp >= 100:
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

            # print(f'curr_row: {self.curr_row["timestamp"]}')
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
