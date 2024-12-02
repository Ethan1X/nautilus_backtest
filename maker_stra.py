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

from utils.helper import iter_tool

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

        self.orders = {}                                                            # 记录所有的订单    
        self.open_positions = {}                                                    # 持仓订单
        self.open_to_close_order_map = {}                                           # 开仓订单及对应的平仓订单
        self.close_to_open_order_map = {}                                           # 平仓订单及对应的开仓订单 for record
        self.submit_open_orders = {}
        self.submit_close_orders = {}
        self.filled_open_orders = {}
        self.filled_close_orders = {}
        self.unfilled_orders = {}
        # self.order_list = {}
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
        self.stop_loss_taker_threshold = config.stop_loss_taker_threshold

        self.ask_price = None
        self.bid_price = None

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
        self.order_side_dict = {}
        self.order_amount_dict = {}
        # 订单策略ID，多笔order可能对应一笔策略ID
        self.order_strategy_id = {}
        self.not_stop_win_order = set()
        self.open_orders = set()

        self.stop_win_rate = config.stop_win_rate
        self.stop_loss_rate = config.stop_loss_rate
        self.stop_loss_flag = False if config.stop_loss_type == 'NONE' else True
        self.stop_loss_type = config.stop_loss_type

        self.price_interval = config.price_interval
        
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

        self.clock.set_timer(name='calculte_signal', interval=pd.Timedelta(milliseconds=100), start_time=self.start_time, callback=lambda e: self.on_signal(e))
        
        self.clock.set_timer(name='update', interval=pd.Timedelta(milliseconds=1), start_time=self.start_time, callback=lambda e: self.on_update_ms(e))
        
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

        if self.pos != 0:
            # self.stats_record.order_list.append(from_on_stop(
            # # self.stats_record.order_list.append(from_on_stop(
            #     instrument_id=self.instrument_id,
            #     symbol=self.symbol_info, 
            #     close_amount=abs(self.pos),
            #     close_price=float(self.bid_price) if self.pos > 0 else float(self.ask_price),
            #     close_side=OrderSide.BUY if self.pos < 0 else OrderSide.SELL,
            #     close_ts=self.end_time.timestamp() * 1e9,
            #     commissions_rate=float(self.instrument.taker_fee),
            # ), self.symbol_info, is_maker=True)
            
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

        # np.savez(f"./{self.save_path}/order_strategy_type.npz", 
        #          order_ts=np.array(self.order_record['order_ts']), 
        #          order_id=np.array(self.order_record['order_id']), 
        #          strategy_type=np.array(self.order_record['strategy_type']),
        #          strategy_id=np.array(self.order_record['strategy_id']),
        #          order_info=self.order_record['order_info']
        #          )
        np.savez_compressed(f"./{self.save_path}/order_strategy_type.npz", **self.order_record)
        logging.warning(f"save npz and stat plot time needed in mins: {(time.time()- self.run_end_time) / 60}")

    # 处理bar数据
    def on_bar(self, bar: Bar) -> None:
        pass

    def record_order_type(self, ts, order_id, strategy_id, strategy_type, order_info={}):
        self.order_record['order_ts'].append(ts)
        self.order_record['order_id'].append(order_id)
        self.order_record['strategy_id'].append(strategy_id)
        self.order_record['strategy_type'].append(strategy_type)
        self.order_record['order_info'].append(order_info)

    def on_signal(self, event: TimeEvent):
        t1 = time.time()
        UTC_time = event.ts_event / 1000 / 1000
        if self.ask_price is None or self.bid_price is None:
            return
        """strategy calculate"""
        logging.debug(f"on_signal 计算: {UTC_time}")

        feature_value = self.get_feature_value(UTC_time)
        self.on_signal_get_feature_value_time += time.time() - t1

        # 先判断是否要止损

        if self.stop_loss_flag and len(self.not_stop_win_order - self.open_orders) != 0:
            return
        
        if self.stop_loss_flag and abs(self.pos) > 1e-10 and self.stop_loss():

            _order_list = self.cache.orders_open()
            for order in _order_list:
                self.cancel_order(order)
            if self.pos < 0:
                price = self.ask_price - self.instrument.price_increment
            else:
                price = self.bid_price + self.instrument.price_increment
            if 'alltaker' in self.stop_loss_type:
                close_order_id = self.market_order(side=OrderSide.BUY if self.pos < 0 else OrderSide.SELL, volume=abs(self.pos), reduce_only=False)
            else:
                close_order_id = self.limit_order(side=OrderSide.BUY if self.pos < 0 else OrderSide.SELL, volume=abs(self.pos), 
                                                        price=price, 
                                                        post_only=True, reduce_only=False, expire_time=None, life_time=1000 * 60)
            logging.debug(f"止损发单完成")
            self.not_stop_win_order.add(close_order_id)

            self.record_order_type(UTC_time, close_order_id, close_order_id, "stop_loss")
            self.order_strategy_id[close_order_id] = close_order_id

            self.on_signal_time += time.time() - t1
            
            return

        signal_buy, signal_sell = False, False
        if feature_value is not None:
            if self.factor_type == 'vinci_reg':
                signal_value_buy = feature_value.reg_preds_t3_2s_d16
                signal_value_sell = feature_value.reg_preds_t3_2s_d15
                if feature_value.reg_preds_t3_2s_d16 > self.threshold['above']:
                    signal_buy = True
                elif feature_value.reg_preds_t3_2s_d15 < self.threshold['below']:
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
            elif self.factor_type == 'vinci_maker_label_mixer' or self.factor_type == 'vinci_maker_gte0ms_stoplossTrue' or self.factor_type == 'vinci_maker_gte0ms_stoplossTrue_equal' or self.factor_type == 'vinci_maker_gte3ms_stoplossTrue' or self.factor_type == 'vinci_maker_gte3ms_stoplossFalse' or self.factor_type == 'vinci_maker_1021':
                signal_value_buy = feature_value.maker_cls_up_d1
                signal_value_sell = feature_value.maker_cls_down_d1
                if feature_value.maker_cls_up_d1 > self.threshold['above']:
                    signal_buy = True
                if feature_value.maker_cls_down_d1 > self.threshold['below']:
                    signal_sell = True
            elif self.factor_type == 'vinci_maker_online':
                if feature_value.maker_cls_down_d1 > self.threshold['above']:
                    signal_value_buy = feature_value.maker_cls_down_d1
                    signal_buy = True
                if feature_value.maker_cls_up_d1 > self.threshold['below']:
                    signal_value_sell = feature_value.maker_cls_up_d1
                    signal_sell = True
            elif self.factor_type == 'xuefeng_0926':
                signal_value_buy = feature_value.SignalModel01
                signal_value_sell = feature_value.SignalModel01
                if feature_value.SignalModel01 > self.threshold['above']:
                    signal_buy = True
                elif feature_value.SignalModel01 < self.threshold['below']:
                    signal_sell = True

        buy_order_price =  self.ask_price-self.instrument.price_increment
        sell_order_price = self.bid_price+self.instrument.price_increment
        if signal_buy and signal_sell:
            if signal_value_buy < signal_value_sell:
                buy_order_price = self.bid_price
            else:
                sell_order_price = self.ask_price

        
        # 满足开仓条件
        if signal_buy:
            open_order_id = self.limit_order(side=OrderSide.BUY, volume=self.order_amount, 
                                             price=buy_order_price, 
                                             post_only=True, reduce_only=False, expire_time=None, life_time=100)
            logging.debug(f"信号发单完成: BUY")

            self.not_stop_win_order.add(open_order_id)
            self.open_orders.add(open_order_id)
       
            # self.record_order_type(UTC_time, open_order_id, open_order_id, "signal_open", order_info={'signal_value': signal_value_buy, 'stop_loss': stop_loss})
            self.record_order_type(UTC_time, open_order_id, open_order_id, "signal_open", order_info={'signal_value': signal_value_buy})
            self.order_strategy_id[open_order_id] = open_order_id
            
        if signal_sell:
            open_order_id = self.limit_order(side=OrderSide.SELL, volume=self.order_amount, 
                                             price=sell_order_price, 
                                             post_only=True, reduce_only=False, expire_time=None, life_time=100)
            logging.debug(f"信号发单完成: SELL")

            self.not_stop_win_order.add(open_order_id)
            self.open_orders.add(open_order_id)
            
            # TODO: signal value buy 什么时候补充
            # self.record_order_type(UTC_time, open_order_id, open_order_id, "signal_open", order_info={'signal_value': signal_value_sell, 'stop_loss': stop_loss})
            self.record_order_type(UTC_time, open_order_id, open_order_id, "signal_open", order_info={'signal_value': signal_value_sell})
            self.order_strategy_id[open_order_id] = open_order_id
  
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
        
        # if len(self.open_orders) > 1:
        #     logging.debug(f"信号单数：{len(self.open_orders)}, {self.open_orders}")        

        _order_list = self.cache.orders_open()
        if len(_order_list) > 0:
            for order in _order_list:
                
                if order.is_pending_cancel:
                    continue

                if order.client_order_id in self.open_orders and len(self.open_orders) > 1:
                    continue
                
                if order.client_order_id in self.not_stop_win_order:
                    # 有止损单且没有仓位
                    if self.pos == 0 and order.client_order_id not in self.open_orders:
                        self.cancel_order(order)
                        continue
                    
                    if order.side == OrderSide.SELL and self.bid_price < self.last_bid_price:
                        expire_time = self.order_expire_time[order.client_order_id]
                        self.cancel_order(order)
                        if expire_time > self.clock.utc_now():
                            if (order.client_order_id not in self.open_orders
                                and self.stop_loss_taker_threshold != 0
                                # and self.bid_price / self.open_price - 1 > self.stop_loss_taker_threshold
                                and self.ask_price / self.open_price - 1 < - self.stop_loss_taker_threshold
                                ):
                                _order_id = self.market_order(side=OrderSide.SELL, volume=order.leaves_qty, reduce_only=False)
                                # _order_id = self.limit_order(side=OrderSide.SELL, volume=order.leaves_qty, price=self.bid_price+self.instrument.price_increment, 
                                #                                  post_only=True, reduce_only=False, expire_time=expire_time, life_time=None)
                            else:
                                _order_id = self.limit_order(side=OrderSide.SELL, volume=order.leaves_qty, price=self.bid_price+self.instrument.price_increment, 
                                                                 post_only=True, reduce_only=False, expire_time=expire_time, life_time=None)
                            # _order_id = self.limit_order(side=OrderSide.SELL, volume=order.leaves_qty, price=self.bid_price+self.instrument.price_increment, 
                            #                                      post_only=True, reduce_only=False, expire_time=expire_time, life_time=None)
                                
                            self.not_stop_win_order.add(_order_id)

                            if order.client_order_id in self.open_orders:
                                self.open_orders.add(_order_id)
                                self.record_order_type(UTC_time, _order_id, self.order_strategy_id[order.client_order_id], "signal_open")
                                self.order_strategy_id[_order_id] = self.order_strategy_id[order.client_order_id]
                                del self.order_strategy_id[order.client_order_id]
                                logging.debug(f"信号追单完成: SELL")
                            else:
                                self.record_order_type(UTC_time, _order_id, self.order_strategy_id[order.client_order_id], "stop_loss")
                                self.order_strategy_id[_order_id] = self.order_strategy_id[order.client_order_id]
                                del self.order_strategy_id[order.client_order_id]
                                logging.debug(f"止损追单完成: SELL")


                    elif order.side == OrderSide.BUY and self.ask_price > self.last_ask_price:
                        expire_time = self.order_expire_time[order.client_order_id]
                        self.cancel_order(order)

                        if expire_time > self.clock.utc_now():
                            # if order.client_order_id not in self.open_orders and self.open_price is None:
                            #     _account = self.portfolio.account(self.instrument_id.venue)
                            #     _pos = _account.balance(Currency.from_str(self.quote_token))
                            #     _pos_total = float(_pos.total) - self.init_pos
                            #     logging.debug(f'error!!!{order}; {self.pos}; {_pos_total}')
                            if (order.client_order_id not in self.open_orders 
                                and self.stop_loss_taker_threshold != 0
                                # and self.ask_price / self.open_price - 1 < -self.stop_loss_taker_threshold
                                and self.bid_price / self.open_price - 1 > self.stop_loss_taker_threshold
                                ):
                                _order_id = self.market_order(side=OrderSide.BUY, volume=order.leaves_qty, reduce_only=False)
                            else:
                                _order_id = self.limit_order(side=OrderSide.BUY, volume=order.leaves_qty, price=self.ask_price-self.instrument.price_increment, 
                                                             post_only=True, reduce_only=False, expire_time=expire_time, life_time=None)
                            # _order_id = self.limit_order(side=OrderSide.BUY, volume=order.leaves_qty, price=self.ask_price-self.instrument.price_increment, 
                            #                                  post_only=True, reduce_only=False, expire_time=expire_time, life_time=None)

                            self.not_stop_win_order.add(_order_id)

                            if order.client_order_id in self.open_orders:
                                self.open_orders.add(_order_id)
                                self.record_order_type(UTC_time, _order_id, self.order_strategy_id[order.client_order_id], "signal_open")
                                self.order_strategy_id[_order_id] = self.order_strategy_id[order.client_order_id]
                                del self.order_strategy_id[order.client_order_id]
                                logging.debug(f"信号追单完成: BUY")
                            else:
                                self.record_order_type(UTC_time, _order_id, self.order_strategy_id[order.client_order_id], "stop_loss")
                                self.order_strategy_id[_order_id] = self.order_strategy_id[order.client_order_id]
                                del self.order_strategy_id[order.client_order_id]
                                logging.debug(f"止损追单完成: BUY")


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

    def del_cache_orderid(self, orderid):
        if orderid in self.open_orders:
            self.open_orders.remove(orderid)
        if orderid in self.not_stop_win_order:
            self.not_stop_win_order.remove(orderid)

        try:
            del self.order_expire_time[orderid]
            del self.order_side_dict[orderid]
            del self.order_amount_dict[orderid]
        except Exception as e:
            logging.debug(f'error: {e}')
        
    def on_order_accepted(self, event: OrderSubmitted):
        # logging.debug(f"订单接受:{event}")
        
        return
            
    
    def on_order_rejected(self, event: OrderRejected):
        
        logging.debug(f"订单拒绝:{event}")
        
        order_id = event.client_order_id
        expire_time = self.order_expire_time[order_id]
        side = self.order_side_dict[order_id]
        amount = self.order_amount_dict[order_id]
        UTC_time = event.ts_event / 1e6
        self.stats_record.order_list.update(event)
        

        if (
            order_id in self.open_orders and len(self.open_orders) > 1
            or (order_id not in self.open_orders and self.pos == 0)
           ):
            self.del_cache_orderid(order_id)
            
            return

        
        if expire_time > self.clock.utc_now() and order_id in self.not_stop_win_order:
        
            if side == OrderSide.BUY:
                if (order_id not in self.open_orders
                    and self.stop_loss_taker_threshold != 0
                    and self.ask_price / self.open_price - 1 < -self.stop_loss_taker_threshold):
                    new_order_id = self.market_order(side=OrderSide.BUY, volume=amount, reduce_only=False)
                else:
                    new_order_id = self.limit_order(side=OrderSide.BUY, volume=amount, price=self.ask_price-self.instrument.price_increment, 
                                                 post_only=True, reduce_only=False, expire_time=expire_time, life_time=None)
                # new_order_id = self.limit_order(side=side, volume=amount, 
                #                                 price=self.ask_price-self.instrument.price_increment,
                #                                 post_only=True, reduce_only=False, expire_time=expire_time, life_time=None)
            else:
                if (order_id not in self.open_orders
                    and self.stop_loss_taker_threshold != 0
                    and self.bid_price / self.open_price - 1 > self.stop_loss_taker_threshold):
                    new_order_id = self.market_order(side=OrderSide.SELL, volume=amount, reduce_only=False)
                else:
                    new_order_id = self.limit_order(side=OrderSide.SELL, volume=amount, price=self.bid_price+self.instrument.price_increment, 
                                                     post_only=True, reduce_only=False, expire_time=expire_time, life_time=None)
                # new_order_id = self.limit_order(side=side, volume=amount, 
                #                                 price=self.bid_price+self.instrument.price_increment,
                #                                 post_only=True, reduce_only=False, expire_time=expire_time, life_time=None)
            
            self.not_stop_win_order.add(new_order_id)
            
            if order_id in self.open_orders:
                self.open_orders.add(new_order_id)
                self.record_order_type(UTC_time, new_order_id, self.order_strategy_id[order_id], "signal_open")
                self.order_strategy_id[new_order_id] = self.order_strategy_id[order_id]
                del self.order_strategy_id[order_id]
                # logging.debug(f"信号追单完成")
            else:
                self.record_order_type(UTC_time, new_order_id, self.order_strategy_id[order_id], "stop_loss")
                self.order_strategy_id[new_order_id] = self.order_strategy_id[order_id]
                del self.order_strategy_id[order_id]
                # logging.debug(f"止损追单完成")

            
        self.del_cache_orderid(order_id)
          
        return
        
    
    def on_order_filled(self, event: OrderFilled):
        t1 = time.time()
        # 响应事件“订单完成”
        logging.debug(f"订单成交:{event}")
        # self.log.warning(f"订单成交:{event}")
        # for record & plot
        # self.stats_record.order_list.append(from_order_event(event))
        self.stats_record.order_list.update(event)
        
        filled_amount = float(event.last_qty)
        filled_price = float(event.last_px)
        order_id = event.client_order_id
        # fee = event.commission
        update_timestamp = event.ts_event / 1000 / 1000 # ts_event是纳秒,转换为毫秒

        # self.log.warning(f'[filled order{order_id}]: time: {update_timestamp}; amount: {filled_amount}; price: {filled_price}')

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

        last_pos = self.pos

        self.pos = amount_sum
        
        if order_id in self.open_orders:
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
                    elif event.order_side == OrderSide.BUY:
                        self.cancel_order(_order_list[0])
                self.del_cache_orderid(event.client_order_id)
                return
            
            # 挂止盈单
            if event.order_side == OrderSide.SELL:
                if self.quote_token == "BNB":
                    price = filled_price - self.instrument.price_increment
                else:
                    price = filled_price *(1 - self.stop_win_rate)
                win_order_id = self.limit_order(side=OrderSide.BUY, volume=filled_amount, price=price, 
                                             post_only=True, reduce_only=False, expire_time=None, life_time=100 * 10 * 60)
                self.record_order_type(update_timestamp, win_order_id, win_order_id, "stop_win")
                logging.debug(f"止盈挂单完成: BUY")

            else:
                if self.quote_token == "BNB":
                    price = filled_price + self.instrument.price_increment
                else:
                    price = filled_price *(1 + self.stop_win_rate)
                win_order_id = self.limit_order(side=OrderSide.SELL, volume=filled_amount, price=price, 
                                             post_only=True, reduce_only=False, expire_time=None, life_time=100 * 10 * 60)
                self.record_order_type(update_timestamp, win_order_id, win_order_id, "stop_win")
                logging.debug(f"止盈挂单完成: SELL")

        self.on_order_filled_time += time.time() - t1
        if self.order_amount_dict[order_id] - filled_amount < 1e-10:
            self.del_cache_orderid(order_id) 
        else:
            self.order_amount_dict[order_id] -= filled_amount
            # logging.debug(f"已成交：{filled_amount}，剩余：{self.order_amount_dict[order_id]}")
            
        

    def on_order_canceled(self, event: OrderCanceled):
        # 响应事件“订单已取消”
        # for record & plot
        # self.stats_record.order_list.append(from_order_event(event))
        self.stats_record.order_list.update(event)
        
        logging.debug(f"订单取消:{event}")
        self.del_cache_orderid(event.client_order_id)

        return 
    
    def on_order_expired(self, event: OrderExpired):
        # 响应事件“订单超时”
        # for record & plot
        # self.stats_record.order_list.append(from_order_event(event))
        self.stats_record.order_list.update(event)
        
        logging.debug(f"订单超时:{event}")
        self.del_cache_orderid(event.client_order_id)

        return 

    def on_order_submitted(self, event: OrderSubmitted):
        # for record & plot
        # self.stats_record.order_list.append(from_order_event(event))
        self.stats_record.order_list.update(event)
        
        return 

    # -------------------------------order function--------------------------------- #

    # TODO：优化写法

    def limit_order(self, side, volume, price, post_only=True, reduce_only=False, expire_time=None, life_time=None):
        t1 = time.time()
        # trade_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL

        if expire_time is None:
            expire_time = self.clock.utc_now() + pd.Timedelta(milliseconds=life_time)

        order = self.order_factory.limit(
            instrument_id=self.instrument_id,
            order_side=side,
            price=Price(price, precision=self.instrument.price_precision),
            quantity=self.instrument.make_qty(volume),
            post_only=post_only,
            reduce_only=reduce_only,
            time_in_force=TimeInForce.GTD,
            expire_time=expire_time,
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
        self.order_expire_time[_order_id] = expire_time
        self.order_side_dict[_order_id] = side
        self.order_amount_dict[_order_id] = volume

        self.limit_order_time += time.time() - t1

        return _order_id

    def market_order(self, side, volume, reduce_only=False):
        # trade_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
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
        # self.order_list[_order_id] = order
        # self.order_expire_time[_order_id] = expire_time
        # self.order_side_dict[_order_id] = side
        self.order_amount_dict[_order_id] = volume

        return _order_id
        

    # -------------------------------feature function--------------------------------- #

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


