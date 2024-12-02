import os, sys, logging, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from util.load_s3_data import LoadS3Data
from util.time_method import *
from data_generator import *
from typing import TypedDict
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
from tqdm import tqdm
import numpy as np
from util.statistic_method_v2 import add_data, DataInfo


class Price_Change_Rate_Label(FeatureGenerator):

    # Author: Jungle

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        
        self.ticker_list = []
        self.time_period = config['time_period'] # 0.5, 1, 3
        self.min_period = int(100 * 1e6)      # 100毫秒对应的纳秒数
        self.size = [self.get_window_size(self.time_period[i]*1e9) for i in range(len(self.time_period))]
        self.window_len = int(max(self.size))
        self.buffer_size = self.window_len + 2

        self.active_ticker = [None] * self.buffer_size
        self.data_index = 0
        self.ticker_sum = 0
        
        self.data_source = "backtest"
        if "data_source" in config.keys():
            self.data_source = config['data_source']
            
        self.fe_name = []
        for time in self.time_period:
            self.fe_name.append(f'price_change_rate_{time}s')

    def get_window_size(self, period):
        return int(period) // self.min_period
        
    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None):
            pass

    def get_data_index(self, shift):
        return (self.data_index + shift) % self.buffer_size

    def get_ticker_data(self, shift):
        _data_index = self.get_data_index(shift)
        _data = self.active_ticker[_data_index]
        return _data

    def process(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):

        if ticker is not None:
            self.update(ticker, trade, depth)
            label_ret = self.calculate(ticker)
            return label_ret

        return None
            
    def update(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
        _ticker = ticker
        
        if self.data_source == "S3":
            _ticker = QuoteTick(ticker['ap'], ticker['aa'], ticker['bp'], ticker['ba'], ticker['tp'])
        
        self.update_ticker(_ticker)
        # print(f'change rate ticker: {len(self.ticker_list)} {_ticker}')
        return

    def update_ticker(self, ticker: QuoteTick=None):
        if ticker is None:
            return
        
        self.ticker_list.append(ticker)

        # 如果活跃列表为空，把后备列表中的头条ticker放入活跃列表
        if self.ticker_sum == 0:
            _ticker = self.ticker_list.pop(0)
            self.active_ticker[0] = _ticker
            self.data_index = 0
            self.ticker_sum = 1

        # print(f'{self.size} {self.window_len}')
        # time.sleep(10)

        # 如活跃列表数据未满，则从后备列表中取出所有处于活跃窗口内的ticker放入活跃列表
        while self.ticker_sum <= self.window_len and len(self.ticker_list) > 0:
            _next_ticker = self.ticker_list[0]
            _shift = int(_next_ticker.ts_event - self.active_ticker[self.data_index].ts_event) // self.min_period
            _next_shift = self.get_window_size(_next_ticker.ts_event)
            _cur_shift = self.get_window_size(self.active_ticker[self.data_index].ts_event)
            _shift = _next_shift - _cur_shift
            # print(f'new ticker: {_next_shift} {_cur_shift} {_shift}')
            if _shift < self.buffer_size:
                _data_index = self.get_data_index(_shift)
                # print(f'add ticker: {self.ticker_sum} {_shift} {self.data_index} {_data_index}')
                # time.sleep(1)
                self.active_ticker[_data_index] = _next_ticker
                self.ticker_sum = _shift
                self.ticker_list.pop(0)
            else:
                # 后备队列的数据已经超出最大活跃窗口的范围，丢弃活跃列表里最早的ticker（此ticker已无法得到足够的数据计算标注）
                # 找到下一条非None数据作为新的活跃列表起始ticker
                # 注意：feature manager负责在有数据缺失的情况下适当填充，允许填充的窗口宽度在上层定义（默认为30秒）
                self.remove_active_ticker()
                break

    def remove_active_ticker(self):
            self.active_ticker[self.data_index] = None
            while self.ticker_sum > 0 and self.active_ticker[self.data_index] is None:
                self.data_index = self.get_data_index(1)
                self.ticker_sum -= 1
        
    def calculate(self, ticker):
        if self.ticker_sum <= self.window_len: # 数不满，直接读下一条
            return None

        # self.first_time = False
        # print('debug: ', ticker.ts_event/1e9, self.ticker_list[0].ts_event/1e9, len(self.ticker_list))

        return_list = []
        _cur_ticker = self.active_ticker[self.data_index]
        _cur_mid_price = (_cur_ticker.ask_price + _cur_ticker.bid_price) / 2
        for i in range(len(self.time_period)):
            _size = self.size[i]
            _fore_ticker = self.get_ticker_data(_size)
            if _size <= self.ticker_sum and _fore_ticker is not None:
                _fore_mid_price = (_fore_ticker.ask_price + _fore_ticker.bid_price) / 2
                price_change_rate = _fore_mid_price / _cur_mid_price - 1
                _ts = (_cur_ticker.ts_event // self.min_period) * self.min_period
                return_list.append((self.fe_name[i], int(_ts), price_change_rate))

                # print(f'{self.fe_name[i]}, {_cur_ticker.ts_event}, {price_change_rate}')
                # if _cur_ticker.ts_event > 1709126890000 * 1e6 and _cur_ticker.ts_event < 1709126896000 * 1e6:
                #     print(f'calc({_cur_ticker.ts_event}/{self.data_index}vs{_size}): {price_change_rate} = {_fore_mid_price} / {_cur_mid_price} - 1; {i}/{self.ticker_sum} {_cur_ticker} {_fore_ticker}')
                # if _cur_ticker.ts_event == 1709126890897999872:
                #     for i, t in enumerate(self.active_ticker):
                #         print(f'active tickers: {i} {t}')

        self.remove_active_ticker()
        return return_list


class Price_Change_Trend_Label(FeatureGenerator):

    # Author: Jungle

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        
        self.ticker_list = []
        self.time_period = config['time_period'] # 0.5, 1, 3
        self.threshold = config['threshold']
        self.min_period = int(100 * 1e6)      # 100毫秒对应的纳秒数
        
        self.size = [self.get_window_size(self.time_period[i]*1e9) for i in range(len(self.time_period))]
        self.window_len = int(max(self.size))
        self.buffer_size = self.window_len + 2

        self.active_ticker = [None] * self.buffer_size
        self.data_index = 0
        self.ticker_sum = 0
        
        self.data_source = "backtest"
        if "data_source" in config.keys():
            self.data_source = config['data_source']
            
        self.fe_name = []
        for time in self.time_period:
            self.fe_name.append(f'price_change_trend_{time}s')

    def get_window_size(self, period):
        return int(period) // self.min_period
        
    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None):
            pass

    def get_data_index(self, shift):
        return (self.data_index + shift) % self.buffer_size

    def get_ticker_data(self, shift):
        _data_index = self.get_data_index(shift)
        _data = self.active_ticker[_data_index]
        return _data

    def process(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):

        if ticker is not None:
            self.update(ticker, trade, depth)
            label_ret = self.calculate(ticker)
            return label_ret

        return None
            
    def update(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
        _ticker = ticker
        
        if self.data_source == "S3":
            _ticker = QuoteTick(ticker['ap'], ticker['aa'], ticker['bp'], ticker['ba'], ticker['tp'])
        
        self.update_ticker(_ticker)
        # print(f'change rate ticker: {len(self.ticker_list)} {_ticker}')
        return

    def update_ticker(self, ticker: QuoteTick=None):
        if ticker is None:
            return
        
        self.ticker_list.append(ticker)

        # 如果活跃列表为空，把后备列表中的头条ticker放入活跃列表
        if self.ticker_sum == 0:
            _ticker = self.ticker_list.pop(0)
            self.active_ticker[0] = _ticker
            self.data_index = 0
            self.ticker_sum = 1

        # print(f'{self.size} {self.window_len}')
        # time.sleep(10)

        # 如活跃列表数据未满，则从后备列表中取出所有处于活跃窗口内的ticker放入活跃列表
        while self.ticker_sum <= self.window_len and len(self.ticker_list) > 0:
            _next_ticker = self.ticker_list[0]
            # _shift = int(_next_ticker.ts_event - self.active_ticker[self.data_index].ts_event) // self.min_period
            _next_shift = self.get_window_size(_next_ticker.ts_event)
            _cur_shift = self.get_window_size(self.active_ticker[self.data_index].ts_event)
            _shift = _next_shift - _cur_shift
            # print(f'new ticker: {_next_shift} {_cur_shift} {_shift_delta}')
            if _shift < self.buffer_size:
                _data_index = self.get_data_index(_shift)
                # print(f'add ticker: {self.ticker_sum} {_shift_delta} {self.data_index} {_data_index}')
                # time.sleep(1)
                self.active_ticker[_data_index] = _next_ticker
                self.ticker_sum = _shift
                self.ticker_list.pop(0)
            else:
                # 后备队列的数据已经超出最大活跃窗口的范围，丢弃活跃列表里最早的ticker（此ticker已无法得到足够的数据计算标注）
                # 找到下一条非None数据作为新的活跃列表起始ticker
                # 注意：feature manager负责在有数据缺失的情况下适当填充，允许填充的窗口宽度在上层定义（默认为30秒）
                self.remove_active_ticker()
                break

    def remove_active_ticker(self):
            self.active_ticker[self.data_index] = None
            while self.ticker_sum > 0 and self.active_ticker[self.data_index] is None:
                self.data_index = self.get_data_index(1)
                self.ticker_sum -= 1
        
    def calculate(self, ticker):
        if self.ticker_sum <= self.window_len: # 数不满，直接读下一条
            return None

        # self.first_time = False
        # print('debug: ', ticker.ts_event/1e9, self.ticker_list[0].ts_event/1e9, len(self.ticker_list))

        return_list = []
        _cur_ticker = self.active_ticker[self.data_index]
        # _cur_mid_price = (_cur_ticker.ask_price + _cur_ticker.bid_price) / 2
        for i in range(len(self.time_period)):
            _size = self.size[i]
            _fore_ticker = self.get_ticker_data(_size)
            if _size <= self.ticker_sum and _fore_ticker is not None:
                #_fore_mid_price = (_fore_ticker.ask_price + _fore_ticker.bid_price) / 2
                ask_price_change_rate = _fore_ticker.ask_price / _cur_ticker.ask_price - 1
                bid_price_change_rate = _fore_ticker.bid_price / _cur_ticker.bid_price - 1
                _up_label = 1 if ask_price_change_rate > self.threshold else 0
                _down_label = 1 if bid_price_change_rate < -self.threshold else 0
                _ts = (_cur_ticker.ts_event // self.min_period) * self.min_period
                return_list.append((f'{self.fe_name[i]}_up', _ts, _up_label))
                return_list.append((f'{self.fe_name[i]}_down', _ts, _down_label))
                # if _cur_ticker.ts_event > 1709126890000 * 1e6 and _cur_ticker.ts_event < 1709126896000 * 1e6:
                #     print(f'calc({_cur_ticker.ts_event}/{self.data_index}vs{_size}): {price_change_rate} = {_fore_mid_price} / {_cur_mid_price} - 1; {i}/{self.ticker_sum} {_cur_ticker} {_fore_ticker}')
                # if _cur_ticker.ts_event == 1709126890897999872:
                #     for i, t in enumerate(self.active_ticker):
                #         print(f'active tickers: {i} {t}')

        self.remove_active_ticker()
        return return_list

