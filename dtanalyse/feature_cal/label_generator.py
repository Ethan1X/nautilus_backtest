import os, sys, logging

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

    # Author: Zhihan Cai

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        
        self.ticker_list = []
        self.time_period = config['time_period'] # 0.5, 1, 3

        self.data_source = "backtest"
        if "data_source" in config.keys():
            self.data_source = config['data_source']
            
        self.fe_name = []
        for time in self.time_period:
            self.fe_name.append(f'price_change_rate_{time}s')
        self.right_index = [0] * len(self.time_period)
        self.first_time = True

    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None):
            pass

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
        
        self.ticker_list.append(_ticker)
        # print(f'change rate ticker: {len(self.ticker_list)} {_ticker}')
        return
        
    def calculate(self, ticker):
        if ticker.ts_event - self.ticker_list[0].ts_event < self.time_period[-1]*1e9: # 数不满，直接读下一条
            return None

        # self.first_time = False
        # print('debug: ', ticker.ts_event/1e9, self.ticker_list[0].ts_event/1e9, len(self.ticker_list))

        return_list = []
        last_mid_price = (self.ticker_list[0].ask_price +  self.ticker_list[0].bid_price) / 2
        for i in range(len(self.time_period)):
            while self.right_index[i] < len(self.ticker_list) - 1 and \
                self.ticker_list[self.right_index[i] + 1].ts_event - self.ticker_list[0].ts_event < self.time_period[i]*1e9:
                self.right_index[i] += 1
            current_mid_price = (self.ticker_list[self.right_index[i]].ask_price + self.ticker_list[self.right_index[i]].bid_price) / 2
            price_change_rate = current_mid_price / last_mid_price - 1
            if self.ticker_list[0].ts_event > 1709126890000 * 1e6 and self.ticker_list[0].ts_event < 1709126896000 * 1e6:
                print(f'calc({self.ticker_list[0].ts_event}): {price_change_rate} = {current_mid_price} / {last_mid_price} - 1; {i}/{len(self.ticker_list)}, {self.right_index[i]}, {self.ticker_list[self.right_index[i]]} / {self.ticker_list[0]}')
            return_list.append((self.fe_name[i], self.ticker_list[0].ts_event, price_change_rate))

        # print(f'label(cr): {ticker.ts_event} {len(return_list)} {len(self.ticker_list)}{current_mid_price}')
        self.ticker_list.pop(0)
        for i in range(len(self.time_period)):
            self.right_index[i] -= 1
        return return_list


class Price_Change_Trend_Label(FeatureGenerator):

    # Author: Jungle

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        
        self.ticker_list = []
        self.time_period = config['time_period'] # 0.5, 1, 3
        self.threshold = config['threshold']

        self.data_source = "backtest"
        if "data_source" in config.keys():
            self.data_source = config['data_source']
            
        self.fe_name = []
        for time in self.time_period:
            self.fe_name.append(f'price_change_Trend_{time}s')
        _periods = len(self.time_period)
        # self.right_index = [0] * _periods
        self.bestask_list = [[]] * _periods
        # self.bestask_max = [[]] * _periods
        self.bestbid_list = [[]] * _periods
        # self.bestbid_min = [[]] * _periods

        self.first_time = True
        
    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None):
        pass

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
        
        self.ticker_list.append(_ticker)
        for i in range(len(self.time_period)):
            _data_ask = DataInfo(_ticker.ask_price, _ticker.ts_event)
            add_data(_data_ask, self.bestask_list[i], -1)
            _data_bid = DataInfo(_ticker.bid_price, _ticker.ts_event)
            add_data(_data_bid, self.bestbid_list[i], 1)
            if _ticker.ts_event - self.ticker_list[0].ts_event > self.time_period[i] * 1e9:
                # print(f'update: {i}/{self.time_period[i]} {len(self.bestask_max)}')
                # todo：从bestask_max中删除对应时间点的数据，如果还没被去除的话
                if self.bestask_list[i][0].ts - self.ticker_list[0].ts_event > self.time_period[i] * 1e9:
                    self.bestask_list[i].pop(0)
                # self.bestask_max[i].append(self.bestask_list[i][0])
                # todo：从bestbid_min中删除对应时间点的数据，如果还没被去除的话
                if self.bestbid_list[i][0].ts - self.ticker_list[0].ts_event > self.time_period[i] * 1e9:
                    self.bestbid_list[i].pop(0)
                # self.bestbid_min[i].append(self.bestbid_list[i][0])

        # if len(self.bestask_max[0]) > 0:
        #     print(f'period {self.time_period[0]}', self.bestask_max[0][0], self.bestask_list[0][0])
        # if len(self.bestbid_min[0]) > 0:
        #     print(f'period {self.time_period[0]}', self.bestbid_min[0][0], self.bestbid_list[0][0])
        return
        
    def calculate(self, ticker):
        # print(f'{(ticker.ts_event-self.ticker_list[0].ts_event)/1e9} {self.time_period[-1]}, {len(self.ticker_list)}')
        if self.first_time and ticker.ts_event - self.ticker_list[0].ts_event < self.time_period[-1] * 1e9: # 数不满，直接读下一条
            return None

        self.first_time = False
        # print('debug: ', ticker.ts_event/(10**9), self.ticker_list[0].ts_event/(10**9), len(self.ticker_list), len(self.bestask_max), len(self.bestbid_min))

        return_list = []
        for i in range(len(self.time_period)):
            # while self.right_index[i] < len(self.ticker_list) - 1 and \
            #     self.ticker_list[self.right_index[i] + 1].ts_event - self.ticker_list[0].ts_event < self.time_period[i] * 1e:
            #     self.right_index[i] += 1
            _ask_price = self.ticker_list[0].ask_price
            # _ask_price_change_rate = (self.bestask_max[i][-1].data - _ask_price) / _ask_price
            _ask_price_change_rate = (self.bestask_list[i][0].data - _ask_price) / _ask_price
            _up_label = 1 if _ask_price_change_rate > self.threshold else 0
            _bid_price = self.ticker_list[0].bid_price
            # _bid_price_change_rate = (self.bestbid_min[i][-1].data - _bid_price) / _bid_price
            _bid_price_change_rate = (self.bestbid_list[i][0].data - _bid_price) / _bid_price
            _down_label = 1 if _bid_price_change_rate < -self.threshold else 0
            return_list.append((f'{self.fe_name[i]}_up', self.ticker_list[0].ts_event, _up_label))
            return_list.append((f'{self.fe_name[i]}_down', self.ticker_list[0].ts_event, _down_label))

        self.ticker_list.pop(0)
        # for i in range(len(self.time_period)):
        #     # self.right_index[i] -= 1
        #     self.bestask_max[i].pop(0)
        #     self.bestbid_min[i].pop(0)
        return return_list

    
if __name__ == '__main__':
    begin_time = datetime.datetime(2024, 3, 1, 0, tzinfo=TZ_8)
    end_time = datetime.datetime(2024, 3, 1, 0, 20, tzinfo=TZ_8)
    exchange = 'binance'
    symbol = 'btc_usdt'
    
    # time_period = [0.5, 1, 3]
    # ds = "not S3"
    # config = FeatureConfig(time_period=time_period, data_source=ds)

    time_period_c = [0.5, 1, 3]
    ds_c = "nautilus"
    PRICE_FLUCTUATION = 1e-4
    config_c = FeatureConfig(time_period=time_period_c, data_source=ds_c, threshold=PRICE_FLUCTUATION)

    # ins = Price_Change_Rate_Label(config)
    # fe_list = []
    ins_c = Price_Change_Trend_Label(config_c)
    fe_list_c = []

    data_generator = get_data_generator(begin_time, end_time, exchange, symbol)

    start_time=time.time()
    for idx, row in enumerate(get_data_generator(begin_time, end_time, exchange, symbol)):
        if row[1] == 'ticker':
            # fe_list.append(ins.process(ticker=row[0]))
            fe_list_c.append(ins_c.process(ticker=row[0]))
        elif row[1] == 'depth':
            # fe_list.append(ins.process(depth=row[0]))
            fe_list_c.append(ins_c.process(depth=row[0]))
        else:
            # fe_list.append(ins.process(trade=row[0]))
            fe_list_c.append(ins_c.process(trade=row[0]))
        # if idx > 1000:
        #     break
    # print(fe_list[:100])
    
    end_time=time.time()
    print(end_time-start_time)

    # fe_list = list(filter(lambda x: x is not None, fe_list))
    fe_list_c = list(filter(lambda x: x is not None, fe_list_c))
    
    import joblib
    # joblib.dump(fe_list,"label_list.pkl")
    joblib.dump(fe_list_c,"label_list_c.pkl")


