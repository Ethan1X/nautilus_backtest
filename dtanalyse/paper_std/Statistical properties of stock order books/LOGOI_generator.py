import os, sys, logging
sys.path.append(os.getcwd().split('paper_std')[0])
if '../' not in sys.path:
    sys.path.append('../')

from util.load_s3_data import LoadS3Data
from util.time_method import *
from data_generator import *
from typing import TypedDict
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
import numpy as np

class LOGOIGenerator(FeatureGenerator):
    '''
    利用ticker数据，生成对数版订单不平衡（LOG Order Imbalance）,不需要输入额外参数
    '''

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)

        self.fe_name = 'logoi' 
        self.previous_bid_log_volume = None  
        self.previous_ask_log_volume = None
        self.previous_bid_price = None  
        self.previous_ask_price = None
    def process(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
        
        if ticker == None:  #如果没有传来ticker数据，则不更新
            return 
        self.update(ticker)
        feature_ret = self.calculate(ticker)

        return feature_ret
    
    def update(self, ticker: QuoteTick):
        self.current_bid_log_volume = np.log10(ticker.bid_size)
        self.current_ask_log_volume = np.log10(ticker.ask_size)
        self.current_bid_price = ticker.bid_price
        self.current_ask_price = ticker.ask_price
        self.ts_event = ticker.ts_event
        
    def calculate(self, ticker: QuoteTick):

        if self.previous_bid_log_volume is not None:
            if self.current_bid_price == self.previous_bid_price:
                delta_bid = self.current_bid_log_volume - self.previous_bid_log_volume
            elif self.current_bid_price > self.previous_bid_price:
                delta_bid = self.current_bid_log_volume
            elif self.current_bid_price < self.previous_bid_price:
                delta_bid = - self.previous_bid_log_volume
            
            if self.current_ask_price == self.previous_ask_price:
                delta_ask = self.current_ask_log_volume - self.previous_ask_log_volume
            elif self.current_ask_price < self.previous_ask_price:
                delta_ask = self.current_ask_log_volume
            elif self.current_ask_price > self.previous_ask_price:
                delta_ask = - self.previous_ask_log_volume
            
            logoi = delta_bid - delta_ask
        else:
            logoi = None

        self.previous_bid_log_volume = self.current_bid_log_volume
        self.previous_ask_log_volume = self.current_ask_log_volume
        self.previous_bid_price = self.current_bid_price
        self.previous_ask_price = self.current_ask_price
        
        return [(self.fe_name, self.ts_event, logoi)]


if __name__ == '__main__':     
    initlog(None, 'logoi_feature.log', logging.INFO)
    logoi_config = FeatureConfig()

    begin_time = datetime.datetime(2024, 3, 27, 21,tzinfo=TZ_8) # 要加一个tzinfo
    end_time = datetime.datetime(2024, 3, 27, 21, 1,tzinfo=TZ_8)
    exchange = 'binance'
    symbol = 'btc_usdt'

    ins = LOGOIGenerator(logoi_config)
    fe_list = []

    data_generator = get_data_generator(begin_time, end_time, exchange, symbol)
    for idx, row in enumerate(data_generator):
        #print(idx)
        if row[1] == 'ticker':
            fe_list.append(ins.process(ticker=row[0]))
        elif row[1] == 'depth':
            fe_list.append(ins.process(depth=row[0]))
        else:
            fe_list.append(ins.process(trade=row[0]))

        if idx > 10:
            break
    print(fe_list)
  