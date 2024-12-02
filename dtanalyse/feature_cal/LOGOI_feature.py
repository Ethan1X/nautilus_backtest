import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_generator import *
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
import numpy as np

class LOGOIGenerator(FeatureGenerator):
    '''
    特征名称：'logoi'，对数版订单不平衡（LOG Order Imbalance）
    特征编号：24
    所需数据：ticker数据
    Author: Qiushi Bu
    Reviewer: Yu Liu
    '''

    def __init__(self, config: FeatureConfig) -> None:
        '''
        特征名称：'logoi'
        config: 不需要输入额外参数
        '''
        super().__init__(config)

        self.fe_name = 'logoi' 
        self.previous_bid_log_volume = None  
        self.previous_ask_log_volume = None
        self.previous_bid_price = None  
        self.previous_ask_price = None
        self.current_bid_log_volume = None
        self.current_ask_log_volume = None
        self.current_bid_price = None
        self.current_ask_price = None

    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None):
            pass

    def process(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
        '''
        首先判断来的数据是不是ticker数据，如果是，进行信息更新，如果不是，不进行处理，返回None
        '''
        if ticker is None:  #如果没有传来ticker数据，则不更新
            return [(self.fe_name, None, None)]
        self.update(ticker)
        feature_ret = self.calculate()

        return feature_ret
    
    def update(self, ticker: QuoteTick):
        '''
        更新状态：将上一时刻的ticker信息存到previous_，将当前时刻的ticker信息存入current_
        '''
        self.previous_bid_log_volume = self.current_bid_log_volume
        self.previous_ask_log_volume = self.current_ask_log_volume
        self.previous_bid_price = self.current_bid_price
        self.previous_ask_price = self.current_ask_price
        self.current_bid_log_volume = np.log10(ticker.bid_size)
        self.current_ask_log_volume = np.log10(ticker.ask_size)
        self.current_bid_price = ticker.bid_price
        self.current_ask_price = ticker.ask_price
        self.ts_event = ticker.ts_event
        
    def calculate(self):
        '''
        计算LOGOI：首先判断是否有前一时刻的ticker信息，如果没有返回none，如果有，按照公式计算logoi值
        '''
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
            self.ts_event = None
            logoi = None
        
        return [(self.fe_name, self.ts_event, logoi)]


if __name__ == '__main__':     
    initlog(None, 'logoi_feature.log', logging.INFO)
    logoi_config = FeatureConfig()
    print()

    begin_time = datetime.datetime(2024, 3, 27, 21,tzinfo=TZ_8) # 要加一个tzinfo
    end_time = datetime.datetime(2024, 3, 27, 21, 59,tzinfo=TZ_8)
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

    print(fe_list[:100])
  