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


class OIRGenerator(FeatureGenerator):
    '''
    利用ticker数据，生成订单不平衡率（Order Imbalance Ratio）,不需要输入额外参数
    '''

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)

        self.fe_name = 'oir'  # The feature name for bid price difference
        
    def process(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
        
        if ticker == None:  #如果没有传来ticker数据，则不更新
            return 
        
        self.update(ticker)
        feature_ret = self.calculate()

        return feature_ret
    
    def update(self, ticker: QuoteTick):  
        self.current_bid_volume = ticker.bid_size
        self.current_ask_volume = ticker.ask_size
        self.ts_event = ticker.ts_event
        
    def calculate(self):

        oir = (self.current_bid_volume - self.current_ask_volume)/(self.current_bid_volume + self.current_ask_volume)
        
        return [(self.fe_name, self.ts_event, oir)]


if __name__ == '__main__':     
    initlog(None, 'oir_feature.log', logging.INFO)
    oir_config = FeatureConfig()
    print()

    begin_time = datetime.datetime(2024, 3, 27, 21,0,tzinfo=TZ_8) # 要加一个tzinfo
    end_time = datetime.datetime(2024, 3, 27, 21, 1,tzinfo=TZ_8)
    exchange = 'binance'
    symbol = 'btc_usdt'

    ins = OIRGenerator(oir_config)
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
  