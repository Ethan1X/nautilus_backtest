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
import time

class BreadthGenerator(FeatureGenerator):
     '''
     注意 时间点相同的trade只算一次
     '''
    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        self.left = config['left']
        self.right = config['right']
        self.ticker_count = np.zeros(self.right+1)# 记录right次trade中间有多少次 ticker，初始都是0
        self.trade_volume = np.zeros(self.right+1)# 记录right个时刻trade交易量之和，初始为0
        self.trade_num = -1 # 记录已经存储的交易次数
        self.last_trade_time = 0 
        self.ts_event = 0
        self.fe_name1 = f'Breadth_{self.left}_{self.right}'  
        self.fe_name2 = f'Immediacy_{self.left}_{self.right}'
        self.fe_name3 = f'VolumeAvg_{self.left}_{self.right}'
        
    def process(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):

        self.update(ticker, trade, depth)
        feature_ret = self.calculate()

        return feature_ret
    
    def update(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
        if trade != None: 
            self.ts_event = trade.ts_event
            if self.ts_event != self.last_trade_time:  # 判断交易时间是否相同，如果是同一个时刻发生的交易则不更新trade_num
                self.trade_num += 1 
                self.last_trade_time = self.ts_event
            if self.trade_num > self.right: # 如果大于endtime，将第一个元素删除，并在最后补一个0
                self.ticker_count = np.append(self.ticker_count[1:], 0)
                self.trade_volume = np.append(self.trade_volume[1:], 0)
                self.trade_num -= 1
            self.trade_volume[self.trade_num] += trade.size
        elif ticker != None: # ticker数据只更新状态，不返回信息
            self.ts_event = ticker.ts_event
            if self.trade_num >= 0:
                self.ticker_count[self.trade_num] += 1 
            return 
        elif depth != None: # 不处理depth数据
            self.ts_event = depth.ts_event
            return 
        return 
    
    def calculate(self):   
        if self.trade_num >= self.right: # 如果到right，计算并返回breadth值
            breadth = sum(self.ticker_count[0:(self.right-self.left)])
            immediacy = (self.right-self.left) / breadth
            volumeavg = sum(self.trade_volume[0:(self.right-self.left)]) / breadth
        else:  # 如果没有到right，返回None
            breadth, immediacy, volumeavg = None, None, None
        return [(self.fe_name1, self.ts_event, breadth),
                (self.fe_name2, self.ts_event, immediacy),
                (self.fe_name3, self.ts_event, volumeavg)]


if __name__ == '__main__':     
    initlog(None, 'breadth_feature.log', logging.INFO)
    breadth_config = FeatureConfig(left = 2, right = 4)

    begin_time = datetime.datetime(2024, 3, 27, 21,0,tzinfo=TZ_8) # 要加一个tzinfo
    right = datetime.datetime(2024, 3, 27, 21, 59,tzinfo=TZ_8)
    exchange = 'binance'
    symbol = 'btc_usdt'

    ins = BreadthGenerator(breadth_config)
    fe_list = []

    data_generator = get_data_generator(begin_time, right, exchange, symbol)

 
    start = time.perf_counter() 
          
    
    for idx, row in enumerate(data_generator):
        #print(idx)
        if row[1] == 'ticker':
            fe_list.append(ins.process(ticker=row[0]))
        elif row[1] == 'depth':
            fe_list.append(ins.process(depth=row[0]))
        else:
            fe_list.append(ins.process(trade=row[0]))
#         if idx > 100:
#             break
     print(fe_list[:100])
    end = time.perf_counter()   
    duration = end - start 
    print("程序运行时间：", duration)