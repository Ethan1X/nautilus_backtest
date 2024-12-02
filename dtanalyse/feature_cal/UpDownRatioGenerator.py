import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# from util.load_s3_data import LoadS3Data
from util.time_method import *
from data_generator import *
from typing import TypedDict
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
from util.statistic_method_v2 import add_data, DataInfo

class UpDownRatioGenerator(FeatureGenerator):
    '''
    Author: Zeng Yunjia
    Feature: to be added
    '''
    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        # FeatureConfig中包括left和freq两组参数，对于每个trade，记录其价格相对freqs前价格的变化率；
        # 每次on trade需要计算自己对freqs前的价格变化率，以及更新从left到当前每个freqs前的价格变化率（up）的平方和，价格变化率（down）的平方和
        
        self.data_mode = "normal"
        if "data_mode" in config.keys():
            self.data_mode = config['data_mode']

        self.left = config['left']
        self.freq = config['freq']
        print("config:",self.left,self.freq)
        self.fe_name1 = f'UpDownRatio'
        self.latest_trade_ts = None
        self.last_trade_ts = None
        '''self.trade_volumes_buy = []
        self.trade_price_buy = []
        self.trade_amounts_buy = []
        self.trade_volumes_sell = []
        self.trade_price_sell = []
        self.trade_amounts_sell = []'''
        self.trade_volumes = []
        self.trade_prices = []
        self.trade_amounts = []
        self.trade_times = []
        self.change = [[] for i in range(len(self.freq))]
        self.signal = False

        _periods = len(self.left)
        self.periods = len(self.left)
        self.index_freq = [[0 for i in range(len(self.freq))] for i in range(len(self.left))]  ##记录freq之前的trade的位置用来算changerate
        self.index_end = [0 for i in range(len(self.left))]  ##入队&加和的时候用到
        self.index_start = [0 for i in range(len(self.left))]  ##出队&减去的时候用到
        self.squaresum_up = [[0.0 for i in range(len(self.freq))] for i in range(len(self.left))]
        self.squaresum_dw = [[0.0 for i in range(len(self.freq))] for i in range(len(self.left))]
        
        
    def process(self, ticker: QuoteTick = None, trade: TradeTick = None, depth: OrderBook = None):
        '''
            feature计算主流程 供外部调用
        '''
        if self.data_mode == "central":
            self.update_signal(trade)
        else:
            self.update(ticker, trade, depth)
        feature_ret = self.calculate()

        return feature_ret

    def update_signal(self, trade: TradeTick=None):
        if trade is not None:
            self.latest_trade_ts = trade.ts_event
            self.signal = True
        else:
            self.latest_trade_ts = None
            self.signal = False

    def update(self, ticker: QuoteTick = None, trade: TradeTick = None, depth: OrderBook = None):
        '''
        更新自定义特征的内部状态
        '''
        # print(ticker,trade,depth)
        # 检查最新的交易数据时间戳是否更新，如果更新则清空交易量列表
        if trade is not None:
            self.signal = True
            if self.last_trade_ts is None or (trade.ts_event//1000 != self.last_trade_ts//1000):
                self.last_trade_ts = trade.ts_event
                # 在这里实现自定义特征的更新逻辑
                # 将trade数据插入队列 更新volume_sum,amount_sum列表
                self.trade_times.append(trade.ts_event)
                self.trade_volumes.append(trade.size*trade.price)
                self.trade_amounts.append(trade.size)
                self.trade_prices.append(trade.price)
                for i in range(len(self.freq)):
                    self.change[i].append(0.0)
                
            else:
                # 如果是相同时刻的trade，直接更新当前时刻
                self.trade_volumes[-1] += trade.size*trade.price
                self.trade_amounts[-1] += trade.size
                self.trade_prices[-1] = self.trade_volumes[-1]/self.trade_amounts[-1]
            self.latest_trade_ts = trade.ts_event
            
        elif (ticker is not None and (ticker.ts_event//1000**2 != self.last_trade_ts//1000**2)) or (depth is not None and (depth.ts_event//1000**2 != self.last_trade_ts//1000**2)): # 确保时间已经到了下一毫秒 不会有移动指针后回头还在更新volume的问题
            # 移动index_end,index_freq并记录price变化
            # 这里可能要么不能循环套循环 要么要修改数据格式(修改了数据格式)
            for t in range(len(self.left)):
                time_window = self.left[t]
                while self.index_end[t]<len(self.trade_times):
                    #尾指针经过新交易数据，移动index_freq，计算changerate，维护平方和
                    time_trade = self.trade_times[self.index_end[t]]
                    for i in range(len(self.freq)):
                        while self.index_freq[t][i]<len(self.trade_times)-1 \
                        and self.trade_times[self.index_freq[t][i]+1]<=time_trade-self.freq[i]*1000**3:
                            self.index_freq[t][i] += 1
                        self.change[i][self.index_end[t]]=self.trade_prices[self.index_end[t]]/self.trade_prices[self.index_freq[t][i]]-1
                        if self.change[i][self.index_end[t]]>0:
                            self.squaresum_up[t][i] += self.change[i][self.index_end[t]]**2
                        elif self.change[i][self.index_end[t]]<0:
                            self.squaresum_dw[t][i] += self.change[i][self.index_end[t]]**2
                    self.index_end[t] += 1
                    
            # 移动index_start 并维护平方和
            for t in range(len(self.left)):
                time_window = self.left[t]
                while self.index_start[t]<len(self.trade_times) \
                and self.trade_times[self.index_start[t]]<self.trade_times[-1]-time_window*1000**3:
                    #头指针移动，更新两个squaresum数组
                    for i in range(len(self.freq)):
                        if self.change[i][self.index_start[t]]>0:
                            self.squaresum_up[t][i] -= self.change[i][self.index_start[t]]**2
                        elif self.change[i][self.index_start[t]]<0:
                            self.squaresum_dw[t][i] -= self.change[i][self.index_start[t]]**2
                    self.index_start[t] += 1
                        
            # 截短trade_volumes,trade_amounts,trade_times列表，并修正所有指针
            min_index = min(self.index_start)
            '''if len(self.trade_volumes)-min_index>1000*max(self.right):
                print(min_index,len(self.trade_volumes),self.trade_times[-1]==self.last_trade_ts)
                print(self.right_start_index,self.left_start_index,self.right_end_index,self.left_start_index)'''
            self.trade_times = self.trade_times[min_index:]
            self.trade_volumes = self.trade_volumes[min_index:]
            self.trade_amounts = self.trade_amounts[min_index:]
            self.trade_prices = self.trade_prices[min_index:]
            for t in range(len(self.left)):
                self.index_start[t] -= min_index
                self.index_end[t] -= min_index
                for i in range(len(self.freq)):
                    self.index_freq[t][i] -= min_index
            
            # print(self.right_start_index,self.left_start_index,self.right_end_index,self.left_start_index)
            
            self.latest_trade_ts = None
            self.signal = False
            # 将最新的交易数据时间戳更新为当前交易数据时间戳
        
    def calculate(self):
        '''
        计算自定义特征
        returns: [(fe_name, {'tp': int, 'ret':}), ...]
        '''
        # 在这里实现自定义特征的计算逻辑
        # 可能需要再次更新4个指针!!!!!
        if not self.signal:
            return None
        res = []
        for t in range(len(self.left)):
            for i in range(len(self.freq)):
                time_window = self.left[t]
                freq_tmp = self.freq[i]
                up_ratio,dw_ratio = 0.5,0.5
                # print(self.squaresum_up[t][i],self.squaresum_dw[t][i])
                if self.squaresum_up[t][i]+self.squaresum_dw[t][i]>0:
                    up_ratio = self.squaresum_up[t][i]/(self.squaresum_up[t][i]+self.squaresum_dw[t][i])
                    dw_ratio = self.squaresum_dw[t][i]/(self.squaresum_up[t][i]+self.squaresum_dw[t][i])
                # 返回特征结果
                res.append((f'{self.fe_name1}_{time_window}_{freq_tmp}_up', self.latest_trade_ts, up_ratio))
                res.append((f'{self.fe_name1}_{time_window}_{freq_tmp}_down', self.latest_trade_ts, dw_ratio))
            # print(res)
        # self.last_data = self.get_trade_data(0)
        return res

    
if __name__ == '__main__':
    # initlog(None, 'ofi_feature.log', logging.INFO)
    import time
    start = time.time()
    config_list = []
    for i in range(9):
        config_list.append(FeatureConfig(left = 2**(i-1) if i >0 else 0, right = 2**i))

    begin_time = datetime.datetime(2024, 3, 27, 21)
    end_time = datetime.datetime(2024, 3, 27, 21, 59)
    exchange = 'binance'
    symbol = 'btc_usdt'
    trans_config = config_list[0]

    ins = PastReturnGenerator(trans_config)
    fe_list = []

    for idx, row in enumerate(get_data_generator(begin_time, end_time, exchange, symbol)):
        # print(idx, row)

        if idx >= 300:
            break
        
        if row[1] == 'ticker':
            fe_list.append(ins.process(ticker=row[0]))
        elif row[1] == 'depth':
            fe_list.append(ins.process(depth=row[0]))
        else:
            fe_list.append(ins.process(trade=row[0]))
            print(idx, row)
            # print(fe_list[-1])

    print(time.time()-start)
