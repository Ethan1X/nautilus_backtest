import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# from util.load_s3_data import LoadS3Data
from util.time_method import *
from data_generator import *
from typing import TypedDict
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
from util.statistic_method_v2 import add_data, DataInfo

class VwapPriceChangeRateGenerator(FeatureGenerator):
    '''
    Author: Zeng Yunjia
    Feature: to be added
    '''
    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        # FeatureConfig中包括left和right两个参数，所求的值为P_vwap(T-left,T)-P_vwap(T-right,T)/P_vwap(T-right,T)
        # 每次on trade需要更新trade_ts,last_trade_ts,volume_left_sum[period],volume_right_sum[period],amount_left_sum[period],amount_right_sum[period]
        
        self.data_mode = "normal"
        if "data_mode" in config.keys():
            self.data_mode = config['data_mode']

        self.left = config['left']
        self.right = config['right']
        print("config:",self.left,self.right)
        self.fe_name1 = f'VwapPriceChangeRate'
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
        ##vwap=volume/amount
        self.signal = False

        _periods = len(self.right)
        self.periods = len(self.right)
        # self.event_count = np.zeros(self.window_len) * np.nan
        # self.data_index = 0
        # self.counter = 0
        # self.last_data:QuoteTick = None
        # self.price_down_list = [[]] * _periods
        self.price_sum = [0.0] * _periods
        self.volume_left_sum = [0.0] * _periods
        self.volume_right_sum = [0.0] * _periods
        self.amount_left_sum = [0.0] * _periods
        self.amount_right_sum = [0.0] * _periods
        self.left_start_index = [0] * _periods
        self.right_start_index = [0] * _periods
        self.left_end_index = [0] * _periods
        self.right_end_index = [0] * _periods
        self.window_len = [self.right[i]*1000 - self.left[i]*1000 for i in range(_periods)]
        self.trade_count = [0] * _periods

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
            if self.last_trade_ts is None or (trade.ts_event != self.last_trade_ts):
                self.last_trade_ts = trade.ts_event
                # 在这里实现自定义特征的更新逻辑
                # 将trade数据插入队列 更新volume_sum,amount_sum列表
                self.trade_times.append(trade.ts_event)
                self.trade_volumes.append(trade.size*trade.price)
                self.trade_amounts.append(trade.size)
                
            else:
                # 如果是相同时刻的trade，直接更新当前时刻
                self.trade_volumes[-1] += trade.size*trade.price
                self.trade_amounts[-1] += trade.size
            self.latest_trade_ts = trade.ts_event
            
        else:
            # 移动right_end_index,left_end_index指针 并维护和
            for i in range(self.periods):
                if self.right_end_index[i]<len(self.trade_volumes):
                    while self.right_end_index[i]<len(self.trade_volumes):
                        self.volume_right_sum[i] += self.trade_volumes[self.right_end_index[i]]
                        self.amount_right_sum[i] += self.trade_amounts[self.right_end_index[i]]
                        self.right_end_index[i] += 1
                if self.left_end_index[i]<len(self.trade_volumes):
                    while self.left_end_index[i]<len(self.trade_volumes):
                        self.volume_left_sum[i] += self.trade_volumes[self.left_end_index[i]]
                        self.amount_left_sum[i] += self.trade_amounts[self.left_end_index[i]]
                        self.left_end_index[i] += 1
                    
            # 移动right_start_index,left_start_index指针 并维护和 需要确认一下时间戳是不是都是毫秒 ……是纳秒……
            for i in range(self.periods):
                if self.trade_times[self.right_start_index[i]]<self.last_trade_ts-self.right[i]*1000**3:
                    while self.trade_times[self.right_start_index[i]]<self.last_trade_ts-self.right[i]*1000**3:
                        self.volume_right_sum[i] -= self.trade_volumes[self.right_start_index[i]]
                        self.amount_right_sum[i] -= self.trade_amounts[self.right_start_index[i]]
                        self.right_start_index[i] += 1
                if self.trade_times[self.left_start_index[i]]<self.last_trade_ts-self.left[i]*1000**3:
                    while self.trade_times[self.left_start_index[i]]<self.last_trade_ts-self.left[i]*1000**3:
                        self.volume_left_sum[i] -= self.trade_volumes[self.left_start_index[i]]
                        self.amount_left_sum[i] -= self.trade_amounts[self.left_start_index[i]]
                        self.left_start_index[i] += 1
                        
            # 截短trade_volumes,trade_amounts,trade_times列表，并修正所有指针
            min_index = min(min(self.right_start_index),min(self.left_start_index))
            '''if len(self.trade_volumes)-min_index>1000*max(self.right):
                print(min_index,len(self.trade_volumes),self.trade_times[-1]==self.last_trade_ts)
                print(self.right_start_index,self.left_start_index,self.right_end_index,self.left_start_index)'''
            self.trade_times = self.trade_times[min_index:]
            self.trade_volumes = self.trade_volumes[min_index:]
            self.trade_amounts = self.trade_amounts[min_index:]
            for i in range(self.periods):
                self.right_end_index[i] -= min_index
                self.right_start_index[i] -= min_index
                self.left_end_index[i] -= min_index
                self.left_start_index[i] -= min_index
            
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
        for i in range(len(self.left)):
            left, right = self.left[i], self.right[i]
            change_rate = 0.0 # 或者left right有一个大于0才是0.0 其它时候是None
            
            if self.volume_left_sum[i]>0 and self.volume_right_sum[i]>0:
                price_left = self.volume_left_sum[i]/self.amount_left_sum[i]
                price_right = self.volume_right_sum[i]/self.amount_right_sum[i]
                change_rate = (price_left-price_right)/price_right
                # print(price_left,price_right,change_rate)
            # 返回特征结果
            res.append((f'{self.fe_name1}_{left}_{right}', self.latest_trade_ts, change_rate))
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
