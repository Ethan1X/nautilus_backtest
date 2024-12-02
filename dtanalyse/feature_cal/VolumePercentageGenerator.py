import os, sys, logging, math
from bisect import bisect_left,bisect_right

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# from util.load_s3_data import LoadS3Data
from util.time_method import *
from data_generator import *
from typing import TypedDict
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
from util.statistic_method_v2 import add_data, DataInfo

class VolumePercentageGenerator(FeatureGenerator):
    '''
    Author: Zeng Yunjia
    Feature: to be added
    '''
    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        # FeatureConfig中包括time(e.g [1,5,10,20,40,80,160])和percentage（e.g [50,80,95]）两个参数，分别记录了回望时间与需要保留的trade数据
        # 每次on trade需要更新trade_volume列表，用log_n时间插入&删除……有点不想这么搞 略麻烦orz(留一下分buy sell的空间？)
        # 横向比较：买卖分位数相减（实现）/买卖交易按成交量分位数分好之后计算价格etc.(未实现)
        # 纵向比较：分位数&分位数的差的变化（不一定make sense 需要时间窗口长一些）
        
        self.data_mode = "normal"
        if "data_mode" in config.keys():
            self.data_mode = config['data_mode']

        self.time = config['time']
        self.percentage = config['percentage']
        self.time_len = len(self.time)
        self.percentage_len = len(self.percentage)
        
        self.fe_name1 = f'VolumePercentage'
        self.latest_trade_ts = None
        self.last_trade_ts = None
        self.trade_volumes_buy = []
        self.trade_prices_buy = []
        self.trade_amounts_buy = []
        self.trade_times_buy = []
        self.trade_volumes_sell = []
        self.trade_prices_sell = []
        self.trade_amounts_sell = []
        self.trade_times_sell = []
        '''self.trade_volumes = []
        self.trade_prices = []
        self.trade_amounts = []
        self.trade_times = []'''
        self.buy_end_index = [0 for i in range(self.time_len)]
        self.buy_start_index = [0 for i in range(self.time_len)]
        self.sell_end_index = [0 for i in range(self.time_len)]
        self.sell_start_index = [0 for i in range(self.time_len)]
        ##更新有序的trade_volumes 从小到大排列
        self.trade_volumes_buy_sorted = [[] for i in range(self.time_len)]
        self.trade_prices_buy_sorted = [[] for i in range(self.time_len)]
        self.trade_amounts_buy_sorted = [[] for i in range(self.time_len)]
        self.trade_times_buy_sorted = [[] for i in range(self.time_len)]
        self.trade_volumes_sell_sorted = [[] for i in range(self.time_len)]
        self.trade_prices_sell_sorted = [[] for i in range(self.time_len)]
        self.trade_amounts_sell_sorted = [[] for i in range(self.time_len)]
        self.trade_times_sell_sorted = [[] for i in range(self.time_len)]
        self.signal = False
        

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
    
    def add_to_sorted_list(self, volume, price, amount, timestamp, time_index, side):
        if side == 'buy':
            index_add = bisect_right(self.trade_volumes_buy_sorted[time_index],volume)
            self.trade_volumes_buy_sorted[time_index].insert(index_add,volume)
            self.trade_prices_buy_sorted[time_index].insert(index_add,price)
            self.trade_amounts_buy_sorted[time_index].insert(index_add,amount)
            self.trade_times_buy_sorted[time_index].insert(index_add,timestamp)
            
        elif side == 'sell':
            index_add = bisect_right(self.trade_volumes_sell_sorted[time_index],volume)
            self.trade_volumes_sell_sorted[time_index].insert(index_add,volume)
            self.trade_prices_sell_sorted[time_index].insert(index_add,price)
            self.trade_amounts_sell_sorted[time_index].insert(index_add,amount)
            self.trade_times_sell_sorted[time_index].insert(index_add,timestamp)
            
        
    def delete_from_sorted_list(self, volume, price, amount, timestamp, time_index, side):
        if side == 'buy':
            index_del = bisect_left(self.trade_volumes_buy_sorted[time_index],volume)
            '''print(index_del,len(self.trade_volumes_buy_sorted[time_index]))
            if index_del == len(self.trade_volumes_buy_sorted[time_index]):
                print(volume,timestamp,self.last_trade_ts,self.trade_volumes_buy_sorted[time_index])
                print(self.trade_volumes_buy,self.trade_times_buy)'''
            self.trade_volumes_buy_sorted[time_index].pop(index_del)
            self.trade_prices_buy_sorted[time_index].pop(index_del)
            self.trade_amounts_buy_sorted[time_index].pop(index_del)
            self.trade_times_buy_sorted[time_index].pop(index_del)
            
        if side == 'sell':
            index_del = bisect_left(self.trade_volumes_sell_sorted[time_index],volume)
            '''print(index_del,len(self.trade_volumes_sell_sorted[time_index]))
            if index_del == len(self.trade_volumes_sell_sorted[time_index]):
                print(volume,timestamp,self.last_trade_ts,self.trade_volumes_sell_sorted[time_index])
                print(self.trade_volumes_sell,self.trade_times_sell)'''
            self.trade_volumes_sell_sorted[time_index].pop(index_del)
            self.trade_prices_sell_sorted[time_index].pop(index_del)
            self.trade_amounts_sell_sorted[time_index].pop(index_del)
            self.trade_times_sell_sorted[time_index].pop(index_del)
        

    def update(self, ticker: QuoteTick = None, trade: TradeTick = None, depth: OrderBook = None):
        '''
        更新自定义特征的内部状态
        '''
        # print(ticker,trade,depth)
        # 检查最新的交易数据时间戳是否更新，如果更新则清空交易量列表
        if trade is not None:
            self.signal = True
            if self.last_trade_ts is None or (trade.ts_event // 1000 != self.last_trade_ts // 1000):
                self.last_trade_ts = trade.ts_event
                # 在这里实现自定义特征的更新逻辑
                # 将trade数据插入队列 更新time,volume,amount列表
                if trade.aggressor_side == 'buy':
                    self.trade_times_buy.append(trade.ts_event)
                    self.trade_volumes_buy.append(trade.size*trade.price)
                    self.trade_amounts_buy.append(trade.size)
                elif trade.aggressor_side == 'sell':
                    self.trade_times_sell.append(trade.ts_event)
                    self.trade_volumes_sell.append(trade.size*trade.price)
                    self.trade_amounts_sell.append(trade.size)
                
            else:
                # 如果是相同时刻的trade，直接更新当前时刻，把buy和sell分开了（偷了个小懒没有再开一个self.buy_ts self.sell_ts记录是否这微秒内已经有buy/sell数据了）
                if trade.aggressor_side == 'buy':
                    last_buy_ts = self.trade_times_buy[-1]
                    if (trade.ts_event // 1000 != last_buy_ts // 1000):
                        self.trade_times_buy.append(trade.ts_event)
                        self.trade_volumes_buy.append(trade.size*trade.price)
                        self.trade_amounts_buy.append(trade.size)
                    else:
                        self.trade_volumes_buy[-1] += trade.size*trade.price
                        self.trade_amounts_buy[-1] += trade.size
                elif trade.aggressor_side == 'sell':
                    last_sell_ts = self.trade_times_sell[-1]
                    if (trade.ts_event // 1000 != last_sell_ts // 1000):
                        self.trade_times_sell.append(trade.ts_event)
                        self.trade_volumes_sell.append(trade.size*trade.price)
                        self.trade_amounts_sell.append(trade.size)
                    else:
                        self.trade_volumes_sell[-1] += trade.size*trade.price
                        self.trade_amounts_sell[-1] += trade.size
            self.latest_trade_ts = trade.ts_event
            
        elif (ticker is not None and (ticker.ts_event//1000 != self.last_trade_ts//1000)) or (depth is not None and (depth.ts_event//1000 != self.last_trade_ts//1000)): # 确保到了下一微秒 不会有移动指针后回头还在更新volume的问题
            # 移动buy_end_index,sell_end_index指针 并维护一堆有序列表
            for i in range(self.time_len):
   
                if self.buy_end_index[i]<len(self.trade_volumes_buy):
                    while self.buy_end_index[i]<len(self.trade_volumes_buy):
                        timestamp_tmp = self.trade_times_buy[self.buy_end_index[i]]
                        volume_tmp = self.trade_volumes_buy[self.buy_end_index[i]]
                        price_tmp = self.trade_volumes_buy[self.buy_end_index[i]]/self.trade_amounts_buy[self.buy_end_index[i]]
                        amount_tmp = self.trade_amounts_buy[self.buy_end_index[i]]
                        self.add_to_sorted_list(volume=volume_tmp,price=price_tmp,amount=amount_tmp,\
                                         timestamp=timestamp_tmp,time_index=i,side='buy')
                        self.buy_end_index[i] += 1
                        
                if self.sell_end_index[i]<len(self.trade_volumes_sell):
                    while self.sell_end_index[i]<len(self.trade_volumes_sell):
                        timestamp_tmp = self.trade_times_sell[self.sell_end_index[i]]
                        volume_tmp = self.trade_volumes_sell[self.sell_end_index[i]]
                        price_tmp = self.trade_volumes_sell[self.sell_end_index[i]]/self.trade_amounts_sell[self.sell_end_index[i]]
                        amount_tmp = self.trade_amounts_sell[self.sell_end_index[i]]
                        self.add_to_sorted_list(volume=volume_tmp,price=price_tmp,amount=amount_tmp,\
                                         timestamp=timestamp_tmp,time_index=i,side='sell')
                        self.sell_end_index[i] += 1
                    
            # 移动buy_start_index,sell_start_index指针 并维护和 需要确认一下时间戳是不是都是毫秒 ……dbq，确认了，是纳秒……
            for i in range(self.time_len):
                
                if len(self.trade_times_buy)>0 and self.buy_start_index[i]<len(self.trade_times_buy)\
                and self.trade_times_buy[self.buy_start_index[i]]<self.last_trade_ts-self.time[i]*1000**3:
                    
                    while self.buy_start_index[i]<len(self.trade_times_buy) \
                    and self.trade_times_buy[self.buy_start_index[i]]<self.last_trade_ts-self.time[i]*1000**3:
                        timestamp_tmp = self.trade_times_buy[self.buy_start_index[i]]
                        volume_tmp = self.trade_volumes_buy[self.buy_start_index[i]]
                        price_tmp = self.trade_volumes_buy[self.buy_start_index[i]]/self.trade_amounts_buy[self.buy_start_index[i]]
                        amount_tmp = self.trade_amounts_buy[self.buy_start_index[i]]
                        self.delete_from_sorted_list(volume=volume_tmp,price=price_tmp,amount=amount_tmp,\
                                         timestamp=timestamp_tmp,time_index=i,side='buy')
                        self.buy_start_index[i] += 1
                        
                if len(self.trade_times_sell)>0 and self.sell_start_index[i]<len(self.trade_times_sell) \
                and self.trade_times_sell[self.sell_start_index[i]]<self.last_trade_ts-self.time[i]*1000**3:
                    
                    while self.sell_start_index[i]<len(self.trade_times_sell) \
                    and self.trade_times_sell[self.sell_start_index[i]]<self.last_trade_ts-self.time[i]*1000**3:
                        timestamp_tmp = self.trade_times_sell[self.sell_start_index[i]]
                        volume_tmp = self.trade_volumes_sell[self.sell_start_index[i]]
                        price_tmp = self.trade_volumes_sell[self.sell_start_index[i]]/self.trade_amounts_sell[self.sell_start_index[i]]
                        amount_tmp = self.trade_amounts_sell[self.sell_start_index[i]]
                        self.delete_from_sorted_list(volume=volume_tmp,price=price_tmp,amount=amount_tmp,\
                                         timestamp=timestamp_tmp,time_index=i,side='sell')
                        self.sell_start_index[i] += 1
                        
            # 截短trade_volumes,trade_amounts,trade_times列表，并修正所有指针 把buy sell分开
            min_index_buy,min_index_sell = min(self.buy_start_index),min(self.sell_start_index)
            '''if len(self.trade_volumes)-min_index>1000*max(self.right):
                print(min_index,len(self.trade_volumes),self.trade_times[-1]==self.last_trade_ts)
                print(self.right_start_index,self.left_start_index,self.right_end_index,self.left_start_index)'''
            self.trade_times_buy = self.trade_times_buy[min_index_buy:]
            self.trade_volumes_buy = self.trade_volumes_buy[min_index_buy:]
            self.trade_amounts_buy = self.trade_amounts_buy[min_index_buy:]
            self.trade_times_sell = self.trade_times_sell[min_index_sell:]
            self.trade_volumes_sell = self.trade_volumes_sell[min_index_sell:]
            self.trade_amounts_sell = self.trade_amounts_sell[min_index_sell:]
            for i in range(self.time_len):
                self.buy_end_index[i] -= min_index_buy
                self.buy_start_index[i] -= min_index_buy
                self.sell_end_index[i] -= min_index_sell
                self.sell_start_index[i] -= min_index_sell
            
            # print(self.buy_start_index,self.sell_start_index,self.buy_end_index,self.sell_start_index)
            
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
        side_cal = ['buy','sell','buy-sell']
        for i in range(len(self.percentage)):
            for j in range(len(self.time)):
                volume_pct_buy,volume_pct_sell,volume_pct_buy_sell = 0.0,0.0,0.0
                pct = self.percentage[i]/100
                time_window = self.time[j]
                if len(self.trade_volumes_buy_sorted[j])>0:
                    pct_index = min(math.ceil(len(self.trade_volumes_buy_sorted[j])*pct),len(self.trade_volumes_buy_sorted[j])-1)
                    volume_pct_buy = self.trade_volumes_buy_sorted[j][pct_index]
                if len(self.trade_volumes_sell_sorted[j])>0:
                    pct_index = min(math.ceil(len(self.trade_volumes_sell_sorted[j])*pct),len(self.trade_volumes_sell_sorted[j])-1)
                    volume_pct_sell = self.trade_volumes_sell_sorted[j][pct_index]
                volume_pct_buy_sell = volume_pct_buy-volume_pct_sell
                # 返回特征结果
                res.append((f'{self.fe_name1}_buy_{time_window}_{pct}', self.latest_trade_ts, volume_pct_buy))
                res.append((f'{self.fe_name1}_sell_{time_window}_{pct}', self.latest_trade_ts, volume_pct_sell))
                res.append((f'{self.fe_name1}_buy_sell_{time_window}_{pct}', self.latest_trade_ts, volume_pct_buy_sell))
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
