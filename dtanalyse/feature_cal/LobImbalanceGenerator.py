import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# from util.load_s3_data import LoadS3Data
from util.time_method import *
from data_generator import *
from typing import TypedDict
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
from util.statistic_method_v2 import DataInfo, ts_to_second, get_second_shifted

class LobImbalanceGenerator(FeatureGenerator):
    '''
    Author: Li Haotong
    Reviewer: Qiushi Bu
    Feature: 13
    '''
    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        self.left = config['left']
        self.right = config['right']
        self.min_max = (min(self.left), max(self.right))
        self.window_len = self.min_max[1] + 1
        self.period = len(self.right)
        self.fe_name1 = f'LobImbalance'
        self.fe_name2 = f'LobImbalanceSTD'
        self.latest_trade_ts = None
        self.tick_ts_index = np.zeros(self.min_max[1])
        self.last_trade_ts = None
        self.load_ts_num = 0
        self.tick_ask = []
        self.tick_bid = []
        self.signal = False
        self.data_index = None

        self.first_time = True
        # self.history_ticker = np.array([0,0]) #初始化历史ticker信息，格式为[时间，oir]
        self.time_level = 1e9 # 交易所的时间单位为ns，left和right的时间单位是秒，如果输入的时间单位不同，则需要相应更改
        self.lobimb_sum = [0.0] * self.period
        self.lobimb_square_sum = [0.0] * self.period
        self.lobimb_count = [0] * self.period
        self.history_ticker = []
        self.history_sum = []
        self.history_square_sum = []
        self.history_ts = []
        self.left_index = [0] * self.period
        self.right_index = [0] * self.period
        self.last_data = None

    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None):
        self.load_ts_num = load_sum
        if self.data_index is not None and data_index != self.data_index:
            self.last_data = None
            self.data_index = data_index
            self.history_sum[self.data_index] = 0.0
            self.history_square_sum[self.data_index] = 0.0
            self.history_ticker[self.data_index] = 0

            for i in range(self.period):
                left, right = self.left[i], self.right[i]
                if self.load_ts_num > left:
                    # 新数据，窗口滑动
                    self.calc_feat_value(left, right, i)
        
        if self.latest_trade_ts is None and price_list_buy is not None:
            self.trade_volumes_buy = volumes_list_buy
            self.trade_volumes_sell = volumes_list_sell
            self.data_index = -1
            self.window_len = len(self.trade_volumes_buy)
            self.history_ticker = np.zeros(self.window_len)
            self.history_sum = np.zeros(self.window_len)
            self.history_square_sum = np.zeros(self.window_len)

    def get_data_index(self, shift):
        return (self.window_len + self.data_index - shift) % self.window_len
        
    def update_signal(self, trade: TradeTick=None):
        if trade is not None:
            self.latest_trade_ts = trade.ts_event
            self.signal = True
        else:
            self.latest_trade_ts = None
            self.signal = False

    def process(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
        '''
            feature计算主流程 供外部调用
        '''
        if trade is not None:
            self.update_signal(trade)
        feature_ret = None
        if ticker is not None:
            self.update(ticker)
            feature_ret = self.calculate()

        return feature_ret
    
    def update(self, ticker: QuoteTick):  
        '''
        更新信息
        '''
        self.ts_event = ticker.ts_event
        #添加新数据,这里直接计算出lobi并储存次数与lobi值
        lobimb = (ticker.ask_size - ticker.bid_size)/(ticker.bid_size + ticker.ask_size)
        lobimb_p2 = lobimb ** 2
        # self.history_ticker = np.vstack((self.history_ticker, [self.ts_event, oir]))

        if self.load_ts_num == 0:
            return

        self.history_ticker[self.data_index] += 1
        
        #if self.last_data is not None:
        self.last_data = lobimb
        
        self.history_sum[self.data_index] += lobimb
        self.history_square_sum[self.data_index] += lobimb_p2

        return

    def calc_feat_value(self, left, right, period, old_data=None):
        if old_data is None and self.load_ts_num > right:
            _old_data_index = self.get_data_index(right+1)
        else:
            _old_data_index = None        
        _new_data_index = self.get_data_index(left)

        _old_lobimb_square_sum = self.lobimb_square_sum[period]
        _old_lobimb_sum = self.lobimb_sum[period]
        _old_lobimb_count = self.lobimb_count[period]
        if _old_data_index is None or _old_data_index < 0:
            _old_sum = _old_square_sum = 0.0
            _old_count = 0
        else:
            _old_sum = self.history_sum[_old_data_index]
            _old_square_sum = self.history_square_sum[_old_data_index]
            _old_count = self.history_ticker[_old_data_index]

        if old_data is None:
            _new_sum = self.history_sum[_new_data_index]
            _new_square_sum = self.history_square_sum[_new_data_index]
            _new_count = self.history_ticker[_new_data_index]
        else:
            _new_sum = old_data
            _new_square_sum = old_data ** 2
            _new_count = 1

        _new_square_sum = _old_lobimb_square_sum - _old_square_sum + _new_square_sum
        _new_sum = _old_lobimb_sum - _old_sum + _new_sum

        _cur_count = _new_count
        _new_count = _old_lobimb_count - _old_count + _new_count

        _new_mean = 0.0
        if _new_count > 0:
            _new_mean = _new_sum / _new_count
        _new_std = _new_square_sum - 2 * _new_mean * _new_sum + _new_mean ** 2 * _new_count
        if _new_count > 1:
            _new_std = (_new_std / (_new_count - 1)) ** 1/2
        else:
            _new_std = None
        # if period == 0:
        #     print(self.history_ticker[-left-1:-right])
        # print(f'lobi calc values: {period} {left}/{_new_data_index} {right}/{_old_data_index}  {self.load_ts_num} {self.history_ticker[_new_data_index]} {_old_lobimb_count}-{_old_count}+{_cur_count}={_new_count}')

        self.lobimb_square_sum[period] = _new_square_sum
        self.lobimb_sum[period] = _new_sum
        self.lobimb_count[period] = _new_count
        # print(f'values: {period} {self.lobimb_square_sum[period]},{self.lobimb_sum[period]},{self.lobimb_count[period]}')
        return _new_mean, _new_std

    def calculate(self):     
        if not self.signal:
            return None
            
        res = []
        for i in range(self.period):
            left, right = self.left[i], self.right[i]

            lob_imbalance = lob_imbalance_std = None
            if self.load_ts_num > left:
                if self.last_data is None:
                    # 新数据，窗口滑动
                    lob_imbalance, lob_imbalance_std = self.calc_feat_value(left, right, i)
                else:
                    # 仅更新当前数据
                    if left > 0 and self.load_ts_num >= right:
                        if self.lobimb_count[i] > 0:
                            _count = self.lobimb_count[i]
                            _sum = self.lobimb_sum[i]
                            lob_imbalance =  _sum / _count
                            lob_imbalance_std = self.lobimb_square_sum[i] - 2 * lob_imbalance * _sum + lob_imbalance ** 2 * _count
                            if _count > 1:
                                lob_imbalance_std = (lob_imbalance_std / (_count - 1)) ** 1/2
                            else:
                                lob_imbalance_std = None
                    else:
                        lob_imbalance, lob_imbalance_std = self.calc_feat_value(left, right, i, self.last_data)

            # print(f'lobi calc: {self.load_ts_num} {left} {right} {len(self.history_ticker)} {self.data_index} {self.last_data} {lob_imbalance_std}')
            if self.load_ts_num >= right:
                res.extend([(f'{self.fe_name1}_{left}_{right}', self.latest_trade_ts, lob_imbalance),
                           (f'{self.fe_name2}_{left}_{right}', self.latest_trade_ts, lob_imbalance_std)])

        # self.last_data = self.history_sum[self.data_index]
        return res
    
    # def update(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
    #     '''
    #     更新自定义特征的内部状态
    #     '''
    #     # 检查最新的交易数据时间戳是否更新，如果更新则清空交易量列表
    #     if ticker is not None:
    #         self.tick_ask.append(ticker.ask_size)
    #         self.tick_bid.append(ticker.bid_size)
    #     if (trade is not None) and (not self.last_trade_ts or trade.ts_event // 1000 != self.last_trade_ts // 1000):
    #         # 在这里实现自定义特征的更新逻辑
    #         # 如果列表中数据量超过 right，则删除最早的数据
    #         if self.load_ts_num == self.min_max[1]:
    #             origin_len = int(self.tick_ts_index[0])
    #             self.tick_ts_index[0:-1] = self.tick_ts_index[1:] - origin_len
    #             self.tick_ask = self.tick_ask[origin_len:]
    #             self.tick_bid = self.tick_bid[origin_len:]
    #             self.tick_ts_index[-1] = len(self.tick_ask)
    #         else:
    #             self.load_ts_num = self.load_ts_num + 1
    #             self.tick_ts_index[self.load_ts_num - 1] = len(self.tick_ask)
    #         self.latest_trade_ts = trade.ts_event
    #         self.last_trade_ts = trade.ts_event
    #         self.signal = True
    #     else:
    #         if trade is not None:
    #             self.latest_trade_ts = trade.ts_event
    #             self.signal = True
    #         else:
    #             self.latest_trade_ts = None
    #             self.signal = False
    #             # 将最新的交易数据时间戳更新为当前交易数据时间戳

    # def calculate(self):
    #     '''
    #     计算自定义特征
    #     returns: [(fe_name, {'tp': int, 'ret':}), ...]
    #     '''
    #     # 在这里实现自定义特征的计算逻辑
    #     if not self.signal:
    #         return None
    #     res = []
    #     for i in range(len(self.left)):
    #         left, right = self.left[i], self.right[i]
    #         if (len(self.tick_ask) >= self.tick_ts_index[-left]) and (self.tick_ts_index[-left] != 0):
    #             a = np.array(self.tick_ask[int(self.tick_ts_index[max(left-1,0)]):int(self.tick_ts_index[right-1])])
    #             b = np.array(self.tick_bid[int(self.tick_ts_index[max(left-1,0)]):int(self.tick_ts_index[right-1])])
    #             a_ = (a-b)/(a+b)
    #             lob_imbalance = np.nanmean(a_)
    #             lob_imbalance_std = np.sqrt(np.nanmean((a_ - lob_imbalance)**2))
    #             res.extend([(f'{self.fe_name1}_{left}_{right}', self.latest_trade_ts, lob_imbalance),
    #                        (f'{self.fe_name2}_{left}_{right}', self.last_trade_ts, lob_imbalance_std)])
                
    #     return res

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

    ins = LobImbalanceGenerator(trans_config)
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
