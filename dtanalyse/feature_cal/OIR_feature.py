import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_generator import *
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
import numpy as np
from util.statistic_method_v2 import DataInfo, ts_to_second, get_second_shifted

class OIRGenerator(FeatureGenerator):
    '''
    特征名称：'oir'，订单不平衡率（Order Imbalance Ratio）
    特征编号：25
    所需数据：ticker数据
    Author: Qiushi Bu
    Reviewer: Haotong Li
    Notes: 如果输入为一个区间，则返回时间段内OIR值的均值，求和与标准差三个值。如果输入(0,0)，则只返回的sum和avg是相同的，std是0
    '''

    def __init__(self, config: FeatureConfig) -> None:
        '''
        特征名称：'oir'
        config: 'left','right': 回望区间的两端
        '''
        super().__init__(config)
        self.left = config['left']
        self.right = config['right']
        self.min_max = (min(self.left), max(self.right))
        self.window_len = self.min_max[1]
        self.period = len(self.right)
        self.fe_name1 = f'oir_avg' 
        self.fe_name2 = f'oir_sum' 
        self.fe_name3 = f'oir_std' 
        self.first_time = True
        # self.history_ticker = np.array([0,0]) #初始化历史ticker信息，格式为[时间，oir]
        self.time_level = 1e9 # 交易所的时间单位为ns，left和right的时间单位是秒，如果输入的时间单位不同，则需要相应更改
        self.oir_sum = [0.0] * self.period
        self.oir_square_sum = [0.0] * self.period
        self.oir_count = [0] * self.period
        self.history_ticker = []
        self.history_sum = []
        self.history_square_sum = []
        self.history_ts = []
        self.left_index = [0] * self.period
        self.right_index = [0] * self.period
        self.last_data = None
        
    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None):
        pass

    def process(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
        '''
        首先判断来的数据是不是ticker数据，如果是，进行信息更新，如果不是，不进行处理，返回None
        '''
        if ticker is None:  #如果没有传来ticker数据，则不更新,返回None
            # return  [(self.fe_name1, None, None),(self.fe_name2, None, None),(self.fe_name3, None, None)]
            return None

        self.update(ticker)
        feature_ret = self.calculate(ticker)

        return feature_ret

    def get_data_index(self, left, right, period):
        left_index = self.left_index[period]
        right_index = self.right_index[period]
        _new_left_time = self.history_ts[-1] - left
        _new_right_time = self.history_ts[-1] - right
        if _new_right_time < self.history_ts[0]:
            right_index = None
        if _new_left_time < self.history_ts[0]:
            left_index = None
        if right_index is not None and left_index is not None:
            while self.history_ts[right_index] < _new_right_time:
                right_index += 1
            while self.history_ts[left_index] < _new_left_time:
                left_index += 1
            # if self.history_ts[left_index] > _new_left_time:
            #     left_index -= 1
        return left_index, right_index

    def update(self, ticker: QuoteTick):  
        '''
        更新信息:首先将新传入的ticker数据加入到history_ticker中，然后将时间窗之外的数据删除
        '''
        self.ts_event = ticker.ts_event
        #添加新数据,这里直接计算出oir并储存时间与oir值
        oir = (ticker.bid_size - ticker.ask_size)/(ticker.bid_size + ticker.ask_size)
        oir_p2 = oir ** 2
        # self.history_ticker = np.vstack((self.history_ticker, [self.ts_event, oir]))

        if len(self.history_sum) == 0:
            self.history_ticker.append(1)
            self.history_sum.append(oir)
            self.history_square_sum.append(oir_p2)
            self.history_ts.append(ts_to_second(self.ts_event, self.time_level))
            self.last_data = None
        else:
            _new_second = ts_to_second(self.ts_event, self.time_level)
            _old_second = self.history_ts[-1]
            if _new_second > _old_second:
                while _new_second > _old_second + 1:
                    _old_second += 1
                    self.history_ticker.append(0)
                    self.history_sum.append(0.0)
                    self.history_square_sum.append(0.0)
                    self.history_ts.append(_old_second)
                    
                self.history_ticker.append(1)
                self.history_sum.append(oir)
                self.history_square_sum.append(oir_p2)
                self.history_ts.append(_new_second)
                self.last_data = None
            else:
                self.history_ticker[-1] += 1
                self.last_data = oir
                self.history_sum[-1] += oir
                self.history_square_sum[-1] += oir_p2

        
        #找到需要删除的数据的时间点，并保留后面的
        # del_time = self.ts_event - self.window_len * self.time_level #由于ticker时间戳为ns，config时间戳为s，需要乘一个time_level
        # self.history_ticker = self.history_ticker[self.history_ticker[:, 0] > del_time]
                
        return 

    def calc_feat_value(self, left, left_index, right, right_index, period, old_data=None):
        _old_square_sum = self.oir_square_sum[period]
        _old_sum = self.oir_sum[period]
        _old_count = self.oir_count[period]
        
        _old_oir_sum = _new_oir_sum = 0.0
        _old_oir_square_sum = _new_oir_square_sum = 0.0
        _old_oir_count = _new_oir_count = 0
        
        _old_left_index = self.left_index[period]
        _old_right_index = self.right_index[period]
        if old_data is None:
            # 滚动窗口数据
            if right_index is not None:
                while _old_right_index < right_index:
                    _old_oir_sum += self.history_sum[_old_right_index]
                    _old_oir_square_sum += self.history_square_sum[_old_right_index]
                    _old_oir_count += self.history_ticker[_old_right_index]
                    _old_right_index += 1
            if left_index is not None:
                while _old_left_index < left_index:
                    _new_oir_sum += self.history_sum[_old_left_index]
                    _new_oir_square_sum += self.history_square_sum[_old_left_index]
                    _new_oir_count += self.history_ticker[_old_left_index]
                    _old_left_index += 1
        else:
            _new_oir_sum = old_data
            _new_oir_square_sum = old_data ** 2
            _new_oir_count = 1

        _new_square_sum = _old_square_sum + _new_oir_square_sum - _old_oir_square_sum
        _new_sum = _old_sum + _new_oir_sum - _old_oir_sum
        _new_count = _old_count + _new_oir_count - _old_oir_count

        _new_mean = 0.0
        if _new_count > 0:
            _new_mean = _new_sum / _new_count
        _new_std = _new_square_sum - 2 * _new_mean * _new_sum + _new_mean ** 2 * _new_count
        if _new_count > 1:
            _new_std = (_new_std / (_new_count - 1)) ** 1/2
        else:
            _new_std = None

        self.oir_square_sum[period] = _new_square_sum
        self.oir_sum[period] = _new_sum
        self.oir_count[period] = _new_count

        self.left_index[period] = left_index
        if left_index is None:
            self.left_index[period] = 0
        self.right_index[period] = right_index
        if right_index is None:
            self.right_index[period] = 0
        # print(f'oir calc: {period} {left}/{left_index} {right}/{right_index} {_new_count}/{_old_count} {_new_std}')
        return _new_mean, _new_sum, _new_std

    def calculate(self, ticker):            
        res = []
        for i in range(self.period):
            left, right = self.left[i], self.right[i]

            left_index, right_index = self.get_data_index(left, right, i)
            if left_index is None:
                # 数据范围未达到窗口宽度
                continue
            # print(f'calc: period({i}): {left}/{left_index} {right}/{right_index}; {len(self.history_sum)} {len(self.history_square_sum)} {len(self.history_ticker)}; {self.first_time} {self.window_len}')

            _ts = ticker.ts_event
            oir_avg = oir_sum = oir_std = None
            if right == 0: # 如果right=0，则输出当前时刻的oir值，     
                oir_avg = (ticker.bid_size - ticker.ask_size)/(ticker.bid_size + ticker.ask_size)
                oir_sum = oir_avg
                oir_std = 0
            else:
                if self.last_data is None:
                    # 新数据，窗口滑动
                    oir_avg, oir_sum, oir_std = self.calc_feat_value(left, left_index, right, right_index, i)
                else:
                    # 仅更新当前数据
                    # if left > 0:
                        continue
                    # oir_avg, oir_sum, oir_std = self.calc_feat_value(left, left_index, right, right_index, i, self.last_data)

            if not self.first_time or (right == self.window_len and right_index is not None):
                res_data1 = (f'{self.fe_name1}_{left}_{right}', _ts, oir_avg)
                res_data2 = (f'{self.fe_name2}_{left}_{right}', _ts, oir_sum)
                res_data3 = (f'{self.fe_name3}_{left}_{right}', _ts, oir_std)
                res.append(res_data1)
                if right != 0:
                    res.append(res_data2)
                    res.append(res_data3)
                self.first_time = False

        _new_second = ts_to_second(ticker.ts_event, self.time_level)
        _new_head_second = _new_second - self.window_len
        if _new_head_second > self.history_ts[0]:
            _new_head_index = 0
            while self.history_ts[_new_head_index] < _new_head_second:
                _new_head_index += 1

            del self.history_ticker[0:_new_head_index]
            del self.history_square_sum[0:_new_head_index]
            del self.history_sum[0:_new_head_index]
            del self.history_ts[0:_new_head_index]
            for i in range(self.period):
                self.left_index[i] -= _new_head_index
                if self.left_index[i] < 0:
                    self.left_index[i] = 0
                self.right_index[i] -= _new_head_index
                if self.right_index[i] < 0:
                    self.right_index[i] = 0
        return res

    # def calculate(self,ticker):
    #     '''
    #     直接计算OIR值
    #     '''
    #     res = []

    #     for i in range(len(self.left)):
    #         left, right = self.left[i], self.right[i]

    #         _ts = ticker.ts_event
    #         if right == 0: # 如果right=0，则输出当前时刻的oir值，     
    #             oir_avg = (ticker.bid_size - ticker.ask_size)/(ticker.bid_size + ticker.ask_size)
    #             oir_sum = oir_avg
    #             oir_std = 0
    #         else:          
    #             right_time = self.ts_event - left * self.time_level
    #             left_time = self.ts_event - right * self.time_level
    #             right_idx = np.searchsorted(self.history_ticker[:, 0], right_time, side='right') #找到回望区间对应的行数：0-idx
    #             left_idx = np.searchsorted(self.history_ticker[:, 0], left_time, side='right')
    #             if right_idx == 0: #如果idx=0，说明回望周期没到，返回None
    #                 oir_avg = oir_sum = oir_std = None
    #                 _ts = None
    #             else :
    #                 oir_avg = np.mean(self.history_ticker[left_idx:right_idx,1])
    #                 oir_sum = np.sum(self.history_ticker[left_idx:right_idx,1])
    #                 oir_std = np.std(self.history_ticker[left_idx:right_idx,1])
    #         res_data1 = (f'{self.fe_name1}_{left}_{right}', _ts, oir_avg)
    #         res_data2 = (f'{self.fe_name2}_{left}_{right}', _ts, oir_sum)
    #         res_data3 = (f'{self.fe_name3}_{left}_{right}', _ts, oir_std)
    #         res.append(res_data1)
    #         if right != 0:
    #             res.append(res_data2)
    #             res.append(res_data3)
    #     return res


if __name__ == '__main__':     
    initlog(None, 'oir_feature.log', logging.INFO)
    config_list = [] 
    for i in range(9):
        config_list.append(FeatureConfig(left = 2**(i-1)/10 if i > 0 else 0, right = 2**i/10))
    oir_config = config_list[0]
    begin_time = datetime.datetime(2024, 3, 27, 21,tzinfo=TZ_8) # 要加一个tzinfo
    end_time = datetime.datetime(2024, 3, 27, 21, 10,tzinfo=TZ_8)
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

    print(fe_list[:1000])
