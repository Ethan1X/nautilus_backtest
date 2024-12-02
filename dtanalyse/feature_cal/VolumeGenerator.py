import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# from util.load_s3_data import LoadS3Data
from util.time_method import *
from data_generator import *
from typing import TypedDict
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
from util.statistic_method_v2 import add_data, DataInfo


class VolumeGenerator(FeatureGenerator):
    '''
    Author: Li Haotong
    Reviewer: Qiushi Bu
    Feature: 9，10(变形，直接算Average)，11
    '''
    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        # FeatureConfig中包括left和right两个参数，回望时间为(T-right,T-left]
        self.data_mode = "normal"
        if "data_mode" in config.keys():
            self.data_mode = config['data_mode']
        
        self.left = config['left']
        self.right = config['right']
        self.min_max = (min(self.left), max(self.right))
        self.window_len = self.min_max[1]
        self.fe_name1 = f'VolumeAll'
        self.fe_name2 = f'VolumeAvg'
        self.fe_name3 = f'VolumeMax'
        self.latest_trade_ts = None # 记录当前时间戳
        self.last_trade_ts = None # 记录上一笔trade的时间
        self.load_ts_num = 0 # 记录已经存入trade的量
        self.trade_volumes_buy = np.zeros(self.window_len) * np.nan
        self.trade_price_buy = np.zeros(self.window_len) * np.nan
        self.trade_volumes_sell = np.zeros(self.window_len) * np.nan
        self.trade_price_sell = np.zeros(self.window_len) * np.nan
        self.signal = False

        _periods = len(self.right)
        self.event_count = np.zeros(self.window_len) * np.nan
        self.data_index = 0
        self.last_data:QuoteTick = None
        self.volume_down_list = [[]] * _periods
        self.volume_sum = [0.0] * _periods
        self.trade_count = [0] * _periods
        self.counter = 0

    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None):
        self.load_ts_num = load_sum
        if data_index != self.data_index:
            self.last_data = None
            self.data_index = data_index

        if self.latest_trade_ts is None and price_list_buy is not None:
            self.trade_volumes_buy = volumes_list_buy
            self.trade_volumes_sell = volumes_list_sell
            self.trade_price_buy = price_list_buy
            self.trade_price_sell = price_list_sell
            self.window_len = len(self.trade_volumes_buy)
            self.event_count = np.zeros(self.window_len) * np.nan

    def get_data_index(self, shift):
        return (self.window_len + self.data_index - shift) % self.window_len

    def get_trade_data(self, shift):
        _data_index = self.get_data_index(shift)
        _data = QuoteTick(self.trade_price_sell[_data_index], self.trade_volumes_sell[_data_index],
                          self.trade_price_buy[_data_index], self.trade_volumes_buy[_data_index], 
                          self.event_count[_data_index])
        return _data
        
    def process(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
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
            self.event_count[self.data_index] = self.counter
            self.counter += 1
            self.signal = True
        else:
            self.latest_trade_ts = None
            self.signal = False
        # print(f'vg update_signal: {self.signal} {self.latest_trade_ts} {trade}')

    def update(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
        '''
        更新自定义特征的内部状态
        '''
        # 检查最新的交易数据时间戳是否更新，如果更新则清空交易量列表
        if trade is not None:
            if self.last_trade_ts is None or ((trade.ts_event // 1000) != (self.last_trade_ts // 1000)):
                self.last_trade_ts = trade.ts_event
                # 在这里实现自定义特征的更新逻辑
                # 如果成交量列表中数据量超过 right，则删除最早的数据
                if self.load_ts_num == self.window_len:
                    # pop出最先进入的数据
                    self.trade_volumes_buy[:-1] = self.trade_volumes_buy[1:]
                    self.trade_volumes_sell[:-1] = self.trade_volumes_sell[1:]
                    # 重置最新的数据
                    if trade.aggressor_side == 'buy':
                        self.trade_volumes_buy[self.load_ts_num - 1] = trade.size
                        self.trade_volumes_sell[self.load_ts_num - 1] = np.nan
                    else:
                        self.trade_volumes_sell[self.load_ts_num - 1] = trade.size
                        self.trade_volumes_buy[self.load_ts_num - 1] = np.nan
                else:
                    self.load_ts_num = self.load_ts_num + 1 # 如果没有存满，不考虑pop问题，直接更新
                    if trade.aggressor_side == 'buy':
                        self.trade_volumes_buy[self.load_ts_num - 1] = trade.size
                    else:
                        self.trade_volumes_sell[self.load_ts_num - 1] = trade.size
            else:
                # 如果是相同时刻的trade，直接更新当前时刻
                if trade.aggressor_side == 'buy':
                    self.trade_volumes_buy[self.load_ts_num - 1] = np.nansum([self.trade_volumes_buy[self.load_ts_num - 1], trade.size])
                else:
                    self.trade_volumes_sell[self.load_ts_num - 1] = np.nansum([self.trade_volumes_sell[self.load_ts_num - 1], trade.size])
            self.latest_trade_ts = trade.ts_event
            self.signal = True
        else:
            self.latest_trade_ts = None
            self.signal = False
                # 将最新的交易数据时间戳更新为当前交易数据时间戳

    def calc_feat_values(self, period, left, right, old_data=None):
        if old_data is None:
            _old_data = self.get_trade_data(right+1)
        else:
            _old_data = old_data
        _old_sum = 0.0
        _old_count = 0
        if not np.isnan(_old_data.bid_size):
            _old_sum += _old_data.bid_size
            _old_count += 1
        if not np.isnan(_old_data.ask_size):
            _old_sum += _old_data.ask_size
            _old_count += 1

        _new_index = self.get_data_index(left)
        _new_data = self.get_trade_data(left)
        _new_size = _new_data.bid_size
        if np.isnan(_new_size) or _new_size < _new_data.ask_size:
            _new_max_size = _new_data.ask_size
        else:
            _new_max_size = _new_size
        _new_max_data = DataInfo(_new_max_size, _new_data.ts_event)
        _new_sum = 0.0
        _new_count = 0
        if not np.isnan(_new_data.bid_size):
            _new_sum += _new_data.bid_size
            _new_count += 1
        if not np.isnan(_new_data.ask_size):
            _new_sum += _new_data.ask_size
            _new_count += 1

        if old_data is None:
            # 新数据，滚动
            if len(self.volume_down_list[period]) > 0 and self.counter - self.volume_down_list[period][0].ts >= right:
                self.volume_down_list[period].pop(0)
        else:
            # 更新当前数据
            self.volume_down_list[period].pop(-1)
        self.volume_down_list[period] = add_data(_new_max_data, self.volume_down_list[period], -1)
        
        _max = self.volume_down_list[period][0].data
        # print(f'vol cal: {_max} {period} {self.volume_down_list[period][0]}/{self.volume_down_list[period][-1]} {len(self.volume_down_list[period])} {self.load_ts_num}/{self.data_index}')
        self.volume_sum[period] += _new_sum - _old_sum
        _sum = self.volume_sum[period]
        self.trade_count[period] += _new_count - _old_count
        if self.trade_count[period] > 0:
            _mean = self.volume_sum[period] / self.trade_count[period]
        else:
            _mean = None
        return _sum, _mean, _max
        
    def calculate(self):
        '''
        计算自定义特征
        returns: [(fe_name, {'tp': int, 'ret':}), ...]
        '''
        # 在这里实现自定义特征的计算逻辑
        res = []
        if not self.signal:
            return None
        for i in range(len(self.left)):
            left, right = self.left[i], self.right[i]
            # print(f'vg cal: {left}, {right}, {self.load_ts_num} / {self.window_len} {self.last_data}')

            total_volume, avg_volume, max_volume = None, None, None
            if self.load_ts_num > left:
                if self.last_data is None:
                    total_volume, avg_volume, max_volume = self.calc_feat_values(i, left, right)
                else:
                    if left > 0 and self.load_ts_num >= right:
                        total_volume = self.volume_sum[i]
                        if self.trade_count[i] > 0:
                            avg_volume = self.volume_sum[i] / self.trade_count[i]
                        max_volume = self.volume_down_list[i][0].data
                    else:
                        total_volume, avg_volume, max_volume = self.calc_feat_values(i, left, right, self.last_data)

            # 返回特征结果
            if self.load_ts_num >= right:
                feature_data1 = (f'{self.fe_name1}_{left}_{right}', self.latest_trade_ts, total_volume)
                feature_data2 = (f'{self.fe_name2}_{left}_{right}', self.latest_trade_ts, avg_volume)
                feature_data3 = (f'{self.fe_name3}_{left}_{right}', self.latest_trade_ts, max_volume)
                res.append(feature_data1)
                res.append(feature_data2)
                res.append(feature_data3)
        # print(f'vg res: {len(res)}')
        self.last_data = self.get_trade_data(0)
        return res

    # def calculate(self):
    #     '''
    #     计算自定义特征
    #     returns: [(fe_name, {'tp': int, 'ret':}), ...]
    #     '''
    #     # 在这里实现自定义特征的计算逻辑
    #     res = []
    #     if not self.signal:
    #         return None
    #     for i in range(len(self.left)):
    #         left, right = self.left[i], self.right[i]
    #         # print(f'vg cal: {left}, {right}, {self.load_ts_num} / {self.window_len}')

    #         if (self.load_ts_num > left):
    #             if left != 0:
    #                 trade_volumes = np.append(self.trade_volumes_buy[-right:-left],self.trade_volumes_sell[-right:-left])
    #             else:
    #                 trade_volumes = np.append(self.trade_volumes_buy[-right:], self.trade_volumes_sell[-right:])
    #             # 计算 left 和 right 区间内的成交量之和
    #             total_volume, avg_volume, max_volume = np.nansum(trade_volumes), np.nanmean(trade_volumes), np.nanmax(trade_volumes)
    #         else:
    #             total_volume, avg_volume, max_volume = None, None, None

    #     # 返回特征结果
    #         feature_data1 = (f'{self.fe_name1}_{left}_{right}', self.latest_trade_ts, total_volume)
    #         feature_data2 = (f'{self.fe_name2}_{left}_{right}', self.latest_trade_ts, avg_volume)
    #         feature_data3 = (f'{self.fe_name3}_{left}_{right}', self.latest_trade_ts, max_volume)
    #         res.append(feature_data1)
    #         res.append(feature_data2)
    #         res.append(feature_data3)
    #     # print(f'vg res: {len(res)}')
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

    ins = VolmeGenrator(trans_config)
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
