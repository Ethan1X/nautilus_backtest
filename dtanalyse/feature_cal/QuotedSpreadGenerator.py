import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# from util.load_s3_data import LoadS3Data
from util.time_method import *
from data_generator import *
from typing import TypedDict
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog

class QuotedSpreadGenerator(FeatureGenerator):
    '''
    Author: Li Haotong
    Reviewer: Qiushi Bu
    Feature: 17
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
        self.periods = len(self.right)
        self.fe_name1 = f'QuotedSpread'
        self.latest_trade_ts = None
        self.last_trade_ts = None
        self.load_ts_num = 0
        self.last_data = None
        self.data_index = None
        self.trade_volumes_buy = np.zeros(self.window_len) * np.nan
        self.trade_price_buy = np.zeros(self.window_len) * np.nan
        self.trade_volumes_sell = np.zeros(self.window_len) * np.nan
        self.trade_price_sell = np.zeros(self.window_len) * np.nan
        self.tick_spread = None
        self.trade_index = 0
        self.ratio_buy = np.zeros(self.window_len+1)
        self.ratio_sell = np.zeros(self.window_len+1)
        self.ratio_sum = [0.0] * self.periods
        self.ratio_count = [0] * self.periods
        self.signal = False

    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None):
        self.load_ts_num = load_sum
        if self.data_index is not None:
            if data_index != self.data_index:
                self.data_index = data_index
                self.last_data = None

        if self.latest_trade_ts is None and price_list_buy is not None:
            self.trade_volumes_buy = volumes_list_buy
            self.trade_volumes_sell = volumes_list_sell
            self.trade_price_buy = price_list_buy
            self.trade_price_sell = price_list_sell
            self.window_len = len(self.trade_price_buy)
            self.ratio_buy = np.zeros(self.window_len)
            self.ratio_sell = np.zeros(self.window_len)
            self.data_index = 0

    def get_data_index(self, shift, index):
        return (self.window_len + index - shift) % self.window_len

    def get_trade_data(self, shift):
        _data_index = self.get_data_index(shift, self.data_index)
        _data = QuoteTick(self.trade_price_sell[_data_index], self.trade_volumes_sell[_data_index],
                          self.trade_price_buy[_data_index], self.trade_volumes_buy[_data_index], 0)
        return _data

    def get_ratio_data(self, shift):
        _data_index = self.get_data_index(shift, self.trade_index)
        return self.ratio_buy[_data_index], self.ratio_sell[_data_index]       

    def process(self, ticker: QuoteTick = None, trade: TradeTick = None, depth: OrderBook = None):
        '''
            feature计算主流程 供外部调用
        '''
        if self.data_mode == "central":
            self.update_signal(trade=trade, ticker=ticker)
        else:
            self.update(ticker, trade, depth)

        feature_ret = None
        if trade is not None and self.data_index is not None:
            feature_ret = self.calculate()

        return feature_ret

    def update_signal(self, trade: TradeTick=None, ticker: QuoteTick=None):
        if ticker is not None:
            self.tick_spread = ticker.ask_price - ticker.bid_price
        if trade is not None:
            self.latest_trade_ts = trade.ts_event
            if self.last_data is None:
                self.trade_index = self.get_data_index(-1, self.trade_index)
            if self.tick_spread is not None:
                if trade.aggressor_side == 1:
                    if not np.isnan(self.trade_price_buy[self.data_index]):
                        self.ratio_buy[self.trade_index] = self.tick_spread / self.trade_price_buy[self.data_index]
                else:
                    if not np.isnan(self.trade_price_sell[self.data_index]):
                        self.ratio_sell[self.trade_index] = self.tick_spread / self.trade_price_sell[self.data_index]
            else:
                self.ratio_buy[self.trade_index] = 0.0
                self.ratio_sell[self.trade_index] = 0.0
            self.signal = True
        else:
            self.latest_trade_ts = None
            self.signal = False
        # print(f'update signal: {ticker} {trade} {self.data_index} {self.load_ts_num} {self.signal}')

    def update(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
        '''
        更新自定义特征的内部状态
        '''
        # 检查最新的交易数据时间戳是否更新，如果更新则清空交易量列表
        if ticker is not None:
            self.tick_spread[self.load_ts_num] = ticker.ask_price - ticker.bid_price
        if trade is not None:
            self.signal = True
            if self.last_trade_ts is None or (trade.ts_event // 1000 != self.last_trade_ts // 1000):
                self.last_trade_ts = trade.ts_event
                if self.load_ts_num == self.window_len:
                    self.tick_spread[0:-1] = self.tick_spread[1:]
                    self.trade_volumes_buy[:-1] = self.trade_volumes_buy[1:]
                    self.trade_volumes_sell[:-1] = self.trade_volumes_sell[1:]
                    self.trade_price_buy[:-1] = self.trade_price_buy[1:]
                    self.trade_price_sell[:-1] = self.trade_price_sell[1:]
                    # print(len(self.tick_spread), len(self.trade_price_buy))
                    if trade.aggressor_side == 'buy':
                        self.trade_volumes_buy[self.load_ts_num - 1] = trade.size
                        self.trade_price_buy[self.load_ts_num - 1] = trade.price
                        self.trade_volumes_sell[self.load_ts_num - 1] = np.nan
                        self.trade_price_sell[self.load_ts_num - 1] = np.nan
                    else:
                        self.trade_volumes_sell[self.load_ts_num - 1] = trade.size
                        self.trade_price_sell[self.load_ts_num - 1] = trade.price
                        self.trade_volumes_buy[self.load_ts_num - 1] = np.nan
                        self.trade_price_buy[self.load_ts_num - 1] = np.nan
                else:
                    self.load_ts_num = self.load_ts_num + 1
                    if trade.aggressor_side == 'buy':
                        self.trade_volumes_buy[self.load_ts_num - 1] = trade.size
                        self.trade_price_buy[self.load_ts_num - 1] = trade.price
                    else:
                        self.trade_volumes_sell[self.load_ts_num - 1] = trade.size
                        self.trade_price_sell[self.load_ts_num - 1] = trade.price
            else:
                # 如果是相同时刻的trade，直接更新当前时刻
                if trade.aggressor_side == 'buy':
                    self.trade_price_buy[self.load_ts_num - 1] = np.nansum([(self.trade_price_buy[self.load_ts_num - 1] \
                                                                             * self.trade_volumes_buy[
                                                                                 self.load_ts_num - 1]),
                                                                            trade.size * trade.price]) / np.nansum([
                                                                     trade.size, self.trade_volumes_buy[
                                                                         self.load_ts_num - 1]])
                    self.trade_volumes_buy[self.load_ts_num - 1] = np.nansum(
                        [self.trade_volumes_buy[self.load_ts_num - 1], trade.size])
                else:
                    self.trade_price_sell[self.load_ts_num - 1] = np.nansum(
                        [(self.trade_price_sell[self.load_ts_num - 1] \
                          * self.trade_volumes_sell[self.load_ts_num - 1]), trade.size * trade.price]) / np.nansum(
                        [trade.size, self.trade_volumes_sell[self.load_ts_num - 1]])
                    self.trade_volumes_sell[self.load_ts_num - 1] = np.nansum(
                        [self.trade_volumes_sell[self.load_ts_num - 1], trade.size])
            self.latest_trade_ts = trade.ts_event
        else:
            self.latest_trade_ts = None
            self.signal = False
                # 将最新的交易数据时间戳更新为当前交易数据时间戳

    def calc_feat_values(self, period, left, right, last_data=None):
        _new_data_buy, _new_data_sell = self.get_ratio_data(left)
        if last_data is None:
            _old_data_buy, _old_data_sell = self.get_ratio_data(right+1)
        else:
            _old_data_buy, _old_data_sell = self.last_data
    
        _ratio_delta = 0
        _count_delta = 0
        if _new_data_buy != 0:
            _ratio_delta += _new_data_buy
            _count_delta += 1
        if _new_data_sell != 0:
            _ratio_delta += _new_data_sell
            _count_delta += 1
        if _old_data_buy != 0:
            _ratio_delta -= _old_data_buy
            _count_delta -= 1
        if _old_data_sell != 0:
            _ratio_delta -= _old_data_sell
            _count_delta -= 1
        
        _ratio_sum = self.ratio_sum[period] + _ratio_delta
        _count = self.ratio_count[period] + _count_delta
        if _count > 0:
            _ratio_mean = self.ratio_sum[period] / _count
        else:
            _ratio_mean = None
            
        self.ratio_sum[period] = _ratio_sum
        self.ratio_count[period] = _count
        # print(f'qs calc values: {period} {left} {right} {self.data_index} {_count}/{_count_delta} {_ratio_mean}')
        return _ratio_mean
    
    def calculate(self):
        '''
        计算自定义特征
        returns: [(fe_name, {'tp': int, 'ret':}), ...]
        '''
        res = []
        if not self.signal:
            return None
            
        # 在这里实现自定义特征的计算逻辑
        for i in range(self.periods):
            left, right = self.left[i], self.right[i]
            # print(f'calc: {self.load_ts_num} {i} {left}-{right} ')
            # print(f'qs calc: {self.data_index} {self.load_ts_num} {self.signal} {self.periods} {left} {right}')

            quoted_spread = None
            if self.load_ts_num > left:
                if self.last_data is None:
                    # 新数据，窗口滑动一格
                    quoted_spread = self.calc_feat_values(i, left, right)
                else:
                    # 仅更新当前数据
                    if left > 0:
                        if self.ratio_count[i] > 0:
                            quoted_spread = self.ratio_sum[i] / self.ratio_count[i]
                    else:
                        quoted_spread = self.calc_feat_values(i, left, right, self.last_data)

            # if right >= 7200:
                # print(f'qs calc: {self.load_ts_num} {left} {right} {self.data_index} {quoted_spread} {self.last_data}')
            if self.load_ts_num >= right:
                res_data = (f'{self.fe_name1}_{left}_{right}', self.latest_trade_ts, quoted_spread)
                # print(res_data)
                res.append(res_data)

        # 返回特征结果
        self.last_data = self.get_ratio_data(0)
        return res
        
    # def calculate(self):
    #     '''
    #     计算自定义特征
    #     returns: [(fe_name, {'tp': int, 'ret':}), ...]
    #     '''
    #     res = []
    #     if not self.signal:
    #         return None
            
    #     # 在这里实现自定义特征的计算逻辑
    #     for i in range(len(self.left)):
    #         left, right = self.left[i], self.right[i]

    #         if self.load_ts_num > left:
    #             if left != 0:
    #                 a = self.trade_price_buy[-right:-left]
    #                 b = self.trade_price_sell[-right:-left]
    #                 c = self.tick_spread[-right+1:-left+1]
    #             else:
    #                 a = self.trade_price_buy[:]
    #                 b = self.trade_price_sell[:]
    #                 c = self.tick_spread[1:]
    #                 # print(a, b, c)
    #             quoted_spread = np.nanmean(np.append(c/a, c/b))
    #         else:
    #             quoted_spread = None
    #         res_data = (f'{self.fe_name1}_{left}_{right}', self.latest_trade_ts, quoted_spread)
    #         # print(res_data)
    #         res.append(res_data)

    #     # 返回特征结果
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

    ins = QuotedSpreadGenerator(trans_config)
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
