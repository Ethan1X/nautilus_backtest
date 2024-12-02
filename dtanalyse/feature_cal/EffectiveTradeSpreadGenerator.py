import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# from util.load_s3_data import LoadS3Data
from util.time_method import *
from data_generator import *
from typing import TypedDict
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog

class EffectiveTradeSpreadGenerator(FeatureGenerator):
    '''
    Author: Li Haotong
    Reviewer: Qiushi Bu
    Feature: 18
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
        self.fe_name1 = f'EffectiveTradeSpread'

        self.trade_volumes_buy = np.zeros(self.window_len) * np.nan
        self.trade_price_buy = np.zeros(self.window_len) * np.nan
        self.trade_volumes_sell = np.zeros(self.window_len) * np.nan
        self.trade_price_sell = np.zeros(self.window_len) * np.nan
        self.prices = self.net_values = self.total_values = self.weighted_values = None
        self.load_ts_num = 0
        self.data_index = 0
        self.last_data:QuoteTick = None
        self.latest_trade_ts = None
        self.last_trade_ts = None
        self.signal = False

    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None):
        self.load_ts_num = load_sum
        if data_index != self.data_index:
            self.last_data = None
            self.data_index = data_index
 
        # print(f'=========={self.data_index}, {self.window_len}, {self.latest_trade_ts}, {price_list_buy}')
        if self.latest_trade_ts is None and price_list_buy is not None:
            self.trade_volumes_buy = volumes_list_buy
            self.trade_volumes_sell = volumes_list_sell
            self.trade_price_buy = price_list_buy
            self.trade_price_sell = price_list_sell
            self.window_len = len(self.trade_price_buy)
            self.prices = [0.0] * self.window_len
            self.total_values = [[]] * self.window_len
            self.net_values = [[]] * self.window_len
            self.weighted_values = [[]] * self.window_len
            for i in range(len(self.left)):
                self.total_values[i] = [0.0] * self.window_len
                self.net_values[i] = [0.0] * self.window_len
                self.weighted_values[i] = [0.0] * self.window_len

        if self.latest_trade_ts is not None:
            self.prices[self.data_index] = self.trade_price_buy[self.data_index]
            if np.isnan(self.prices[self.data_index]) or self.prices[self.data_index] == 0:
                self.prices[self.data_index] = self.trade_price_sell[self.data_index]

    def get_data_index(self, shift):
        return (self.window_len + self.data_index - shift) % self.window_len

    def get_trade_data(self, shift):
        _data_index = self.get_data_index(shift)
        _data = QuoteTick(self.trade_price_sell[_data_index], self.trade_volumes_sell[_data_index],
                          self.trade_price_buy[_data_index], self.trade_volumes_buy[_data_index], 0)
        return _data

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

    def update(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
        '''
        更新自定义特征的内部状态
        '''
        # 检查最新的交易数据时间戳是否更新，如果更新则清空交易量列表
        if trade is not None:
            if not self.last_trade_ts or trade.ts_event // 1000 != self.last_trade_ts // 1000:
                self.signal = True
                self.last_trade_ts = trade.ts_event
                # 在这里实现自定义特征的更新逻辑
                if self.load_ts_num == self.min_max[1]:
                    self.trade_volumes_buy[:-1] = self.trade_volumes_buy[1:]
                    self.trade_volumes_sell[:-1] = self.trade_volumes_sell[1:]
                    self.trade_price_buy[:-1] = self.trade_price_buy[1:]
                    self.trade_price_sell[:-1] = self.trade_price_sell[1:]
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

    def calc_feat_values(self, price_list, volumes_list, shift, old_data=None, is_buy=True):
        if old_data is None:
            _data_index = self.get_data_index(shift+1)
        else:
            _data_index = None
        if _data_index is None:
            _price = old_data.bid_price if is_buy else old_data.ask_price
            _size = old_data.bid_size if is_buy else old_data.ask_size
        else:
            _price = price_list[_data_index]
            _size = volumes_list[_data_index]
        _sign = 1 if is_buy else -1
        _value = _price * _size
        _weighted_value = np.log(_price) * _sign * _value
        return _weighted_value, _value, _value * _sign

    def calc_feat_delta(self, shift, old_data=None):
        _weighted_value_buy, _value_buy, _net_value_buy = self.calc_feat_values(self.trade_price_buy, self.trade_volumes_buy, shift, old_data, True)
        _weighted_value_sell, _value_sell, _net_value_sell = self.calc_feat_values(self.trade_price_sell, self.trade_volumes_sell, shift, old_data, False)

        _weighted_value_sum = 0.0
        _value_sum = 0.0
        _net_value_sum = 0.0
        if not np.isnan(_weighted_value_sell):
            _weighted_value_sum += _weighted_value_sell
            _value_sum += _value_sell
            _net_value_sum += _net_value_sell
        if not np.isnan(_weighted_value_buy):
            _weighted_value_sum += _weighted_value_buy
            _value_sum += _value_buy
            _net_value_sum += _net_value_buy
        return _weighted_value_sum, _value_sum, _net_value_sum

    def calculate(self):
    #     '''
    #     计算自定义特征
    #     returns: [(fe_name, {'tp': int, 'ret':}), ...]
    #     '''
    #     # 在这里实现自定义特征的计算逻辑
        if not self.signal:
            return None
        res = []
        for i in range(len(self.left)):
            left, right = self.left[i], self.right[i]
            if self.load_ts_num > left:
                # effective_spread = np.nansum(np.log(a/a[-1])*b*c*a)/np.nansum(a*c)
                _cur_index = self.get_data_index(left)
                _cur_price = self.prices[_cur_index]
                if self.last_data is None:
                    # 新数据，窗口滑动一格
                    _last_index = self.get_data_index(left + 1)
                    _old_weighted, _old_value, _old_net_value = self.calc_feat_delta(right)
                else:
                    # 仅更新当前数据
                    if left > 0:
                        continue
                    _last_index = _cur_index
                    _old_weighted, _old_value, _old_net_value = self.calc_feat_delta(left, self.last_data)
                _new_weighted, _new_value, _new_net_value = self.calc_feat_delta(left)
                
                _weighted_sum = self.weighted_values[i][_last_index] - _old_weighted + _new_weighted
                _value_sum = self.total_values[i][_last_index] - _old_value + _new_value
                _net_value_sum = self.net_values[i][_last_index] - _old_net_value + _new_net_value
                self.weighted_values[i][_cur_index] = _weighted_sum
                self.total_values[i][_cur_index] = _value_sum
                self.net_values[i][_cur_index] = _net_value_sum
                if self.load_ts_num >= right:
                    effective_spread = (_weighted_sum - _net_value_sum * np.log(_cur_price)) / _value_sum
                    res.append((f'{self.fe_name1}_{left}_{right}', self.latest_trade_ts, effective_spread))
        self.last_data = self.get_trade_data(0)
        return res
        
    # def calculate(self):
    #     '''
    #     计算自定义特征
    #     returns: [(fe_name, {'tp': int, 'ret':}), ...]
    #     '''
    #     # 在这里实现自定义特征的计算逻辑
    #     if not self.signal:
    #         return None
    #     res = []
    #     for k in range(len(self.left)):
    #         left, right = self.left[k], self.right[k]
    #         if self.load_ts_num > left:
    #             len_ = right - left
    #             if left != 0:
    #                 a, b, c = np.ones(len_*2), np.ones(len_*2), np.ones(len_*2)
    #                 for i in range(len_):
    #                     a[i * 2] = self.trade_price_buy[-right + i]
    #                     a[i * 2 + 1] = self.trade_price_sell[-right + i]
    #                     b[i * 2] = 1 if np.isnan(self.trade_price_sell[-right + i]) else -1
    #                     b[i * 2 + 1] = -1 if np.isnan(self.trade_price_buy[-right + i]) else 1
    #                     c[i * 2] = self.trade_volumes_buy[-right + i]
    #                     c[i * 2 + 1] = self.trade_volumes_sell[-right + i]
    #             else:
    #                 a, b, c = np.ones(len_*2), np.ones(len_*2), np.ones(len_*2)
    #                 for i in range(len_):
    #                     a[i * 2] = self.trade_price_buy[-right + i]
    #                     a[i * 2 + 1] = self.trade_price_sell[-right + i]
    #                     b[i * 2] = 1 if np.isnan(self.trade_price_sell[-right + i]) else -1
    #                     b[i * 2 + 1] = -1 if np.isnan(self.trade_price_buy[-right + i]) else 1
    #                     c[i * 2] = self.trade_volumes_buy[-right + i]
    #                     c[i * 2 + 1] = self.trade_volumes_sell[-right + i]
    #             effective_spread = np.nansum(np.log(a/a[-1])*b*c*a)/np.nansum(a*c)
    #             res.append((f'{self.fe_name1}_{left}_{right}', self.latest_trade_ts, effective_spread))
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

    ins = EffectiveTradeSpreadGenerator(trans_config)
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
