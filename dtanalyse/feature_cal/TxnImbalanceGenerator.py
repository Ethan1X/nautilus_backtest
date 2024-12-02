import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# from util.load_s3_data import LoadS3Data
from util.time_method import *
from data_generator import *
from typing import TypedDict
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
from util.statistic_method_v2 import add_data, DataInfo


class TxnImbalanceGenerator(FeatureGenerator):
    '''
    Author: Li Haotong
    Reviewer: Qiushi Bu
    Reviewer2: Wu Shaofan
    Feature: 14
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
        self.fe_name1 = f'TxnImbalance'
        self.latest_trade_ts = None
        self.last_trade_ts = None
        self.load_ts_num = 0
        self.trade_volumes_buy = np.zeros(self.window_len) * np.nan
        self.trade_volumes_sell = np.zeros(self.window_len) * np.nan
        self.signal = False

        _periods = len(self.right)
        self.data_index = 0
        self.last_data:QuoteTick = None
        self.volume_buy_sum = [0.0] * _periods
        self.volume_sell_sum = [0.0] * _periods

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

    def get_data_index(self, shift):
        return (self.window_len + self.data_index - shift) % self.window_len

    def get_trade_data(self, shift):
        _data_index = self.get_data_index(shift)
        _data = QuoteTick(self.trade_price_sell[_data_index], self.trade_volumes_sell[_data_index],
                          self.trade_price_buy[_data_index], self.trade_volumes_buy[_data_index], 
                          0)
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
            if self.last_trade_ts is None or ((trade.ts_event // 1000) != (self.last_trade_ts // 1000)):
                self.last_trade_ts = trade.ts_event
                # 在这里实现自定义特征的更新逻辑
                if self.load_ts_num == self.window_len:
                    self.trade_volumes_buy[:-1] = self.trade_volumes_buy[1:]
                    self.trade_volumes_sell[:-1] = self.trade_volumes_sell[1:]
                    if trade.aggressor_side == 'buy':
                        self.trade_volumes_buy[self.load_ts_num - 1] = trade.size
                        self.trade_volumes_sell[self.load_ts_num - 1] = np.nan
                    else:
                        self.trade_volumes_sell[self.load_ts_num - 1] = trade.size
                        self.trade_volumes_buy[self.load_ts_num - 1] = np.nan
                else:
                    self.load_ts_num = self.load_ts_num + 1
                    if trade.aggressor_side == 'buy':
                        self.trade_volumes_buy[self.load_ts_num - 1] = trade.size
                    else:
                        self.trade_volumes_sell[self.load_ts_num - 1] = trade.size
            else:
                # 如果是相同时刻的trade，直接更新当前时刻
                if trade.aggressor_side == 'buy':
                    self.trade_volumes_buy[self.load_ts_num - 1] = np.nansum(
                        [self.trade_volumes_buy[self.load_ts_num - 1], trade.size])
                else:
                    self.trade_volumes_sell[self.load_ts_num - 1] = np.nansum(
                        [self.trade_volumes_sell[self.load_ts_num - 1], trade.size])
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
        _new_data = self.get_trade_data(left)
        _delta_buy = _new_data.bid_size
        if np.isnan(_delta_buy):
            _delta_buy = 0
        if not np.isnan(_old_data.bid_size):
            _delta_buy -= _old_data.bid_size
        _sum_buy = self.volume_buy_sum[period] + _delta_buy
        self.volume_buy_sum[period] = _sum_buy
        _delta_sell = _new_data.ask_size
        if np.isnan(_delta_sell):
            _delta_sell = 0
        if not np.isnan(_old_data.ask_size):
            _delta_buy -= _old_data.ask_size
        _sum_sell = self.volume_sell_sum[period] + _delta_sell
        self.volume_sell_sum[period] = _sum_sell
        return _sum_buy, _sum_sell

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

            txn_imbalance = None
            _sum_buy = _sum_sell = 0.0
            if self.load_ts_num > left:
                # todo
                if self.last_data is None:
                    # 新数据，窗口滑动一格
                    _sum_buy, _sum_sell = self.calc_feat_values(i, left, right)
                else:
                    # 仅更新当前数据
                    if left > 0 and self.load_ts_num >= right:
                        _sum_buy = self.volume_buy_sum[i]
                        _sum_sell = self.volume_sell_sum[i]
                    else:
                        _sum_buy, _sum_sell = self.calc_feat_values(i, left, right, self.last_data)
                if _sum_buy + _sum_sell > 0:
                    txn_imbalance = (_sum_buy - _sum_sell) / (_sum_buy + _sum_sell)
            if self.load_ts_num >= right:
                res.append((f'{self.fe_name1}_{left}_{right}', self.latest_trade_ts, txn_imbalance))
        # 返回特征结果
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
    
    #         if self.load_ts_num > left:
    #             if left != 0:
    #                 txn_imbalance = (np.nansum(self.trade_volumes_buy[-right:-left])- np.nansum(self.trade_volumes_sell[-right:-left]))/(np.nansum(self.trade_volumes_buy[-right:-left])+ np.nansum(self.trade_volumes_sell[-right:-left]))
    #             else:
    #                 txn_imbalance = (np.nansum(self.trade_volumes_buy[-right:]) - np.nansum(self.trade_volumes_sell[-right:])) / (np.nansum(self.trade_volumes_buy[-right:]) + np.nansum(
    #                                         self.trade_volumes_sell[-right:]))
    #         else:
    #             txn_imbalance = None

    #         res.append((f'{self.fe_name1}_{left}_{right}', self.latest_trade_ts, txn_imbalance))
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

    ins = TxnImbalanceGenerator(trans_config)
    fe_list = []

    for idx, row in enumerate(get_data_generator(begin_time, end_time, exchange, symbol)):
        # print(idx, row)

        
        if row[1] == 'ticker':
            fe_list.append(ins.process(ticker=row[0]))
        elif row[1] == 'depth':
            fe_list.append(ins.process(depth=row[0]))
        else:
            fe_list.append(ins.process(trade=row[0]))
            print(idx, row)
            # print(fe_list[-1])

    print(time.time()-start)
