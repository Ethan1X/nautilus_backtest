import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# from util.load_s3_data import LoadS3Data
from util.time_method import *
from data_generator import *
from typing import TypedDict
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
from util.statistic_method_v2 import add_data, DataInfo
from util.lookbackwnd import *


class TxnImbalance_lbw_Generator(FeatureGenerator):
    '''
    Author: Li Haotong
    Reviewer: Qiushi Bu
    Reviewer2: Wu Shaofan
    Feature: 14
    '''
    def __init__(self, config: FeatureConfig, lbw_type=LBW_TYPE_TIME) -> None:
        super().__init__(config)
        # FeatureConfig中包括left和right两个参数，回望时间为(T-right,T-left]
        self.data_mode = "normal"
        if "data_mode" in config.keys():
            self.data_mode = config['data_mode']

        self.left = config['left']
        self.right = config['right']
        self.min_max = (min(self.left), max(self.right))
        self.window_len = self.min_max[1]

        _lbw_str = LBW_TIME_STR
        if lbw_type == LBW_TYPE_VOLUME:
            _lbw_str = LBW_VOLUME_STR
        self.lbw_type = lbw_type
        self.fe_name1 = f'TxnImbalance{_lbw_str}'
        self.trade_lbw: LookBackWindow = None
        self.data_lbw = None
        
        self.latest_trade_ts = None
        self.last_trade_ts = None
        self.load_ts_num = 0
        self.trade_volumes_buy = np.zeros(self.window_len) * np.nan
        self.trade_volumes_sell = np.zeros(self.window_len) * np.nan
        self.last_trade = QuoteTick(0.0, 0.0, 0.0, 0.0, 0)
        self.signal = False

        _periods = len(self.right)
        self.data_index = 0
        self.last_data:QuoteTick = None
        self.volume_buy_sum = [0.0] * _periods
        self.volume_sell_sum = [0.0] * _periods

    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None, lbw_list=None):
        self.load_ts_num = load_sum
        
        if data_index != self.data_index:
            self.data_index = data_index

        if self.latest_trade_ts is None and price_list_buy is not None:
            self.trade_volumes_buy = volumes_list_buy
            self.trade_volumes_sell = volumes_list_sell
            self.trade_price_buy = price_list_buy
            self.trade_price_sell = price_list_sell
            self.window_len = len(self.trade_volumes_buy)
            if self.lbw_type in lbw_list:
                self.trade_lbw = lbw_list[self.lbw_type]
            self.data_lbw = {"buy":[0.0], "sell":[0.0], "all":[0.0]}

    def update_lbw_data(self, trade):
        if self.trade_lbw is not None:
            _idx = self.trade_lbw.rolling_wnd
            if self.trade_lbw.type == LBW_TYPE_TIME:
                _bid_size = trade.size if trade.aggressor_side == 'buy' else 0.0
                _ask_size = trade.size if trade.aggressor_side == 'sell' else 0.0
    
                while _idx > 0:
                    self.data_lbw["buy"].append(0.0)
                    self.data_lbw["sell"].append(0.0)
                    _idx -= 1
                    # print(f'update lbw data({self.trade_lbw.type}): {_idx} {len(self.data_lbw["buy"])}')
                self.data_lbw["buy"][-1] += _bid_size
                self.data_lbw["sell"][-1] += _ask_size
    
                if self.trade_lbw.is_rolling:
                    self.last_data = None
                else:
                    self.last_data = {"buy":self.data_lbw["buy"][-1], "sell":self.data_lbw["sell"][-1]}
                # print(f'lbw data updated({self.trade_lbw.type}): {self.trade_lbw.is_rolling} {self.trade_lbw.rolling_wnd} {self.last_data}')
            else:
                # todo: for volume，暂不可用
                # todo: 每档保存的订单量，并不对应于交易额，需重新计算。。。
                while _idx > 0:
                    self.data_lbw["all"].append(self.trade_lbw[-_idx-1][2]/trade.price)
                    _idx -= 1
                    # print(f'update lbw data({self.trade_lbw.type}): {_idx} {len(self.data_lbw["all"])}')
                self.data_lbw["all"][-1] = self.trade_lbw[-1][2]/trade.price
    
                if self.trade_lbw.is_rolling:
                    self.last_data = None
                else:
                    self.last_data = {"all":self.data_lbw["all"][-1]}
                print(f'lbw data updated({self.trade_lbw.type}): {self.trade_lbw.is_rolling} {self.trade_lbw.rolling_wnd} {self.last_data}')
        
    def rolling(self):
        # 与回望窗口同步rolling
        if self.trade_lbw.type == LBW_TYPE_TIME:
            _rolling_steps = len(self.data_lbw["buy"]) - len(self.trade_lbw.container) - 1
            if _rolling_steps > 0:
                del self.data_lbw["buy"][:_rolling_steps]
                del self.data_lbw["sell"][:_rolling_steps]
        else:    # todo: for volume, 暂不可用
            _rolling_steps = len(self.data_lbw["all"]) - len(self.trade_lbw.container) - 1
            if _rolling_steps > 0:
                del self.data_lbw["all"][:_rolling_steps]

    def get_data_index(self, shift):
        return (self.window_len + self.data_index - shift) % self.window_len

    def get_trade_data(self, shift, data=None):
        _data = data

        _data_index = self.get_data_index(shift)
        if _data is not None:
            _data.ask_price = self.trade_price_sell[_data_index]
            _data.ask_size = self.trade_volumes_sell[_data_index]
            _data.bid_price = self.trade_price_buy[_data_index]
            _data.bid_size = self.trade_volumes_buy[_data_index]
        else:
            _data = QuoteTick(self.trade_price_sell[_data_index], self.trade_volumes_sell[_data_index],
                              self.trade_price_buy[_data_index], self.trade_volumes_buy[_data_index], 
                              0)
        return _data

    def process(self, ticker: QuoteTick = None, trade: TradeTick = None, depth: OrderBook = None):
        '''
            feature计算主流程 供外部调用
        '''
        # if self.data_mode == "central":
        # 只支持central模式
        self.update_signal(trade)
        feature_ret = self.calculate()

        return feature_ret

    def update_signal(self, trade: TradeTick=None):
        if trade is not None:
            self.update_lbw_data(trade)
            self.latest_trade_ts = trade.ts_event
            self.signal = True
        else:
            self.latest_trade_ts = None
            self.signal = False

    def calc_feat_values(self, period, left, right, old_data=None):
        if self.trade_lbw.type == LBW_TYPE_TIME:
            return self._calc_feat_values_t(period, left, right, old_data)
        else:
            return self._calc_feat_values_v(period, left, right, old_data)
        
    def _calc_feat_values_t(self, period, left, right, old_data=None):
        if old_data is None:
            # _old_data = self.get_trade_data(right+1)
            if len(self.data_lbw) > right:    # self.trade_lbw.size() > right:
                _rolling_wnd = self.trade_lbw.rolling_wnd
                _old_data = {"buy":self.data_lbw["buy"][-right-_rolling_wnd], "sell":self.data_lbw["sell"][-right-_rolling_wnd]}
                _rolling_wnd -= 1
                while _rolling_wnd > 0:
                    _old_data["buy"] += self.data_lbw["buy"][-right-_rolling_wnd]
                    _old_data["sell"] += self.data_lbw["sell"][-right-_rolling_wnd]
                    _rolling_wnd -= 1
            else:
                _old_data = {"buy":0.0, "sell":0.0}
        else:
            _old_data = old_data
        # _new_data = self.get_trade_data(left)
        # _delta_buy = _new_data.bid_size
        _left = -left if left > 0 else -1
        _new_data = {"buy":self.data_lbw["buy"][_left], "sell":self.data_lbw["sell"][_left]}
        _rolling_wnd = self.trade_lbw.rolling_wnd
        if _rolling_wnd > 0:
            _rolling_wnd -= 1
            while _rolling_wnd > 0:
                _new_data["buy"] += self.data_lbw["buy"][_left-_rolling_wnd]
                _new_data["sell"] += self.data_lbw["sell"][_left-_rolling_wnd]
                _rolling_wnd -= 1
        
        _delta_buy = _new_data["buy"]
        _delta_buy -= _old_data["buy"]
        _sum_buy = self.volume_buy_sum[period] + _delta_buy
        self.volume_buy_sum[period] = _sum_buy
        _delta_sell = _new_data["sell"]
        _delta_sell -= _old_data["sell"]
        _sum_sell = self.volume_sell_sum[period] + _delta_sell
        self.volume_sell_sum[period] = _sum_sell
        return _sum_buy, _sum_sell

    def _calc_feat_values_v(self, period, left, right, old_data=None):
        if old_data is None:
            # _old_data = self.get_trade_data(right+1)
            if len(self.data_lbw) > right:    # self.trade_lbw.size() > right:
                _rolling_wnd = self.trade_lbw.rolling_wnd
                _old_data = {
                                "buy":self.data_lbw["buy"][-right-_rolling_wnd],
                                "sell":self.data_lbw["sell"][-right-_rolling_wnd],
                                "all":self.data_lbw["all"][-right-_rolling_wnd]
                            }
                _rolling_wnd -= 1
                while _rolling_wnd > 0:
                    _old_data["buy"] += self.data_lbw["buy"][-right-_rolling_wnd]
                    _old_data["sell"] += self.data_lbw["sell"][-right-_rolling_wnd]
                    _old_data["all"] += self.data_lbw["all"][-right-_rolling_wnd]
                    _rolling_wnd -= 1
            else:
                _old_data = {"buy":0.0, "sell":0.0, "all":0.0}
        else:
            _old_data = old_data
        # _new_data = self.get_trade_data(left)
        # _delta_buy = _new_data.bid_size
        _left = -left if left > 0 else -1
        _new_data = {"buy":self.data_lbw["buy"][_left], "sell":self.data_lbw["sell"][_left], "all":self.data_lbw["all"][_left]}
        _rolling_wnd = self.trade_lbw.rolling_wnd
        if _rolling_wnd > 0:
            _rolling_wnd -= 1
            while _rolling_wnd > 0:
                _new_data["buy"] += self.data_lbw["buy"][_left-_rolling_wnd]
                _new_data["sell"] += self.data_lbw["sell"][_left-_rolling_wnd]
                _new_data["all"] += self.data_lbw["all"][_left-_rolling_wnd]
                _rolling_wnd -= 1
        
        _delta_buy = _new_data["buy"]
        _delta_buy -= _old_data["buy"]
        _sum_buy = self.volume_buy_sum[period] + _delta_buy
        self.volume_buy_sum[period] = _sum_buy
        _delta_sell = _new_data["sell"]
        _delta_sell -= _old_data["sell"]
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
            _lbw_size = self.trade_lbw.size()

            # if not self.trade_lbw.is_rolling:
            # print(f'right: {right}, left: {left}, {len(self.data_lbw["buy"])}; {_lbw_size}, {self.trade_lbw.get_attr()}')
            # if self.load_ts_num > left:
            if _lbw_size > left:
                # todo
                # if self.last_data is None:
                if self.trade_lbw.is_rolling:
                    # 新数据，窗口滑动self.trade_lbw.rolling_wnd
                    _sum_buy, _sum_sell = self.calc_feat_values(i, left, right)
                else:
                    # 仅更新当前数据
                    # if left > 0 and self.load_ts_num >= right:
                    if left > 0 and _lbw_size >= right:
                        _sum_buy = self.volume_buy_sum[i]
                        _sum_sell = self.volume_sell_sum[i]
                        # print(f'odd: {_lbw_size} {self.trade_lbw.is_rolling}')
                    else:
                        _sum_buy, _sum_sell = self.calc_feat_values(i, left, right, self.last_data)
                if _sum_buy + _sum_sell > 0:
                    txn_imbalance = (_sum_buy - _sum_sell) / (_sum_buy + _sum_sell)
            # if self.load_ts_num >= right:
            if _lbw_size >= right:
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
