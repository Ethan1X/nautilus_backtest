import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# from util.load_s3_data import LoadS3Data
from util.time_method import *
from data_generator import *
from typing import TypedDict
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog

class AutoCovGenerator(FeatureGenerator):
    '''
    Author: Li Haotong
    Reviewer: Qiushi Bu
    Feature: 16
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
        self.window_len = self.min_max[1] + 10
        self.period = len(self.left)
        self.auto_cov = [0.0] * self.period
        self.cov_count = [0] * self.period
        
        self.fe_name1 = f'AutoCov'
        self.trade_volumes_buy = np.zeros(self.window_len) * np.nan
        self.trade_price_buy = np.zeros(self.window_len) * np.nan
        self.trade_volumes_sell = np.zeros(self.window_len) * np.nan
        self.trade_price_sell = np.zeros(self.window_len) * np.nan
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

        if self.latest_trade_ts is None and price_list_buy is not None:
            self.trade_volumes_buy = volumes_list_buy
            self.trade_volumes_sell = volumes_list_sell
            self.trade_price_buy = price_list_buy
            self.trade_price_sell = price_list_sell
            self.window_len = len(self.trade_price_buy)

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

    def update(self, ticker: QuoteTick = None, trade: TradeTick = None, depth: OrderBook = None):
        '''
        更新自定义特征的内部状态
        '''
        # 检查最新的交易数据时间戳是否更新，如果更新则清空交易量列表
        if trade is not None:
            if not self.last_trade_ts or trade.ts_event // 1000 != self.last_trade_ts // 1000:
                self.signal = True
                self.last_trade_ts = trade.ts_event
                # 在这里实现自定义特征的更新逻辑
                # 如果成交量列表中数据量超过 right，则删除最早的数据
                if self.load_ts_num == self.window_len:
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

    def calc_cov(self, shift, old_data=None):
        _shift = shift
        if old_data is None:
            _shift += 1
            _data_index_a = self.get_data_index(_shift)
        else:
            _data_index_a = None
        _data_index_b = self.get_data_index(_shift+1)
        _data_index_c = self.get_data_index(_shift+2)
        if _data_index_a is None:
            _price_a = old_data.bid_price
        else:
            _price_a = self.trade_price_buy[_data_index_a]
        _price_b = self.trade_price_buy[_data_index_b]
        _price_c = self.trade_price_buy[_data_index_c]
        if _price_b != 0 and _price_c != 0:
            _cov_buy = np.log(_price_a/_price_b) * np.log(_price_b/_price_c)
        else:
            _cov_buy = np.nan

        if _data_index_a is None:
            _price_a = old_data.ask_price
        else:
            _price_a = self.trade_price_sell[_data_index_a]
        _price_b = self.trade_price_sell[_data_index_b]
        _price_c = self.trade_price_sell[_data_index_c]
        if _price_b != 0 and _price_c != 0:
            _cov_sell = np.log(_price_a/_price_b) * np.log(_price_b/_price_c)
        else:
            _cov_sell = np.nan

        _cov_sum = 0.0
        _cov_count = 0
        if not np.isnan(_cov_buy):
            _cov_sum += _cov_buy
            _cov_count += 1
        if not np.isnan(_cov_sell):
            _cov_sum += _cov_sell
            _cov_count += 1
        # print(f'calc cov: {_cov_buy}{type(_cov_buy)} {_cov_sell}{type(_cov_sell)} {_cov_count} {_cov_sum}')
        return _cov_sum, _cov_count

    def calculate(self):
        '''
        计算自定义特征
        returns: [(fe_name, {'tp': int, 'ret':}), ...]
        '''
        # 在这里实现自定义特征的计算逻辑
        res = []
        if not self.signal:
            return None
        for i in range(self.period):
            left, right = self.left[i], self.right[i]
            auto_cov = None
            _old_cov_sum, _old_cov_count = 0.0, 0
            if (self.load_ts_num > left+2):
                if self.last_data is None:
                    # 新数据，窗口滑动一格
                    _old_cov_sum, _old_cov_count = self.calc_cov(right)
                    # print(f'{self.fe_name1}_{left}_{right}, {self.latest_trade_ts}, {self.last_data}')
                else:
                    # 仅更新当前数据
                    if left > 0:
                        continue
                    _old_cov_sum, _old_cov_count = self.calc_cov(left, self.last_data)
                _new_cov_sum, _new_cov_count = self.calc_cov(left)
                _cov_sum = self.auto_cov[i] * self.cov_count[i] - _old_cov_sum + _new_cov_sum
                _cov_count = self.cov_count[i] - _old_cov_count + _new_cov_count
                self.cov_count[i] = _cov_count
                if _cov_count > 0:
                    self.auto_cov[i] = _cov_sum / _cov_count
                else:
                    self.auto_cov[i] = 0.0
                    self.cov_count[i] = 0
                auto_cov = self.auto_cov[i]
                if auto_cov == 0.0:
                    auto_cov = None
                # print(f'acov cal: {_old_cov_sum} {_old_cov_count}; {_new_cov_sum} {_new_cov_count}; {i} {self.auto_cov[i]} {self.cov_count[i]}')
            
                # a = np.append(self.trade_price_buy[self.window_len - right:-left],self.trade_price_sell[self.window_len - right:-left]) \
                #     if left != 0 else np.append(self.trade_price_buy[self.window_len - right:],self.trade_price_sell[self.window_len - right:])
                # b = np.append(self.trade_price_buy[self.window_len - right - 1:-left-1],self.trade_price_sell[self.window_len - right - 1:-left-1])
                # c = np.append(self.trade_price_buy[self.window_len - right - 2:-left-2],self.trade_price_sell[self.window_len - right - 2:-left-2])
                # auto_cov = np.nanmean(np.log(a/b)*np.log(b/c))
                # if self.load_ts_num >= self.window_len:
                #     print(a, b, c)
                #     print(f'ac cal: {self.load_ts_num}/{self.window_len}/{self.right} ? {left} - {right} --> {auto_cov}')
            else:
                self.auto_cov[i] = 0.0
                auto_cov = None

            # 返回特征结果
            if self.load_ts_num >= right:
                res.append((f'{self.fe_name1}_{left}_{right}', self.latest_trade_ts, auto_cov))
        # print(f'{self.last_data}, {len(res)}, {res}')
        self.last_data = self.get_trade_data(0)
        # if len(res) <= 1:
        #     time.sleep(100)
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

    ins = AutoCovGenerator(trans_config)
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
