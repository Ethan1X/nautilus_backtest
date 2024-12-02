import os, sys, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_generator import *
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog

class BreadthGenerator(FeatureGenerator):
    '''
    特征名称：
        Breadth： 两次trade之间ticker变化次数,
        Immediacy：回望区间长度除以Breadth, 
        VolumePerBreadth：两次trade之间成交量之和除以Breadth
    特征编号：7，8，10
    所需数据：ticker数据和trade数据
    Author: Qiushi Bu
    Reviewer: Haotong Li
    Notes: 时间点相同的多次trade不重复计数； breadth的计数包括了两次trade中间的ticker更新次数以及中间的trade数量
    '''
    def __init__(self, config: FeatureConfig) -> None:
        '''
        config: 
            'left'，'right': 回望区间的两端，即：从 当前时刻之前第right次trade 到 当前时刻之前第left次trade 之间的各种统计量，例如ticker变化次数。
            'left'和'right'分别对应公式中的\Delta_1和\Delta_2          
            推荐使用的left和right：用2的幂次方，例如(1,2), (2,4), (4,8) 等，并且注意单位不是时间，而是次，要传入整数
        '''
        super().__init__(config)
        self.data_mode = "normal"
        if "data_mode" in config.keys():
            self.data_mode = config['data_mode']

        self.left = config['left']
        self.right = config['right']
        self.min_max = (min(self.left), max(self.right))
        self.window_len = self.min_max[1]+1
        self.period = len(self.right)
        self.ticker_count = np.zeros(self.window_len)
        self.trade_volume = np.zeros(self.window_len)    # 记录right个时刻trade交易量之和，初始为0
        self.trade_num = -1 # 记录已经存储的交易次数
        self.last_trade_time = 0 
        self.ts_event = 0
        self.fe_name1 = f'Breadth'  
        self.fe_name2 = f'Immediacy'
        self.fe_name3 = f'VolumePerBreadth'

        self.trade_volumes_buy = np.zeros(self.window_len) * np.nan
        self.trade_volumes_sell = np.zeros(self.window_len) * np.nan
        self.trade_volume_sum = [0.0] * self.period
        self.ticker_count_sum = [0] * self.period
        self.load_ts_num = 0
        self.data_index = None
        self.last_data:QuoteTick = None
        self.last_ticker_count = 0
        self.latest_trade_ts = None
        self.signal = False

    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None):
        self.load_ts_num = load_sum

        if self.data_index is not None and data_index != self.data_index:
            self.last_data = None
            self.data_index = data_index
            self.ticker_count[data_index] = 0

        if self.latest_trade_ts is None and price_list_buy is not None:
            self.trade_volumes_buy = volumes_list_buy
            self.trade_volumes_sell = volumes_list_sell
            self.data_index = -1
            self.window_len = len(self.trade_volumes_buy)
            self.ticker_count = np.zeros(self.window_len)

    def get_data_index(self, shift):
        return (self.window_len + self.data_index - shift) % self.window_len

    def get_trade_data(self, shift):
        _data_index = self.get_data_index(shift)
        _data = QuoteTick(0.0, self.trade_volumes_sell[_data_index],
                          0.0, self.trade_volumes_buy[_data_index], 
                          0)
        return _data

    def get_ticker_data(self, shift):
        _data_index = self.get_data_index(shift)
        return self.ticker_count[_data_index]

    def process(self, ticker: QuoteTick = None, trade: TradeTick = None, depth: OrderBook = None):
        '''
            feature计算主流程 供外部调用
        '''
        # print(f'process: {ticker} {trade}')
        feature_ret = None
        if trade is not None:
            if self.data_mode == "central":
                self.update_signal(trade)
            else:
                self.update(ticker, trade, depth)
            feature_ret = self.calculate()
        elif ticker is not None:
            self.update(ticker=ticker)
            return None
        else:
            return None

        return feature_ret

    def update_signal(self, trade: TradeTick=None):
        if trade is not None:
            self.latest_trade_ts = trade.ts_event
            self.ts_event = trade.ts_event
            self.signal = True
        else:
            self.latest_trade_ts = None
            self.signal = False

    # def process(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
    #     '''
    #     首先判断来的数据是不是ticker数据或者trade数据，如果是，进行信息更新，如果不是，不进行处理，返回None
    #     '''
    #     if depth is not None:  #如果没有传来ticker数据或者trade数据，则不更新
    #         return None
        
    #     self.update(ticker, trade, depth)
    #     if ticker is not None:  #如果传来的是ticker数据，进行更新，但不返回值
    #         return None
    #     feature_ret = self.calculate()

    #     return feature_ret
    
    def update(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
        '''
        更新信息，包括记录trade次数，记录ticker次数，记录trade volume之和
        '''
        # if trade is not None: 
        #     self.ts_event = trade.ts_event
        #     if self.ts_event // 1000 != self.last_trade_time // 1000:  # 判断交易时间是否相同，如果是同一个时刻(微秒)发生的交易则不更新trade_num
        #         self.trade_num += 1 
        #         self.last_trade_time = self.ts_event
        #     if self.trade_num > self.min_max[1]: # 如果大于endtime，将第一个元素删除，并在最后补一个0
        #         self.ticker_count = np.append(self.ticker_count[1:], 0)
        #         self.trade_volume = np.append(self.trade_volume[1:], 0)
        #         self.trade_num -= 1
        #     self.trade_volume[self.trade_num] += trade.size
        if ticker is not None: # ticker数据只更新状态，不返回信息
            self.ts_event = ticker.ts_event
            if self.load_ts_num >= 0:
                self.ticker_count[self.get_data_index(0)] += 1
            # self.ticker_count[-1] += 1
            return 
        elif depth is not None: # 不处理depth数据
            self.ts_event = depth.ts_event
            return 
        return 

    def calc_feat_values(self, period, left, right, old_data=None):
        if old_data is None:
            _old_data = self.get_trade_data(right+1)
        else:
            _old_data = old_data
        
        _ticker_delta = 0
        if left == 0:
            if self.last_data is None:
                _ticker_delta = self.get_ticker_data(1)
            else:
                _ticker_delta = self.get_ticker_data(0)
        else:
            if self.last_data is None:
                _ticker_delta = self.get_ticker_data(left)
        if self.last_data is None:
            _ticker_delta -= self.get_ticker_data(right)
            
        _new_data = self.get_trade_data(left)
        _delta_buy = _new_data.bid_size
        if np.isnan(_delta_buy):
            _delta_buy = 0
        if not np.isnan(_old_data.bid_size):
            _delta_buy -= _old_data.bid_size
        _delta_sell = _new_data.ask_size
        if np.isnan(_delta_sell):
            _delta_sell = 0
        if not np.isnan(_old_data.ask_size):
            _delta_buy -= _old_data.ask_size

        # print(f'br calc(before): {period} {self.data_index} {_old_data} {_new_data} {self.ticker_count_sum}  {old_data}')
        self.trade_volume_sum[period] += _delta_buy + _delta_sell
        _ticker_count = self.ticker_count_sum[period]
        if self.last_data is None:
            self.ticker_count_sum[period] += _ticker_delta
        # print(f'br calc(after): {period} {self.ticker_count_sum} {left} {right} {self.load_ts_num} {self.ticker_count[0:20]}')
    
        _br = _ticker_count + _ticker_delta + right - left
        _imm = (right - left) / _br
        _vpb = self.trade_volume_sum[period] / _br
        return _br, _imm, _vpb
        
    def calculate(self):
        '''
        如果trade_num到了right，按照公式计算三个统计量的值
        '''
        res = []
        breadth = immediacy = VolumePerBreadth = None
        for i in range(self.period):
            left, right = self.left[i], self.right[i]
            if True or self.load_ts_num > left:
                if self.last_data is None:
                    # 新数据，窗口滑动一格
                    breadth, immediacy, VolumePerBreadth = self.calc_feat_values(i, left, right)
                else:
                    # 仅更新当前数据
                    if left > 0 and self.load_ts_num >= right:
                        breadth = self.ticker_count_sum[i] + right - left
                        immediacy = (right - left) / breadth
                        VolumePerBreadth = self.trade_volume_sum[i] / breadth
                    else:
                        breadth, immediacy, VolumePerBreadth = self.calc_feat_values(i, left, right, self.last_data)
            # print(f'calc: {self.load_ts_num} {i} {left}-{right} ')
            if self.load_ts_num >= right:
                res.extend([(f'{self.fe_name1}_{left}_{right}', self.ts_event, breadth),
                        (f'{self.fe_name2}_{left}_{right}', self.ts_event, immediacy),
                        (f'{self.fe_name3}_{left}_{right}', self.ts_event, VolumePerBreadth)])

        self.last_data = self.get_trade_data(0)
        # if len(res) > 0:
        #     print(f'br res: {self.ts_event} {res[0]}')
        return res

    # def calculate(self):  
    #     '''
    #     如果trade_num到了right，按照公式计算三个统计量的值
    #     '''
    #     res = []
    #     for i in range(len(self.left)):
    #         left, right = self.left[i], self.right[i]
    #         if self.trade_num >= right: # 如果到right，计算并返回breadth值
    #             breadth = sum(self.ticker_count[0:(right-left)]) + right - left # ticker的次数加上trade的次数
    #             immediacy = (right-left) / breadth
    #             VolumePerBreadth = sum(self.trade_volume[0:(right-left)]) / breadth
    #             res.extend([(f'{self.fe_name1}_{left}_{right}', self.ts_event, breadth),
    #                     (f'{self.fe_name2}_{left}_{right}', self.ts_event, immediacy),
    #                     (f'{self.fe_name3}_{left}_{right}', self.ts_event, VolumePerBreadth)])
    #     return res


if __name__ == '__main__':     
    initlog(None, 'breadth_feature.log', logging.INFO)
    config_list = [] 
    for i in range(9):
        config_list.append(FeatureConfig(left = 2**(i-1) if i > 0 else 0, right = 2**i))
    breadth_config = config_list[0]
    begin_time = datetime.datetime(2024, 3, 27, 21,0,tzinfo=TZ_8) # 要加一个tzinfo
    right = datetime.datetime(2024, 3, 27, 21, 59,tzinfo=TZ_8)
    exchange = 'binance'
    symbol = 'btc_usdt'

    ins = BreadthGenerator(breadth_config)
    fe_list = []

    data_generator = get_data_generator(begin_time, right, exchange, symbol)
          
    
    for idx, row in enumerate(data_generator):
        #print(idx)
        if row[1] == 'ticker':
            fe_list.append(ins.process(ticker=row[0]))
        elif row[1] == 'depth':
            fe_list.append(ins.process(depth=row[0]))
        else:
            fe_list.append(ins.process(trade=row[0]))

    print(fe_list[:100])
