import os, sys, logging, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from util.load_s3_data import LoadS3Data
from util.time_method import *
from data_generator import *
from typing import TypedDict
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
from typing import TypedDict

class TradeInfo:
    def __init__(self, trade, contribution, side):
        self.trade = trade
        self.contribution = contribution
        self.side = side

class TFIConfig(TypedDict): 
    '''
        特征自定义的参数: 回望周期
        从回望起始时刻开始，依次往后推移计算TFI
        Author: Yu Liu, Weijing Yin
        Reviewer: Shaofan Wu, Zhihan Cai
    '''
    window_L: float # 回望起始时刻
    window_R: float # 回望结束时刻

class TFIGenerator(FeatureGenerator):
    def __init__(self, config: TFIConfig) -> None:
        super().__init__(config)
        self.unit_sec = 1e9  # 1秒的纳秒数
        self._trade_list = []
        self.right = config['right']
        for w in self.right:
            self._trade_list.append([])
        self.min_max = (min(self.right), max(self.right))
        self.max_index = self.right.index(self.min_max[1])    # 最长周期的列表索引
        self.min_interval = self.min_max[0] * self.unit_sec        # 最小计算区间，纳秒数
        self.fe_name = f'TFI'
        self.feature_counts = len(self.right)
        self.data_index = [0] * self.feature_counts
        self.cumulative_gain = [0] * self.feature_counts
        self.cumulative_loss = [0] * self.feature_counts

    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None):
        pass
    
    def update(self, trade: TradeTick=None):
        # Ensure trades are within the specified window size
        for i in range(self.feature_counts):
            while self._trade_list[i] and trade.ts_event - self._trade_list[i][0].trade.ts_event > self.right[i] * 10**9:
                old_trade_info = self._trade_list[i].pop(0)
                # Subtract the old trade's contribution from the cumulative gain or loss
                # if old_trade_info.contribution > 0:
                if old_trade_info.side == "buy":
                    self.cumulative_gain[i] -= old_trade_info.contribution
                else:
                    self.cumulative_loss[i] -= old_trade_info.contribution
        # start_pos = self.data_index[self.max_index]
        # data_pos = start_pos
        # for i in range(self.feature_counts):
        #     while self._trade_list and trade.ts_event - self._trade_list[data_pos].trade.ts_event > self.min_interval:
        #         _feature_data_pos = self.data_index[i]
        #         if _feature_data_pos > data_pos:
        #             continue
        #         if trade.ts_event - self._trade_list[_feature_data_pos].trade.ts_event <= self.right[i] * self.unit_sec:
        #             continue
        #         old_trade_info = self._trade_list[data_pos]
        #         # Subtract the old trade's contribution from the cumulative gain or loss
        #         if old_trade_info.side == "buy":
        #             self.cumulative_gain[i] -= old_trade_info.contribution
        #         else:
        #             self.cumulative_loss[i] -= old_trade_info.contribution
        #         self.data_index[i] += 1
        #     data_pos += 1
        # start_pos = self.data_index[self.max_index]
        # del self._trade_list[0:start_pos]
        # for i in range(self.feature_counts):
        #     self.data_index[i] -= start_pos

    def process(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
        if not trade:
            return

        # Add the new trade and its contribution to the list
        contribution = trade.size
        side = trade.aggressor_side
        for i in range(self.feature_counts):
            if side == "buy":
                self.cumulative_gain[i] += contribution
            else:
                self.cumulative_loss[i] += contribution
            self._trade_list[i].append(TradeInfo(trade, contribution, side))
        self.update(trade)

        return self.calculate(True)

    def calculate(self, flag):
        if not flag:
            return

        res = []

        for i in range(self.feature_counts):
            end_time = self._trade_list[i][-1].trade.ts_event
            tfi = 0
            _gain = self.cumulative_gain[i]
            _loss = self.cumulative_loss[i]
            if _gain + _loss  > 0:
                tfi = (_gain - _loss) / (_gain + _loss)
                res.append((f'TFI_{self.right[i]}', end_time, tfi))
        return res


if __name__ == '__main__':
    initlog(None, 'TFI_feature.log', logging.INFO)
    TFI_config = TFIConfig(left = 0, right = 300)
    #print(CRSI_config['window_L'])
    #print(CRSI_config['window_R'])

    begin_time = datetime.datetime(2024, 3, 27, 21,tzinfo=TZ_8)
    end_time = datetime.datetime(2024, 3, 27, 21, 59,tzinfo=TZ_8)
    exchange = 'binance'
    symbol = 'btc_usdt'

    ins = TFIGenerator(TFI_config)
    fe_list = []
    
    config_list = []
    for i in range(3):
        config_list.append(FeatureConfig(left = 0, right = 5*(i+1)))
    
    data_generator = get_data_generator(begin_time, end_time, exchange, symbol)

    for idx, row in enumerate(data_generator):

        if row[1] == 'trade':
            fe_list.append(ins.process(trade=row[0]))

        elif row[1] == "depth":
            fe_list.append(ins.process(depth=row[0]))

        else:
            fe_list.append(ins.process(ticker=row[0]))


#         if idx > 10000:
#             break
    
    fe_filter_list = list(filter(lambda x: x is not None, fe_list))
    print(fe_filter_list)

    import joblib
    joblib.dump(fe_filter_list,"fe_list.pkl")
    
    
