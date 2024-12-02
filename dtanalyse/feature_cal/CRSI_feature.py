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
    def __init__(self, trade, contribution):
        self.trade = trade
        self.contribution = contribution

class CRSIConfig(TypedDict): 
    '''
        特征自定义的参数: 回望周期
        从回望起始时刻开始，依次往后推移计算CRSI
        Author: Yu Liu, Weijing Yin
        Reviewer: Shaofan Wu, Zhihan Cai
    '''
    window_L: float # 回望起始时刻
    window_R: float # 回望结束时刻

class CRSIGenerator(FeatureGenerator):
    def __init__(self, config: CRSIConfig) -> None:
        super().__init__(config)
        self._trade_list = []
        self.window_R = config["right"]
        for w in self.window_R:
            self._trade_list.append([])
        self.cumulative_gain = [0] * len(self.window_R)
        self.cumulative_loss = [0] * len(self.window_R)

    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None):
        pass
    
    def update(self, trade: TradeTick=None):
        # Ensure trades are within the specified window size
        for i in range(len(self.window_R)):
            while self._trade_list[i] and trade.ts_event - self._trade_list[i][0].trade.ts_event > self.window_R[i] * 10**9:
                old_trade_info = self._trade_list[i].pop(0)
                # Subtract the old trade's contribution from the cumulative gain or loss
                if old_trade_info.contribution > 0:
                    self.cumulative_gain[i] -= old_trade_info.contribution
                else:
                    self.cumulative_loss[i] -= old_trade_info.contribution

    def process(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
        if not trade:
            return

        # Add the new trade and its contribution to the list
        if len(self._trade_list[0]) > 0:
            if trade.price > 0:            
                prev_trade = self._trade_list[0][-1].trade
                if prev_trade.price == 0 or trade.price == 0:
                    print(f'[error] price == 0: {prev_trade} {trade} {len(self._trade_list[0])}')
                price_change = (trade.price - prev_trade.price) / prev_trade.price if prev_trade.price > 0 else 0.0
                if price_change > 0:
                    contribution = price_change
                    for i in range(len(self.window_R)):
                        self.cumulative_gain[i] += contribution
                else:
                    contribution = price_change
                    for i in range(len(self.window_R)):
                        self.cumulative_loss[i] += contribution
                for i in range(len(self.window_R)):
                    self._trade_list[i].append(TradeInfo(trade, contribution))
        else:
            print(f'[warning] trade list[0] is empty')
            for i in range(len(self.window_R)):
                self._trade_list[i].append(TradeInfo(trade, 0))  # No contribution if it's the first trade

        self.update(trade)

        return self.calculate()

    def calculate(self):
        if len(self._trade_list[0]) <= 1:
            return
        
        end_time = self._trade_list[0][-1].trade.ts_event
        res = []
        for i in range(len(self.window_R)):
            custom_rsi = 0
            if self.cumulative_gain[i] + abs(self.cumulative_loss[i]) > 0:
                custom_rsi = (self.cumulative_gain[i] - abs(self.cumulative_loss[i])) / (self.cumulative_gain[i] + abs(self.cumulative_loss[i]))

            res.append((f'CRSI_{self.window_R[i]}', end_time, custom_rsi))
        return res


if __name__ == '__main__':
    initlog(None, 'CRSI_feature.log', logging.INFO)
    CRSI_config = CRSIConfig(left = 0, right = 300)
    #print(CRSI_config['window_L'])
    #print(CRSI_config['window_R'])

    begin_time = datetime.datetime(2024, 3, 27, 21,tzinfo=TZ_8)
    end_time = datetime.datetime(2024, 3, 27, 21, 59,tzinfo=TZ_8)
    exchange = 'binance'
    symbol = 'btc_usdt'

    ins = CRSIGenerator(CRSI_config)
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
    
    
