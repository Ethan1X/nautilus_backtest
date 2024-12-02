import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_generator import *
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
import numpy as np


class EffectiveSpreadGenerator(FeatureGenerator):
    """
    author: Xilin Liu
    reviewer: Weijie Huang
    EffectiveSpread Generator
    详见文档第2个
    """

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        """
        ES_t = 2(lnP_t - lnM_t).abs()
        P_t 使用 trade 计算
        M_t 使用 tick 计算
        """
        self.fe_name = "EffectiveSpread"
        self.latest_trade_price = None
        self.latest_mid_price = None
        self.ts_event = None

    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None):
        pass
        
    def process(
        self, ticker: QuoteTick = None, trade: TradeTick = None, depth: OrderBook = None
    ):  
        # 数据冗余
        if depth is not None: return [(self.fe_name, None, None)]
        
        # 更新数据
        self.update(ticker, trade, depth)
        
        # 开始计算因子值
        if trade is not None:
            feature_ret = self.calculate()
            return feature_ret
        
        # 返回因子值
        return [(self.fe_name, None, None)]

    def update(
        self, ticker: QuoteTick = None, trade: TradeTick = None, depth: OrderBook = None
    ):
        # 更新 trade 数据
        if trade is not None:
            try:
                self.latest_trade_price = trade.price
                self.ts_event = trade.ts_event
            except: 
                pass
        
        # 更新 ticker 数据
        if ticker is not None:
            try:
                self.latest_mid_price = (ticker.ask_price + ticker.bid_price)/2
                # self.ts_event = depth.ts_event  
            except:
                pass
        return

    def calculate(self):

        # 数据冗余
        if self.latest_trade_price is None:
            return [(self.fe_name, None, None)]
        if self.latest_mid_price is None:
            return [(self.fe_name, None, None)]
        
        # 计算因子值
        f_value = 2*abs(np.log(self.latest_trade_price) - np.log(self.latest_mid_price))
        
        # 返回因子值
        return [(self.fe_name, self.ts_event, f_value)]


if __name__ == "__main__":
    initlog(None, "EffectiveSpread.log", logging.INFO)
    f_Config = FeatureConfig()

    begin_time = datetime.datetime(2024, 3, 27, 21)
    end_time = datetime.datetime(2024, 3, 27, 21, 59)
    exchange = "binance"
    symbol = "btc_usdt"

    ins = EffectiveSpreadGenerator(f_Config)
    fe_list = []

    from time import time
    st = time()
    for idx, row in enumerate(
        get_data_generator(begin_time, end_time, exchange, symbol)
    ):
        # print(idx, row)

        # if idx >= 1000:
        #     break

        if row[1] == "ticker":
            """
             ({'ask_price': 42590.14, 'ask_size': 4.21882, 'bid_price': 42590.13, 'bid_size': 5.99953, 'ts_event': 1702746000000000000}, 'ticker')
            """
            # print('\n', row)
            fe = ins.process(ticker=row[0])
            fe_list.append(fe)
        
        if row[1] == "depth":
            """
             ({'ts_event': 1702746000518000000000, '_asks': [{'p': 42590.14, 's': 4.51367}, {'p': 42590.28, 's': 0.0698}, {'p': 43384.27, 's': 0.04899}, {'p': 49244.6, 's': 0}], 
                                                    '_bids': [{'p': 42588.33, 's': 0.46994}, {'p': 42541.55, 's': 0.001}, {'p': 42164.23, 's': 0.00323}]}, 'depth')
            """
            # print('\n depth is:', row[0]._asks)
            fe = ins.process(depth=row[0])
            fe_list.append(fe)


        if row[1] == "trade":
            """
             ({'aggressor_side': 'sell', 'price': 42590.13, 'size': 0.00117, 'ts_event': 1702746000004000000}, 'trade')
            """
            # print('\n', row)
            fe = ins.process(trade=row[0])
            fe_list.append(fe)

    print(fe_list)
    print('\n time cost:',time()-st)

    import joblib
    joblib.dump(fe_list, 'fe_list.pkl')
