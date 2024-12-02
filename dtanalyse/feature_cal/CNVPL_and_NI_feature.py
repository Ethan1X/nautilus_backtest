import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from util.load_s3_data import LoadS3Data
from util.time_method import *
from data_generator import *
from typing import TypedDict
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
from tqdm import tqdm
import numpy as np

class CNVPL_NI_Config(FeatureConfig): 
    '''
        特征自定义的参数: 回望周期
        将回望周期内计算的CNVPL和NI求平均返回
    '''
    left: float # 回望起始时刻
    right: float # 回望结束时刻
    price_level: int # 对当前时间步，计算多少个价格level的累计名义价值/不平衡度

import copy


class CNVPL_and_NI(FeatureGenerator):

    # Author: Yihang Zhai
    # Reviewer: Xilin Liu & Shaofan Wu

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        """维护一个feature list，保留回望周期内计算出的feature，方便进行均值等计算"""
        self.feature_list = [] 
        self.price_level = config['price_level']
        self.window_L = config["left"]
        self.window_R = config["right"]
        self.min_max = (min(self.window_L), max(self.window_R))
        self.CNVPL_bid_sum = [0] * len(self.window_L)
        self.CNVPL_ask_sum = [0] * len(self.window_L)
        self.NI_sum = [0] * len(self.window_L)
        self.left_index = [0] * len(self.window_L)
        self.right_index = [0] * len(self.window_L)
        self.name_tail = [f'{self.window_L[i]}_{self.window_R[i]}_{self.price_level}' for i in range(len(self.window_L))]
        
    def process(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None, ts_event=None):
        '''
            feature计算主流程 供外部调用
        '''
        if depth is not None:
            self.update(ticker, trade, depth, ts_event)
            res = []
            if ts_event is None:
                _ts_event = depth.ts_event
            else:
                _ts_event = ts_event
            for i in range(len(self.window_L)):
                if self.CNVPL_bid_sum[i] == 0 or self.right_index[i] == 0:
                    return None
                interval = self.right_index[i] - self.left_index[i] + 1
                res.extend([(f'CNVPL_ask_mean_{self.name_tail[i]}', _ts_event, self.CNVPL_ask_sum[i] / interval),
                            (f'CNVPL_bid_mean_{self.name_tail[i]}', _ts_event, self.CNVPL_bid_sum[i] / interval),
                            (f'NI_mean_{self.name_tail[i]}', _ts_event, self.NI_sum[i] / interval)])
            # print(f'cn calc: {res} {depth}')
            return res
        return None


    def update(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None, ts_event=None):
        '''
            更新feature内部状态,如果当前回望窗口 头部时间-尾部时间大于窗口右边界，则pop
        '''
        if ts_event is not None:
            tp = ts_event
        else:
            tp = depth.ts_event
        res = self.calculate(depth, tp)
        
        if res:
            self.feature_list.append(res)
        if len(self.feature_list) == 0:
            return
        
        offset = 0
        while self.feature_list[offset][3] < tp - self.min_max[1]*10**9:
            offset += 1
        for i in range(len(self.window_L)):
            while self.right_index[i] < len(self.feature_list) and self.feature_list[self.right_index[i]][3] <= tp - self.window_L[i]*10**9:
                self.CNVPL_ask_sum[i] += self.feature_list[self.right_index[i]][0]
                self.CNVPL_bid_sum[i] += self.feature_list[self.right_index[i]][1]
                self.NI_sum[i] += self.feature_list[self.right_index[i]][2]
                self.right_index[i] += 1
            
            while self.left_index[i] < len(self.feature_list) and self.feature_list[self.left_index[i]][3] < tp - self.window_R[i]*10**9:
                tmp = self.feature_list[self.left_index[i]]
                self.CNVPL_ask_sum[i] -= tmp[0]
                self.CNVPL_bid_sum[i] -= tmp[1]
                self.NI_sum[i] -= tmp[2]
                self.left_index[i] += 1
            
            self.left_index[i] -= offset
            self.right_index[i] -= offset
        
        if offset > 0:
            self.feature_list = self.feature_list[offset:]
        
        
    def calculate(self,depth,ts=None):
        '''
            计算feature
            returns: [(fe_name, 'tp', 'ret'), ...] #tp: 毫秒
        '''
        if ts is None:
            tp = depth.ts_event
        else:
            tp = ts
            
        _asks = depth.asks()
        _bids = depth.bids()
        if _asks and len(_asks) >= self.price_level and _bids and len(_bids) >= self.price_level:
            x_bid = x_ask = 0
            for i in range(self.price_level):
                pb, sb, pa, sa = _bids[i]['price'], _bids[i]['size'], _asks[i]['price'], _asks[i]['size']
                x_bid += pb * sb
                x_ask += pa * sa
            
            if x_ask + x_bid != 0:
                ni = (x_ask - x_bid) / (x_ask + x_bid)
                return x_ask, x_bid, ni, tp
        return None
    
    
if __name__ == '__main__':
    initlog(None, "cnvpl_feature2.log", logging.INFO)
    begin_time = datetime.datetime(2024, 3, 27, 21,tzinfo=TZ_8)
    end_time = datetime.datetime(2024, 3, 27, 21, 59,tzinfo=TZ_8)
    exchange = 'binance'
    symbol = 'btc_usdt'

    config_list = []
    for i in range(9):
        config_list.append(CNVPL_NI_Config(left = 2**(i-1) if i >0 else 0, right = 2**i, price_level=5))
    # CNVPLConfig = CNVPL_NI_Config(window_L = 0, window_R = 0, price_level=5)
    ins_X = CNVPL_and_NI(config_list[0])
    fe_list = []

    print('开始生成数据')
    data_generator = get_data_generator(begin_time, end_time, exchange, symbol)
    print('数据生成完成')

    time.sleep(20)
    import time
    print('start')
    start_time = time.time()    
    for idx, (row,type) in enumerate(data_generator):
        # print('start2')
        if type == 'depth':
            fe_list.append(ins_X.process(depth=row))
        else:
            pass
        # if idx > 1000:
        #     break
    print(fe_list[:4])
    end_time= time.time()
    print(end_time-start_time)
    
    # fe_filter_list = list(filter(lambda x: x is not None, fe_list))
    
    # import joblib
    # joblib.dump(fe_filter_list,"fe_list.pkl")
