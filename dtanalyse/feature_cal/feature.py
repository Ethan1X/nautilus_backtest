import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from util.load_s3_data import LoadS3Data
from util.time_method import *
from data_generator import *
from typing import TypedDict

class FeatureConfig(TypedDict): 
    '''
        特征自定义的参数
    '''
    # 窗口长度：单位秒
    window_time: int
 

class FeatureGenerator:

    def __init__(self, config:FeatureConfig) -> None:
        self.config = config
        # 特征名称，如果和config有强关系，根据config进行自定义
        self.fe_name = '' # ofi_1_2

    def process(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
        '''
            feature计算主流程 供外部调用
        '''
        self.update(ticker, trade, depth)
        feature_ret = self.calculate()

        return feature_ret
    
    def update(self, ticker: QuoteTick=None, trade: TradeTick=None, depth: OrderBook=None):
        '''
            更新feature内部状态 数据需要判断是否为None，只有更新了才会推送，默认值为None 
        '''
        return
    
    def calculate(self):
        '''
            计算feature
            returns: [(fe_name, 'tp', 'ret'), ...] #tp: 毫秒
        '''
        return
    
    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None):
        return

    
