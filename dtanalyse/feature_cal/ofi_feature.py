import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from util.load_s3_data import LoadS3Data
from util.time_method import *
from data_generator import *
from typing import TypedDict
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog

'''
    样例程序，无实际用途
'''
class OFIGenereator(FeatureGenerator):

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)


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
        return [('ofi', 1)]


if __name__ == '__main__':
    initlog(None, 'ofi_feature.log', logging.INFO)
    ofi_config = FeatureConfig(a=1, b=2, window_time='a')
    print(ofi_config['a'])

    begin_time = datetime.datetime(2024, 3, 27, 21, tzinfo=TZ_8)
    end_time = datetime.datetime(2024, 3, 27, 21, 59, tzinfo=TZ_8)
    exchange = 'binance'
    symbol = 'btc_usdt'

    ins = OFIGenereator(ofi_config)
    fe_list = []
    
    data_generator = get_data_generator(begin_time, end_time, exchange, symbol)
    t1 = time.time()
    for idx, row in enumerate(data_generator):

        if row['type'] == 'ticker':
            fe_list.append(ins.process(ticker=tmp_trans_s3_backtest_ticker(row)))
        elif row['type'] == 'depth':
            fe_list.append(ins.process(depth=tmp_trans_s3_backtest_depth(row)))
        else:
            fe_list.append(ins.process(depth=tmp_trans_s3_backtest_trade(row)))

        print(idx, row)
        if idx > 100:
            break
    print(fe_list[:4])
    print(f"生成特征耗时:{time.time() - t1}") 
