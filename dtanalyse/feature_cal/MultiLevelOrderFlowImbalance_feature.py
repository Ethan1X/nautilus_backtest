import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_generator import *
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
from collections import defaultdict
import numpy as np
import queue


class MultiLevelOrderFlowImbalanceGenerator(FeatureGenerator):
    """
    author: Xilin Liu
    reviewer: Weijie Huang
    MultiLevelOrderFlowImbalance Generator
    公式过于复杂 参考文档里面的 MLOFI
    详见文档第一个
    """

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        """
            参数说明：
                M 计算第几档的 1代表 bid1/ask1
                left right 回望周期
        """
        self.fe_name = "MultiLevelOrderFlowImbalance"
        # 防止赋参失败
        self.M = config['M']
        # self.left = config['left']
        # self.right = config['right']
        self.windows = config['windows']
        self.interval = max([w[1] for w in self.windows])

        self.last_depth = None
        self.expM_queue = {}      # 用于存储历次计算的 expM, 每一个会是一个expM
        self.expM_sum = {}
        for m in self.M:
            self.expM_queue[m] = queue.Queue()
            self.expM_sum[m] = []
            for i in range(len(self.windows)):
                self.expM_sum[m].append({'sum': 0.0, 'left': 0, 'right': 0})

        self.has_accumulated_time = False   # 指标参数，只有为 True 才开始生成因子
        self.ts_event = None
        self.sum_value = 0

    def process(
        self, ticker: QuoteTick = None, trade: TradeTick = None, depth: OrderBook = None, ts_event = None):
        if depth is None: return [(self.fe_name, None, None)]

        if ts_event is None:
            self.ts_event = depth.ts_event
        else:
            self.ts_event = ts_event

        # 冗余检查
        # p,s = get_depth_level(depth, self.M-1, 'asks')
        # if p is None: return [(self.fe_name, self.ts_event, None)]
        # p,s = get_depth_level(depth, self.M-1, 'bids')
        # if p is None: return [(self.fe_name, self.ts_event, None)]

        self.update(ticker, trade, depth)
        feature_ret = self.calculate()
        return feature_ret

    def update(
        self, ticker: QuoteTick = None, trade: TradeTick = None, depth: OrderBook = None
    ):  
        """
        更新数据
        生成中间变量 expM
        """
        # 先处理depth
        data = {}
        d = {'asks': depth.asks(), 'bids': depth.bids()}
        if not d['asks'] or not d['bids']:
            return
        for side in ['asks', 'bids']:
            data[side] = {}
            for m in self.M:
                if len(d[side]) < m:
                    continue
                p, s = d[side][m - 1]['price'], d[side][m - 1]['size']
                data[side][m] = (p, s)
        data['time'] = self.ts_event

        # 开始计算
        if self.last_depth is not None:
            for m in self.M:
                if m not in data['bids'] or m not in data['asks'] or m not in self.last_depth['bids'] or m not in self.last_depth['asks']:
                    continue
                # calc WM
                p, s = data['bids'][m]
                p_last,s_last = self.last_depth['bids'][m]
                if p > p_last:
                    WM = s
                if p == p_last:
                    WM = s - s_last
                if p < p_last:
                    WM = - s_last

                # calc VM
                p, s = data['asks'][m]
                p_last,s_last = self.last_depth['asks'][m]
                if p < p_last:
                    VM = s
                if p == p_last:
                    VM = s - s_last
                if p > p_last:
                    VM = - s_last
            
                # calc expM
                expM_t = (WM - VM, data['time'])
                self.expM_queue[m].put(expM_t)
                for i in range(len(self.windows)):
                    left, right = self.windows[i]
                    while self.expM_sum[m][i]['left'] < self.expM_queue[m].qsize() and self.ts_event - self.expM_queue[m].queue[self.expM_sum[m][i]['left']][1] >= left * 1e9:
                        self.expM_sum[m][i]['sum'] += self.expM_queue[m].queue[self.expM_sum[m][i]['left']][0]
                        self.expM_sum[m][i]['left'] += 1
                    while self.expM_sum[m][i]['right'] < self.expM_queue[m].qsize() and self.ts_event - self.expM_queue[m].queue[self.expM_sum[m][i]['right']][1] > right * 1e9:
                        self.expM_sum[m][i]['sum'] -= self.expM_queue[m].queue[self.expM_sum[m][i]['right']][0]
                        self.expM_sum[m][i]['right'] += 1
                    
                while self.expM_queue[m].qsize()>=1:
                    first = self.expM_queue[m].queue[0]
                    if self.ts_event - first[1] > self.interval * 1e9:
                        self.expM_queue[m].get() # 删除第一个元素
                        self.has_accumulated_time = True
                        for i in range(len(self.windows)):
                            self.expM_sum[m][i]['left'] -= 1
                            self.expM_sum[m][i]['right'] -= 1
                    else: break
        
        self.last_depth = data
        return

    def calculate(self):
        if not self.has_accumulated_time:
            return None
        res = []
        for m in self.M:
            for i in range(len(self.windows)):
                left, right = self.windows[i]
                res.append((f'{self.fe_name}_{left}_{right}_{m}', self.ts_event, self.expM_sum[m][i]['sum']))
        return res


if __name__ == "__main__":
    initlog(None, "MultiLevelOrderFlowImbalance.log", logging.INFO)
    # 可取的参数组合
    # 推荐 config_list[0]
    config_list = list(FeatureConfig(M=i, left=0, right=2) for i in range(1,6)) +\
          list(FeatureConfig(M=i, left=2, right=10) for i in range(1,6)) +\
          list(FeatureConfig(M=i, left=0, right=30) for i in range(1,6))
    f_Config = config_list[14]
    """
    观察发现：
    1.以上的参数解释性都尚可
    2.随着M数字的增加，因子值极值的可解释性越差
    3.涨跌幅和因子值大小并不完全遵守线性关系
    """

    begin_time = datetime.datetime(2024, 3, 27, 21)
    end_time = datetime.datetime(2024, 3, 27, 21, 59)
    exchange = "binance"
    symbol = "btc_usdt"

    ins = MultiLevelOrderFlowImbalanceGenerator(f_Config)
    fe_list = []

    from time import time
    st = None

    for idx, row in enumerate(
        get_data_generator(begin_time, end_time, exchange, symbol)
    ):
        # print(idx, row)

        # if idx >= 1000:
        #     break
        if st is None:
            st = time()

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
            # print('\n depth is:', row[0])
            # print(get_depth_level(row[0], 1, 'asks'))
            # sys.exit()
            fe = ins.process(depth=row[0])
            fe_list.append(fe)


        if row[1] == "trade":
            """
             ({'aggressor_side': 'sell', 'price': 42590.13, 'size': 0.00117, 'ts_event': 1702746000004000000}, 'trade')
            """
            # print('\n', row)
            fe = ins.process(trade=row[0])
            fe_list.append(fe)


    # print(fe_list)
    print('\n time cost:',time()-st)

    import joblib
    joblib.dump(fe_list, 'fe_list.pkl')