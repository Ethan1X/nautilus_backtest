import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_generator import *
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
import joblib
from tqdm import tqdm
import time


class TurnoverGenerator(FeatureGenerator):
    # Author: Vijay Huang, 04/01/2024
    # Reviewer: Xilin Liu
    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        # self.left = config['left']
        self.right = config['right']
        self.min_max = (min(self.right), max(self.right))
        # self.window_len = self.min_max[1]
        self.fe_name = f'turnover'
        self.interval = self.min_max[1] + 10  # 计算区间
        self.min_interval = self.min_max[0]
        self.left_start = 0

        self.turnover_1sec = 0  # 每秒内的成交量
        self.turnover_ls = []  # 存储每秒的成交量
        self.unit_sec = 10**9  # 1秒的纳秒数
        self.ts_event = None
        self.start_time = None  # 每1秒区间的开始时间，从第一个trade时间开始递增1秒
        self.cal_time = None
        self.out = False  #

    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None):
        pass

    def process(
        self, ticker: QuoteTick = None, trade: TradeTick = None, depth: OrderBook = None
    ):
        if trade is None:
            return [(self.fe_name, None, None)]

        else:
            self.update(trade=trade)
            feature_ret = self.calculate()

            return feature_ret

    def update(
        self, ticker: QuoteTick = None, trade: TradeTick = None, depth: OrderBook = None
    ):
        # 初始化时间参数
        self.ts_event = trade.ts_event
        if self.start_time == None:
            self.start_time = trade.ts_event
        if self.cal_time is None:
            self.cal_time = trade.ts_event

        if self.ts_event - self.start_time <= self.unit_sec:  # 每1秒内的计算
            self.turnover_1sec += trade.size * trade.price

        else:  # 新数据大于第一个数据1秒，则计算并存入列表；原始数据清空
            self.turnover_ls.append(self.turnover_1sec)
            self.turnover_1sec = trade.price * trade.size
            
            # 更新首个单秒数据的时间
            self.start_time += self.unit_sec
            while self.ts_event - self.start_time > self.unit_sec:
                self.start_time += self.unit_sec
                self.turnover_ls.append(0.0)

            self.out = True
            if (self.start_time - self.cal_time) > self.interval * self.unit_sec:
                self.cal_time += self.unit_sec
                try:
                    self.turnover_ls.pop(0)
                except:
                    pass
            elif (self.start_time - self.cal_time) < self.min_interval * self.unit_sec:
                self.out = False

    def calculate(self):
        if self.out:
            res = []

            for right in self.right:
                if len(self.turnover_ls) >= right:
                    turnover = np.sum(
                        self.turnover_ls[::-1][
                            self.left_start : self.left_start + right
                        ]
                    )
                    res.append((f'{self.fe_name}_{right}', self.ts_event, turnover))
            self.out = False
            return res
        else:
            return None


if __name__ == "__main__":
    initlog(None, "turnover_feature.log", logging.INFO)

    begin_time = datetime.datetime(2024, 3, 27, 21, tzinfo=TZ_8)  # 设置时区: tzinfo=TZ_8
    end_time = datetime.datetime(2024, 3, 27, 21, 59, tzinfo=TZ_8)
    exchange = "binance"
    symbol = "btc_usdt"

    config_list = list(FeatureConfig(left=0, right=i) for i in range(5))

    # 计时
    # t1 = time.time()
    for j in tqdm(range(5)):
        turnover_config = config_list[j]
        ins = TurnoverGenerator(turnover_config)
        fe_list = []

        for idx, row in enumerate(
            get_data_generator(begin_time, end_time, exchange, symbol)
        ):
            # print(idx, row)

            # if idx >= 1000:
            #     break

            if row[1] == "ticker":
                fe_list.append(ins.process(ticker=row[0]))
            elif row[1] == "depth":
                fe_list.append(ins.process(depth=row[0]))
            else:
                fe_list.append(ins.process(trade=row[0]))

        # t2 = time.time()
        # print("Time cost:", t2 - t1)

        # dump_dir = "turnover.pkl"
        # joblib.dump(fe_list, dump_dir)
        # print("Dumped to", dump_dir)

        # print(fe_list)