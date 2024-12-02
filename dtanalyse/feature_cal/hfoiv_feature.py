import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_generator import *
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
import joblib
from tqdm import tqdm
import time

class HFOIVGenerator(FeatureGenerator):
    # Author: Vijay Huang, 04/01/2024
    # Reviewer: Weijing Yin
    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        self.fe_name = f'hfoiv'
        self.interval = config["right"]  # 计算区间
        self.left_start = 0
        self.buy_size = 0  # 每秒内的买量
        self.sell_size = 0  # 每秒内的卖量
        self.buy_size_ls = []  # 每interval秒的买量列表
        self.sell_size_ls = []  # 每interval秒的卖量列表
        self.unit_sec = 10**9  # 1秒的纳秒数
        self.ts_event = None  # 当前时间
        self.start_time = None  # 每1秒区间的开始时间，从第一个trade时间开始递增1秒
        self.cal_time = None  # 每个5秒区间的开始时间，从第6秒开始递增1秒
        self.out = False  #

    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None):
        pass

    def process(
        self, ticker: QuoteTick = None, trade: TradeTick = None, depth: OrderBook = None
    ):
        if trade is None:
            return [
                (self.fe_name, None, None)
            ]

        else:
            self.update(trade=trade)
            feature_ret = self.calculate()

            return feature_ret

    def update(
        self, ticker: QuoteTick = None, trade: TradeTick = None, depth: OrderBook = None
    ):
        # 首次更新时间
        self.ts_event = trade.ts_event
        if self.start_time == None:
            self.start_time = trade.ts_event
        if self.cal_time == None:
            self.cal_time = trade.ts_event

        # 每1秒内的数据计算
        if self.ts_event - self.start_time <= self.unit_sec:
            if trade.aggressor_side == "buy":
                self.buy_size += trade.size
            else:
                self.sell_size += trade.size

        # 超过1秒，并入列表
        else:  # 新数据大于第一个数据1秒，则计算并存入列表；原始数据清空
            self.buy_size_ls.append(self.buy_size)
            self.sell_size_ls.append(self.sell_size)
            self.buy_size, self.sell_size = 0, 0
            if trade.aggressor_side == "buy":
                self.buy_size = trade.size
            else:
                self.sell_size = trade.size

            while self.ts_event - self.start_time > self.unit_sec:
                self.start_time += self.unit_sec

            # 从第6秒后开始，计算每5秒的一次迭代
            self.out = True
            if (self.start_time - self.cal_time) > self.interval[-1] * self.unit_sec and len(self.sell_size_ls) > self.interval[-1]: 
                try:
                    self.buy_size_ls.pop(0)
                    self.sell_size_ls.pop(0)
                except:
                    pass
                self.cal_time += self.unit_sec
            elif (self.start_time - self.cal_time) < self.interval[0] * self.unit_sec:
                self.out = False
            

    def calculate(self):
        if not self.out:
            return None
        res = []
        for interval in self.interval:
            # if interval == 14400:
            #     print(f'hf calc: {interval} {len(self.sell_size_ls)} {len(self.buy_size_ls)}')
            if interval <= len(self.sell_size_ls):
                sell_size = np.array(self.sell_size_ls)[::-1][
                    self.left_start : self.left_start + interval
                ]
                buy_size = np.array(self.buy_size_ls)[::-1][
                    self.left_start : self.left_start + interval
                ]
                hfoiv = np.std(
                    (np.array(buy_size) - np.array(sell_size))
                    / (np.array(buy_size) + np.array(sell_size))
                )
                res.append((
                        f'{self.fe_name}_{interval}',
                        self.ts_event,
                        hfoiv,
                    ))
        self.out = False
        return res


if __name__ == "__main__":
    initlog(None, "hfoiv_feature.log", logging.INFO)

    begin_time = datetime.datetime(2024, 3, 27, 21, tzinfo=TZ_8)  # 设置时区: tzinfo=TZ_8
    end_time = datetime.datetime(2024, 3, 27, 21, 59, tzinfo=TZ_8)
    exchange = "binance"
    symbol = "btc_usdt"

    config_list = list(FeatureConfig(left=0, right=i) for i in range(5, 10))

    # 计时
    # t1 = time.time()
    for j in tqdm(range(1, 6)):  
        hfoiv_config = config_list[j - 1]
        ins = HFOIVGenerator(hfoiv_config)
        fe_list = []

        for idx, row in enumerate(
            get_data_generator(begin_time, end_time, exchange, symbol)
        ):
            # print(idx, row)

            # if idx >= 5000:
            #     break

            if row[1] == "ticker":
                fe_list.append(ins.process(ticker=row[0]))
            elif row[1] == "depth":
                fe_list.append(ins.process(depth=row[0]))
            else:
                fe_list.append(ins.process(trade=row[0]))

        # t2 = time.time()
        # print("Time cost:", t2 - t1)

        # dump_dir = "features/hfoiv/hfoiv_" + str(j + 4) + ".pkl"
        # joblib.dump(fe_list, dump_dir)
        # print("Dumped to", dump_dir)

    # print(fe_list)
