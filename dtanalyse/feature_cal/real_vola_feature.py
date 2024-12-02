import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_generator import *
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
import joblib
from tqdm import tqdm
import time


class Real_Vola_Generator(FeatureGenerator):
    # Author: Vijay Huang, 04/01/2024
    # Reviewer: Xilin Liu
    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        self.right = config['right']
        self.min_max = (min(self.right), max(self.right))
        self.fe_name = f'real_vola'
        self.period = len(self.right)
        self.interval = self.min_max[1] + 10  # 计算区间
        self.left_start = 0
        self.midprice_1s = []
        self.midprice_bar = []
        self.unit_sec = 10**9  # 1秒的纳秒数
        self.ts_event = None
        self.start_time = None  # 每1秒区间的开始时间，从第一个trade时间开始递增1秒
        self.cal_time = None  # 每个5秒区间的开始时间，从第6秒开始递增1秒
        self.out = False  # 判断当前是否可以输出
        self.midprice_bar_sum = [0.0] * self.period

    def set_trade_data(self, load_sum, data_index, price_list_buy=None, volumes_list_buy=None, price_list_sell=None, volumes_list_sell=None):
        pass

    def process(
        self, ticker: QuoteTick = None, trade: TradeTick = None, depth: OrderBook = None
    ):
        if ticker is None:
            return [(self.fe_name, None, None)]

        else:
            self.update(ticker=ticker)
            feature_ret = self.calculate()

            return feature_ret

    def update(
        self, ticker: QuoteTick = None, trade: TradeTick = None, depth: OrderBook = None
    ):

        # 初始化时间参数
        self.ts_event = ticker.ts_event
        if self.start_time == None:
            self.start_time = ticker.ts_event
        if self.cal_time == None:
            self.cal_time = ticker.ts_event

        if self.ts_event - self.start_time <= self.unit_sec:  # 每1秒内的计算
            self.midprice_1s.append((ticker.ask_price + ticker.bid_price) / 2)

        else:
            if self.midprice_1s:
                midprice_pct_change = ((self.midprice_1s[-1] - self.midprice_1s[0])) / (
                    self.midprice_1s[0]
                )
            else:
                midprice_pct_change = np.nan
            self.midprice_bar.append(midprice_pct_change**2)
            
            # calc sum & rolling
            for i in range(len(self.right)):
                right = self.right[i]
                self.midprice_bar_sum[i] += midprice_pct_change**2
                if len(self.midprice_bar) > right:
                    _old_value = self.midprice_bar[-right-1]
                    self.midprice_bar_sum[i] -= _old_value
            self.midprice_1s = [(ticker.ask_price + ticker.bid_price) / 2]    # add by jungle 20240527

            # 更新首个单秒数据的时间
            self.start_time += self.unit_sec
            while self.ts_event - self.start_time > self.unit_sec:
                self.start_time += self.unit_sec
                self.midprice_bar.append(0)

        if (self.start_time - self.cal_time) > self.interval * self.unit_sec:
            try:
                self.midprice_bar.pop(0)
            except:
                pass

            self.cal_time += self.unit_sec
            self.out = True

    def calculate(self):
        res = None
        if self.out:
            res = []

            for i in range(self.period):
                real_vola = None
                right = self.right[i]
                # print(f'calc(real): {right} {self.period} {self.interval}; {len(self.midprice_bar)}')
                if len(self.midprice_bar) >= right:
                    real_vola = self.midprice_bar_sum[i]
                    # real_vola = sum(
                    #     self.midprice_bar[::-1][
                    #         self.left_start : self.left_start + right
                    #     ]
                    # )
                res.append((f'{self.fe_name}_{right}', self.ts_event, real_vola))
            self.out = False
        return res


if __name__ == "__main__":
    initlog(None, "real_vola_feature.log", logging.INFO)

    begin_time = datetime.datetime(2024, 3, 27, 21, tzinfo=TZ_8)  # 设置时区: tzinfo=TZ_8
    end_time = datetime.datetime(2024, 3, 27, 21, 59, tzinfo=TZ_8)
    exchange = "binance"
    symbol = "btc_usdt"

    config_list = list(FeatureConfig(left=0, right=i) for i in range(5, 10))

    # 计时
    # t1 = time.time()
    for j in tqdm(range(5, 10)):
        real_vola_config = config_list[j - 5]

        ins = Real_Vola_Generator(real_vola_config)
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

        # dump_dir = "real_vola.pkl"
        # joblib.dump(fe_list, dump_dir)
        # print("Dumped to", dump_dir)

    # print(fe_list)
