import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_generator import *
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
import joblib
from tqdm import tqdm
import time


class AveAbsRetGenerator(FeatureGenerator):
    # Author: Vijay Huang, 04/01/2024
    # Reviewer: Weijing Yin
    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        self.fe_name = f'ave_abs_ret'
        self.right = config['right']
        self.min_max = (min(self.right), max(self.right))
        self.interval = self.min_max[1] + 10  # 计算区间
        self.min_interval = self.min_max[0] + 1
        self.price_1s = []  # 1秒内的价格列表
        self.price_interval = []  # 每秒的价格列表
        self.unit_sec = 10**9  # 1秒的纳秒数
        self.ts_event = None  # 当前时间
        self.start_time = None  # 每1秒区间的开始时间，从第一个trade时间开始递增1秒
        self.cal_time = None  # 每interval个秒区间的开始时间，从第interval+1秒开始递增1秒

        self.out = False  

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
        if self.start_time is None:
            self.start_time = trade.ts_event
        if self.cal_time is None:
            self.cal_time = trade.ts_event

        # 若新数据小于第一个数据1秒，则存入列表
        if self.ts_event - self.start_time <= self.unit_sec:  # 每1秒内的计算
            self.price_1s.append(trade.price)

        # 新数据大于第一个数据1秒，则计算收益率后存入列表，并将原始单秒列表清空
        else:  
            if self.price_1s:
                if self.price_1s[0] > 0:
                    abs_ret = np.abs(
                        (self.price_1s[-1] - self.price_1s[0]) / self.price_1s[0]
                    )
                else:
                    abs_ret = 0.0
            else:
                abs_ret = np.nan
            self.price_interval.append(abs_ret)
            self.price_1s = [trade.price]

            # 更新首个单秒数据的时间
            self.start_time += self.unit_sec
            while self.ts_event - self.start_time > self.unit_sec:
                self.start_time += self.unit_sec
                self.price_interval.append(0.0)

            self.out = True
            if (self.start_time - self.cal_time) > self.interval * self.unit_sec:
                self.cal_time += self.unit_sec
                try:
                    self.price_interval.pop(0)
                except:
                    pass
            elif (self.start_time - self.cal_time) < self.min_interval * self.unit_sec:
                self.out = False


    def calculate(self):
        if self.out:
            res = []
            for right in self.right:
                if right <= len(self.price_interval):
                    ave_abs_ret = np.mean(self.price_interval[-right:])
                    res.append((
                            f'{self.fe_name}_{right}',
                            self.ts_event,
                            ave_abs_ret,
                        ))
            self.out = False
            return res
        else:
            return None


if __name__ == "__main__":
    initlog(None, "ave_abs_ret_feature.log", logging.INFO)

    begin_time = datetime.datetime(2024, 3, 27, 21, tzinfo=TZ_8)  # 设置时区: tzinfo=TZ_8
    end_time = datetime.datetime(2024, 3, 27, 21, 59, tzinfo=TZ_8)
    exchange = "binance"
    symbol = "btc_usdt"

    config_list = list(FeatureConfig(left=0, right=i) for i in range(5, 10))

    # 计时
    # t1 = time.time()

    for j in tqdm(range(1, 6)):  # range(1, 6)
        ave_abs_ret_config = config_list[j - 1]
        ins = AveAbsRetGenerator(ave_abs_ret_config)
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

        # dump_dir = "ave_abs_ret_2.pkl"
        # joblib.dump(fe_list, dump_dir)
        # print("Dumped to", dump_dir)

    # print(fe_list)
