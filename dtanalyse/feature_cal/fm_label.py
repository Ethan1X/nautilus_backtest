import os, sys, logging, time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.message import Event
# from nautilus_trader.core.message import OrderEvent
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.data import OrderBookDeltas
from nautilus_trader.model.enums import BookType
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.events import PositionChanged
from nautilus_trader.model.events import PositionClosed
from nautilus_trader.model.events import PositionOpened
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import StrategyId
from nautilus_trader.model.identifiers import AccountId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.objects import Currency
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import PositionSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.enums import TriggerType

from util.load_s3_data import LoadS3Data
from util.time_method import *
from data_generator import *
from collections import defaultdict
from feature import FeatureGenerator, FeatureConfig
from util.hedge_log import initlog
from EffectiveSpread_feature import EffectiveSpreadGenerator
from MultiLevelOrderFlowImbalance_feature import MultiLevelOrderFlowImbalanceGenerator
from CRSI_feature import CRSIGenerator
from TFI_feature import TFIGenerator
from CNVPL_and_NI_feature import CNVPL_and_NI
from hfoiv_feature import HFOIVGenerator
from real_vola_feature import Real_Vola_Generator
from turnover_feature import TurnoverGenerator
from ave_abs_ret_feature import AveAbsRetGenerator
from OI_feature import OIGenerator
from OIR_feature import OIRGenerator
from LOGOI_feature import LOGOIGenerator
from Breadth_feature import BreadthGenerator
from VolumeGenerator import VolumeGenerator
from LambdaGenerator import LambdaGenerator
from TxnImbalanceGenerator import TxnImbalanceGenerator
from PastReturnGenerator import PastReturnGenerator
from AutoCovGenerator import AutoCovGenerator
from EffectiveTradeSpreadGenerator import EffectiveTradeSpreadGenerator
from LobImbalanceGenerator import LobImbalanceGenerator
from QuotedSpreadGenerator import QuotedSpreadGenerator
# from label_generator import Price_Change_Rate_Label, Price_Change_Trend_Label
from for_labeling import Price_Change_Rate_Label, Price_Change_Trend_Label

from s3_data_transfer.s3_utils import *


start_time = pd.Timestamp("2024-03-15 00:00:00", tz="HONGKONG")
end_time =  pd.Timestamp("2024-06-01 00:00:00", tz="HONGKONG")


class FeatureManagerConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    data_mode: str = "normal"    
    # data_mode: str = "central"
    res_path: str = ""


class FeatureManager(Strategy):

    def __init__(self, config: FeatureManagerConfig) -> None:
        super().__init__(config)
        print(config)

        self.res_path = f'feature_res_midterm/'
        # if "res_path" in config.keys():
        #     self.res_path = config['res_path']

        self.data_source = "backtest"
        self.data_mode = config.data_mode
        self.instrument_id = config.instrument_id
        self.instrument: Instrument | None = None  # Initialized in on_start

        self.feature_gen_list = {}

        # 第五组，Label生成，[0.5, 1, 3]
        # time_period = [0.5, 1, 3]
        time_period = [0.5, 1, 3, 10, 30, 60, 180, 360, 600, 1800, 3600]
        config = FeatureConfig(time_period=time_period, data_mode=self.data_mode)
        self.feature_gen_list[f'label_price_change_rate'] = Price_Change_Rate_Label(config)
        # time_period_c = [0.5, 1, 3]
        ds_c = self.data_source
        PRICE_FLUCTUATION = 1e-4
        config_c = FeatureConfig(time_period=time_period, data_source=ds_c, threshold=PRICE_FLUCTUATION, data_mode=self.data_mode)
        # self.feature_gen_list[f'label_price_change_trend'] = Price_Change_Trend_Label(config_c)        

        self.feature_res = defaultdict(list)
        self.feature_cost = defaultdict(float)
        self.count = 0
        self.start_time = start_time
        self.end_time = start_time + pd.Timedelta(days=1)
        for key in self.feature_gen_list.keys():
            self.feature_cost[key] = 0.0
        self.feature_cost['trade'] = 0.0
        self.feature_cost['convert'] = 0.0
        self.feature_cost['distribute'] = 0.0

        # todo: depth, ticker
        pass
        
    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.instrument_id}")
            self.stop()
            return
        # self.subscribe_trade_ticks(self.instrument_id)
        self.subscribe_quote_ticks(self.instrument_id)
        # self.subscribe_order_book_snapshots(self.instrument_id, book_type=BookType.L2_MBP, depth=5, interval_ms=100)

    def on_order_book(self, orderbook: OrderBook) ->None:
        self.process(orderbook, 'depth')
    
    def on_quote_tick(self, tick: QuoteTick) -> None:
        data = tick
        if self.data_source == "backtest":
            data = trans_backtest_ticker(tick)
        # print(f'ticker now: {data}\n')
        self.process(data, 'ticker')
            
    def process(self, data, data_type):
        ts_event = data.ts_event
        # print(f'event: {ts_event}  vs  end: {int(self.end_time.timestamp()) * 1e9}')
        # if ts_event >= int(self.end_time.timestamp()) * 1e9:
        #     # 存储一天的数据
        #     self.save_feature_res()
        #     self.start_time += pd.Timedelta(days=1)
        #     self.end_time += pd.Timedelta(days=1)

        for key, gen in self.feature_gen_list.items():
            start = time.time()
            if data_type == 'ticker':
                res = gen.process(ticker=data)
                # if res is None:
                #     print(f'res is none(ticker): {data_type} {data} {res} {self.count} {ts_event}')
            elif data_type == 'depth':
                res = gen.process(depth=data)
                # if res is None:
                #     print(f'res is none(depth): {data_type} {res} {self.count} {ts_event}')
            else:
                res = gen.process(trade=data)
                # if res is None:
                #     print(f'res is none(trade): {data_type} {res} {self.count} {ts_event}')
            self.feature_cost[key] += time.time() - start
            # if self.count % 10000 == 0:
            #     print(f'processed({self.count}): {key} {res}')
            if res:
                ts_event = res[0][1]
                if ts_event >= int(self.end_time.timestamp()) * 1e9:
                    # 存储一天的数据
                    print(f'save file: event: {ts_event}  vs  end: {int(self.end_time.timestamp()) * 1e9}({self.end_time})')
                    self.save_feature_res()
                    self.start_time += pd.Timedelta(days=1)
                    self.end_time += pd.Timedelta(days=1)

                for r in res:
                    name, tp, value = r[0], r[1], r[2]
                    # if value is not None:
                    if True:
                        last_tp = 0
                        if self.feature_res[name]:
                            last_tp = self.feature_res[name][-1][0]
                        
                        # if last_tp // (10**8) != tp // (10**8): # ticker trade数据采样至每100ms仅保留一条
                        #     self.feature_res[name].append((tp, value))
                        # else:
                        #     self.feature_res[name][-1] = (tp, value)
                        self.feature_res[name].append((tp, value))
            # else:
            #     print(f'res is none: {res} {self.count} {ts_event}')
            if res is not None:
                if self.count % 100000 == 0:
                    print(f'finish {self.count} {len(res)} {self.feature_res.keys()}')
            # else:
            #     if self.count % 10000 == 0:
            #         print(f'has no res yet: {key} {self.count}/{self.window_len}')

        self.count += 1
    
    def save_feature_res(self):
        start = self.start_time.strftime(DATETIME_FORMAT7)
        end = self.end_time.strftime(DATETIME_FORMAT7)
        if self.feature_res is None or len(self.feature_res) == 0:
            return
            
        print(f'feature res: {len(self.feature_res)}')
        for name, v in self.feature_res.items():
            path = f'{self.res_path}{name}_{start}.csv'
            print(f'\n======================save results {start}-{end}: {path}\n')
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    f.write(f'time,{name}')
            with open(path, 'a') as f:
                for tp, value in v:
                    f.write(f'\n{tp},{value}')
                    # print(f'{tp},{value}')
        self.feature_res.clear()
        for k, v in self.feature_cost.items():
            print(f'{k},{v}')


    def on_stop(self) -> None:
        """
        Actions to be performed when the strategy is stopped.
        """
        print(f'on stop: feature res: {len(self.feature_res)}')
        if self.feature_res:
            self.save_feature_res()
        print(f'process: {self.count}')


if __name__ == "__main__":    
    res_root_path = "/data/jiangzhe/notebooks/feats_res"
    date_path = "_2023_0701_0731"
    begin_date = "2023062923"
    end_date = "2023080200"
    start_time = pd.Timestamp("2023-06-30 00:00:00", tz="HONGKONG")
    end_time =  pd.Timestamp("2023-08-01 00:00:00", tz="HONGKONG")
    start_ts = start_time.timestamp()
    end_ts = end_time.timestamp()
    min_period = 100    # 毫秒
    max_period = 30 * 1000    # 毫秒，30秒
    exchange = 'binance'
    print(f'from {start_time}({start_ts}) to {end_time}, with min period {min_period} and max peirod {max_period}')
    
    symbol_list = [ "",
                    "btc_usdt_uswap", "eth_usdt_uswap",   
                    "sol_usdt_uswap", "op_usdt_uswap", "fil_usdt_uswap", "matic_usdt_uswap",
                    "bnb_usdt_uswap", "avax_usdt_uswap", "bch_usdt_uswap", "arb_usdt_uswap",
                    "btc_usdt", "eth_usdt", 
                    "sol_usdt", "op_usdt", "fil_usdt",
                    "bnb_usdt", "avax_usdt", "bch_usdt",
                    "arb_usdt", "matic_usdt",
    ]
    
    postfix = '.log.gz'
    bn = BN_TICKER

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('symbol_set', type=str, help='numbers')
    args = parser.parse_args()
    symbol_set = int(args.symbol_set)
    if symbol_set < len(symbol_list):
        symbol = symbol_list[symbol_set]
        # symbol_str = f'{exchange}_{symbol}_'
        symbol_str = ""

        fmc = FeatureManagerConfig(StrategyConfig())
        fm = FeatureManager(fmc)

        token = symbol.split("_")[0]
        res_path = f'{res_root_path}/{token}{date_path}'
        res_path = f'{res_path}/{symbol_str}'
        fm.res_path = res_path
        fm.data_source = "S3L"
    
        date_list = generate_date_list(begin_date, end_date)
        
        # fn = f'{date_list[0]}/{exchange}/{symbol}{postfix}'
        # print(f'load s3 data for:   {bn}:{fn}')
        # df = load_data_source_from_s3(fn, bn)
        # print(f'{df.keys()}')
        last_ts = 0
        last_ticker = None
        last_index = 0

        print(f'start: {start_ts}')
        last_ticker = None
        last_ts = 0
        for date in date_list:
            fn = f'{date}/{exchange}/{symbol}{postfix}'
            print(f'load s3 data for:   {bn}:{fn}')
            df = load_data_source_from_s3(fn, bn)

            # if df is None:
            #     fn = f'/data/jupyter/jungle/dtanalyse/feature_cal/s3_data/{date}_{symbol}.log'
            #     try:
            #         df = pd.read_json(fn, lines=True)
            #     except Exception as e:
            #         print(f'get data error: {fn} for {e}')

            # df.to_csv(f'{res_root_path}/{date}.csv')
            # continue
            
            # print(f'{df.keys()}')
            last_index = 0
            if last_ticker is None:
                while last_index < df.shape[0] and df['tp'].values[last_index] < start_ts * 1000:
                    last_index += 1
                if last_index < df.shape[0]:
                    last_ts = df['tp'].values[last_index]
                    last_ticker = QuoteTick(
                        df['ap'].values[last_index],
                        df['aa'].values[last_index],
                        df['bp'].values[last_index],
                        df['ba'].values[last_index],
                        int(df['tp'].values[last_index] * 1e6)
                    )
                    fm.on_quote_tick(last_ticker)
                    last_index += 1
                else:
                    continue

            _count = 0
            print(f'{date} data({df["tp"].values[last_index]}): begein: {fm.count} {last_index} {_count}  res: {len(fm.feature_res["price_change_rate_1s"])}')
            for i in range(last_index, df.shape[0]):
                _start_point = df['tp'].values[i]
                while df['tp'].values[i] - last_ts > min_period and last_ts <= _start_point + max_period:
                    last_ts += min_period
                    last_ticker = QuoteTick(
                        last_ticker.ask_price,
                        last_ticker.ask_size,
                        last_ticker.bid_price,
                        last_ticker.bid_size,
                        int(last_ts * 1e6)
                    )
                    fm.on_quote_tick(last_ticker)
                    _count += 1

                ticker = QuoteTick(
                    df['ap'].values[i],
                    df['aa'].values[i],
                    df['bp'].values[i],
                    df['ba'].values[i],
                    int(df['tp'].values[i] * 1e6)
                )
                # fm.on_quote_tick(tmp_trans_s3_backtest_ticker(ticker))
                fm.on_quote_tick(ticker)
                _count += 1
                last_ticker = ticker
                last_ts = df['tp'].values[i]
            print(f'{date} data({last_ts}): end: {fm.count} {df.shape[0]} {_count}  res: {len(fm.feature_res["price_change_rate_1s"])}')

        fm.on_stop()
        print(f'labeling for {exchange}/{symbol} (from {begin_date} to {end_date}) complished... {end_ts}')



