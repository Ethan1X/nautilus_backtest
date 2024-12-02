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
from CNVPL_and_NI_feature import CNVPL_and_NI
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
from label_generator import Price_Change_Rate_Label, Price_Change_Trend_Label

from s3_data_transfer.s3_utils import *
from util.lookbackwnd import *
from TxnImbalance_lbw_Generator import TxnImbalance_lbw_Generator
from PastReturn_lbw_Generator import PastReturn_lbw_Generator

start_time = pd.Timestamp("2024-03-25 00:00:00", tz="HONGKONG")
end_time =  pd.Timestamp("2024-05-02 00:00:00", tz="HONGKONG")
month_str = "2024_0401_0430"
res_path = "/data/jiangzhe/notebooks/feats_res"


class FeatureManagerConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    # data_mode: str = "normal"    
    data_mode: str = "central"
    res_path: str = ""
    fnames = [
        f"TxnImbalance{LBW_TIME_STR}", 
        # f"TxnImbalance{LBW_VOLUME_STR}",
        f"PastReturn{LBW_TIME_STR}",
    ]



class FeatureManager(Strategy):

    def __init__(self, config: FeatureManagerConfig) -> None:
        super().__init__(config)
        print(config)

        self.instrument_id = config.instrument_id
        # print(f'fm_depth: {self.instrument_id}')
        self.token = str(self.instrument_id).split(".")[0].split("USD")[0].lower()
        self.res_path = f'{res_path}/{self.token}_{month_str}/'
        print(f'fm: {self.token} {self.instrument_id} save to: {self.res_path}')
        if not os.path.exists(self.res_path):
            os.makedirs(self.res_path)

        self.data_source = "backtest"
        self.data_mode = config.data_mode
        self.instrument: Instrument | None = None  # Initialized in on_start

        self.feature_gen_list = {}

        # 第一组，采用2^n间隔的不交叉的回望窗口
        # left_list = [5 * 2**(i - 1) if i > 0 else 0 for i in range(6)]
        # right_list = [5 * 2**i for i in range(6)]
        left_list = [5 * 2**(i - 1) if i > 0 else 0 for i in range(0, 13)]
        right_list = [5 * 2**i for i in range(0, 13)]
        fconfig = FeatureConfig(left=left_list, right=right_list, data_mode=self.data_mode)
        # AutoCov
        self.feature_gen_list[f'AutoCov'] = AutoCovGenerator(fconfig)
        # Breadth
        self.feature_gen_list[f'Breadth'] = BreadthGenerator(fconfig)
        # CNVPL_and_NI
        CNVPL_and_NI_config = FeatureConfig(left=left_list, right=right_list, price_level=5, data_mode=self.data_mode)
        self.feature_gen_list[f'CNVPL_and_NI_{CNVPL_and_NI_config["price_level"]}'] = CNVPL_and_NI(CNVPL_and_NI_config)
        # EffectiveTradeSpread
        self.feature_gen_list[f'EffectiveTradeSpread'] = EffectiveTradeSpreadGenerator(fconfig)
        # Lambda
        self.feature_gen_list[f'Lambda'] = LambdaGenerator(fconfig)
        # LobImbalance
        self.feature_gen_list[f'LobImbalance'] = LobImbalanceGenerator(fconfig)
        # OIR
        # oir_left = [0] + [round(i / 10, 1) for i in left_list]
        # oir_right = [0] + [round(i / 10, 1) for i in right_list]
        OIR_config = FeatureConfig(left = left_list, right = right_list)
        self.feature_gen_list[f'OIR'] = OIRGenerator(OIR_config)
        # PastReturn
        self.feature_gen_list[f'PastReturn'] = PastReturnGenerator(fconfig)
        # PastReturn_lbw
        self.feature_gen_list[f'PastReturn{LBW_TIME_STR}'] = PastReturn_lbw_Generator(fconfig, LBW_TYPE_TIME)
        # QuotedSpread
        self.feature_gen_list[f'QuotedSpread'] = QuotedSpreadGenerator(fconfig)
        # TxnImbalance
        self.feature_gen_list[f'TxnImbalance'] = TxnImbalanceGenerator(fconfig)
        # TxnImbalance_lbw
        self.feature_gen_list[f'TxnImbalance{LBW_TIME_STR}'] = TxnImbalance_lbw_Generator(fconfig, LBW_TYPE_TIME)
        self.feature_gen_list[f'TxnImbalance{LBW_VOLUME_STR}'] = TxnImbalance_lbw_Generator(fconfig, LBW_TYPE_VOLUME)
        # Volume
        self.feature_gen_list[f'Volume'] = VolumeGenerator(fconfig)

        # for centralized trade data updating
        self.window_len = int(max(right_list) + 10)
        self.latest_trade_ts = None # 记录当前时间戳
        self.last_trade_ts = None # 记录上一笔trade的时间
        self.load_ts_num = 0
        self.trade_data_index = 0
        self.trade_volumes_buy = np.zeros(self.window_len) * np.nan
        self.trade_price_buy = np.zeros(self.window_len) * np.nan
        self.trade_volumes_sell = np.zeros(self.window_len) * np.nan
        self.trade_price_sell = np.zeros(self.window_len) * np.nan

        self.trade_lbw = {}
        self.trade_lbw[LBW_TYPE_TIME] = LookBackWindow(increament=int(LBW_TIME_TICK), max_scale=int(max(right_list)), type=LBW_TYPE_TIME)
        # self.trade_lbw[LBW_TYPE_VOLUME] = LookBackWindow(increament=int(LBW_VOLUME_TICK), max_scale=int(max(right_list)), type=LBW_TYPE_VOLUME)

        # 第二组，从0开始到i结束的不同回望窗口
        # right_list = [i * 5 for i in range(1, 7)]
        right_list = [5, 10, 30, 60, 120, 300, 600, 1000, 2000, 3600, 7200, 14400]
        # right_list = [7200, 14400]
        fconfig = FeatureConfig(right=right_list)
        # AveAbsRet
        self.feature_gen_list[f'AveAbsRet'] = AveAbsRetGenerator(fconfig)
        # Real_Vola
        self.feature_gen_list[f'Real_Vola'] = Real_Vola_Generator(fconfig)
        # HFOIV
        self.feature_gen_list[f'HFOIV'] = HFOIVGenerator(fconfig)

        # right_list = [5, 10, 30, 60, 120, 300]
        # fconfig = FeatureConfig(right=right_list, data_mode=self.data_mode)
        # CRSI
        self.feature_gen_list[f'CRSI'] = CRSIGenerator(fconfig)
        # TFI
        self.feature_gen_list[f'TFI'] = TFIGenerator(fconfig)

        # right_list = [i for i in range(1, 7)]
        # fconfig = FeatureConfig(right=right_list)
        # Turnover
        # Turnover_config = FeatureConfig(left=left_list, right=right_list)
        self.feature_gen_list[f'Turnover'] = TurnoverGenerator(fconfig)

        # 第三组，一些特殊的参数组合
        # MLOFI
        # MLOFI_config = FeatureConfig(windows=[(0, 2), (2, 10), (0, 30)], M=[i for i in range(1, 6)], data_mode=self.data_mode)
        MLOFI_config = FeatureConfig(windows=[(0, 10), (10, 60), (60, 300), (300, 1000), (1000, 3600), (3600, 14400)], M=[1, 2, 5])
        self.feature_gen_list[f'MLOFI'] = MultiLevelOrderFlowImbalanceGenerator(MLOFI_config)

        # 第四组，无额外参数，计算当前时刻的值
        # EffectiveSpread
        self.feature_gen_list['EffectiveSpread'] = EffectiveSpreadGenerator(FeatureConfig())
        # LOGOI
        self.feature_gen_list['LOGOI'] = LOGOIGenerator(FeatureConfig())
        # OI
        self.feature_gen_list['OI'] = OIGenerator(FeatureConfig())

        # 第五组，Label生成，[0.5, 1, 3]
        # time_period = [0.5, 1, 3]
        time_period = [0.5, 1, 3, 10, 30, 60, 180, 360, 600, 1800, 3600]
        # fconfig = FeatureConfig(time_period=time_period, data_mode=self.data_mode)
        # self.feature_gen_list[f'label_price_change_rate'] = Price_Change_Rate_Label(fconfig)
        # time_period_c = [0.5, 1, 3]
        # ds_c = "S3"
        # PRICE_FLUCTUATION = 1e-4
        # fconfig_c = FeatureConfig(time_period=time_period_c, data_source=ds_c, threshold=PRICE_FLUCTUATION, data_mode=self.data_mode)
        # self.feature_gen_list[f'label_price_change_trend'] = Price_Change_Trend_Label(fconfig_c)        

        if config.fnames is not None and len(config.fnames) > 0:
            self.set_features(config.fnames)
        # print(list(self.feature_gen_list.keys()))
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

        self.set_trade_data(True)
        # todo: depth, ticker
        pass

    def set_features(self, feature_name_list):
        self.feature_gen_list = {k:v for k,v in self.feature_gen_list.items() if k in feature_name_list}

    def set_trade_data(self, first_time=False):
        if self.data_mode != "central":
            return
        _start = time.time()
        price_buy_list = volumes_buy_list = price_sell_list = volumes_sell_list = None
        if first_time:
            price_buy_list = self.trade_price_buy
            volumes_buy_list = self.trade_volumes_buy
            price_sell_list = self.trade_price_sell
            volumes_sell_list = self.trade_volumes_sell
        for key, gen in self.feature_gen_list.items():
            if self.count > 1000 and self.count % 10000 == 0:
                print(f'distribute trade data: {key} {self.load_ts_num}, {self.count}, {self.trade_price_buy[-1]}, {self.trade_price_sell[-1]}; {self.trade_lbw[LBW_TYPE_TIME].container[-1]}, {self.trade_lbw[LBW_TYPE_TIME].last_scale}, {self.trade_lbw[LBW_TYPE_TIME].is_rolling};')
                # {self.trade_lbw[LBW_TYPE_VOLUME].container[-1]}, {self.trade_lbw[LBW_TYPE_VOLUME].last_scale}, {self.trade_lbw[LBW_TYPE_VOLUME].is_rolling}')
            gen.set_trade_data(self.load_ts_num, self.trade_data_index, price_buy_list, volumes_buy_list, price_sell_list, volumes_sell_list, self.trade_lbw)
        self.feature_cost['distribute'] += time.time() - _start
        
    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.instrument_id}")
            self.stop()
            return
        self.subscribe_trade_ticks(self.instrument_id)
        self.subscribe_quote_ticks(self.instrument_id)
        self.subscribe_order_book_at_interval(self.instrument_id, book_type=BookType.L2_MBP, depth=5, interval_ms=100)
        self.trade_lbw[LBW_TYPE_TIME].init_value = int(self.clock.timestamp_ns())
        # self.trade_lbw[LBW_TYPE_VOLUME].init_value = int(0)

    def on_order_book(self, orderbook: OrderBook) ->None:
        self.process(orderbook, 'depth')
    
    def on_quote_tick(self, tick: QuoteTick) -> None:
        data = tick
        if self.data_source == "backtest":
            data = trans_backtest_ticker(tick)
        self.process(data, 'ticker')

    def on_trade_tick(self, trade: TradeTick) -> None:
        _start = time.time()
        data = trans_backtest_trade(trade)
        _end = time.time()
        self.feature_cost['convert'] += _end - _start
        _start = _end

        # data error
        if data.price == 0:
            return

        if self.data_mode == "central":
            # 检查最新的交易数据时间戳是否更新
            if trade is not None:
                _data_index = self.trade_data_index
                if self.last_trade_ts is None or data.ts_event // 1000 != self.last_trade_ts // 1000:
                    self.last_trade_ts = data.ts_event
                    _data_index = (_data_index + 1) % self.window_len
                    if self.load_ts_num == self.window_len:
                        # 新时刻，滚动填充更新
                        if data.aggressor_side == 'buy':
                            self.trade_volumes_buy[_data_index] = data.size
                            self.trade_price_buy[_data_index] = data.price
                            self.trade_volumes_sell[_data_index] = np.nan
                            self.trade_price_sell[_data_index] = np.nan
                        else:
                            self.trade_volumes_sell[_data_index] = data.size
                            self.trade_price_sell[_data_index] = data.price
                            self.trade_volumes_buy[_data_index] = np.nan
                            self.trade_price_buy[_data_index] = np.nan
                    else:
                        self.load_ts_num = self.load_ts_num + 1
                        if data.aggressor_side == 'buy':
                            self.trade_volumes_buy[_data_index] = data.size
                            self.trade_price_buy[_data_index] = data.price
                        else:
                            self.trade_volumes_sell[_data_index] = data.size
                            self.trade_price_sell[_data_index] = data.price
                    self.trade_data_index = _data_index
                else:
                    # 如果是相同时刻的trade，直接更新当前时刻
                    if data.aggressor_side == 'buy':
                        _price = self.trade_price_buy[_data_index]
                        _size = self.trade_volumes_buy[_data_index]
                        if not np.isnan(_price) and not np.isnan(_size):
                            self.trade_price_buy[_data_index] = (_price * _size + data.price * data.size) / (_size + data.size)
                            self.trade_volumes_buy[_data_index] = _size + data.size
                        else:
                            self.trade_price_buy[_data_index] = data.price
                            self.trade_volumes_buy[_data_index] = data.size
                    else:
                        _price = self.trade_price_sell[_data_index]
                        _size = self.trade_volumes_sell[_data_index]
                        if not np.isnan(_price) and not np.isnan(_size):
                            self.trade_price_sell[_data_index] = (_price * _size + data.price * data.size) / (_size + data.size)
                            self.trade_volumes_sell[_data_index] = _size + data.size
                        else:
                            self.trade_price_sell[_data_index] = data.price
                            self.trade_volumes_sell[_data_index] = data.size

                # todo: check for correctness
                self.trade_lbw[LBW_TYPE_TIME].update(_data_index, data.ts_event)
                # self.trade_lbw[LBW_TYPE_VOLUME].update(_data_index, data.price*data.size)
                # print(f'on trade tick: {_data_index}')
                
                self.latest_trade_ts = data.ts_event
            else:
                self.latest_trade_ts = None
        _end = time.time()
        self.feature_cost['trade'] += _end - _start
        
        self.set_trade_data()
        self.process(data, 'trade')
        # self.count += 1

        # todo: check for correctness
        self.trade_lbw[LBW_TYPE_TIME].rolling()
        # self.trade_lbw[LBW_TYPE_VOLUME].rolling()
        
        for key, gen in self.feature_gen_list.items():
            gen.rolling()
            
    def process(self, data, data_type):
        ts_event = data.ts_event
        # print(f'event: {ts_event}  vs  end: {int(self.end_time.timestamp()) * 1e9}')
        if ts_event >= int(self.end_time.timestamp()) * 1e9:
            # 存储一天的数据
            self.save_feature_res()
            self.start_time += pd.Timedelta(days=1)
            self.end_time += pd.Timedelta(days=1)

        for key, gen in self.feature_gen_list.items():
            start = time.time()
            if data_type == 'ticker':
                res = gen.process(ticker=data)
            elif data_type == 'depth':
                res = gen.process(depth=data)
            else:
                res = gen.process(trade=data)
            self.feature_cost[key] += time.time() - start
            # if self.count % 10000 == 0:
            #     print(f'processed({self.count}): {key} {res}')
            if res:
                for r in res:
                    name, tp, value = r[0], r[1], r[2]
                    if value is not None:
                        last_tp = 0
                        if self.feature_res[name]:
                            last_tp = self.feature_res[name][-1][0]
                        if last_tp // (10**8) != tp // (10**8): # ticker trade数据采样至每100ms仅保留一条
                            self.feature_res[name].append((tp, value))
                        else:
                            try:
                                self.feature_res[name][-1] = (tp, value)
                            except Exception as e:
                                print(f'ERROR!!! {key}: {last_tp} / {tp}  {len(self.feature_res)} {e}')

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

    # todo: 划分特征列表和桶，制作对应的列表；从s3上读取depth数据
    bn = BN_TRADE

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
        fnames = [
            # f"TxnImbalance{LBW_TIME_STR}", 
            f"TxnImbalance{LBW_VOLUME_STR}",
        ]
        fm.set_features(fnames)

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
        last_trade = None
        last_index = 0

        print(f'start: {start_ts}')
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
            if bn == BN_TRADE:
                if last_trade is None:
                    while last_index < df.shape[0] and df['tp'].values[last_index] < start_ts * 1000:
                        last_index += 1
                    if last_index < df.shape[0]:
                        last_ts = df['tp'].values[last_index]
                        last_trade = TradeTick(
                            df['m'].values[last_index].lower(),
                            float(df['p'].values[last_index]),
                            float(df['q'].values[last_index]),
                            int(df['tp'].values[last_index] * 1e6)
                        )
                        fm.on_trade_tick(last_trade)
                        last_index += 1
                    else:
                        continue
            elif bn == BN_TICKER:
                pass
            elif bn == BN_DEPTH:
                pass

            _count = 0
            print(f'{date} data({df["tp"].values[last_index]}): begein: {fm.count} {last_index} {_count}  res: {len(fm.feature_res[list(feature_res.keys())[0]])}')
            for i in range(last_index, df.shape[0]):
                _start_point = df['tp'].values[i]
                while df['tp'].values[i] - last_ts > min_period and last_ts <= _start_point + max_period:
                    last_ts += min_period
                    last_trade = TradeTick(
                        df['m'].values[last_index].lower(),
                        float(df['p'].values[last_index]),
                        float(df['q'].values[last_index]),
                        int(df['tp'].values[last_index] * 1e6)
                    )
                    fm.on_trade_tick(last_trade)
                    _count += 1

                trade = TradeTick(
                    df['m'].values[last_index].lower(),
                    float(df['p'].values[last_index]),
                    float(df['q'].values[last_index]),
                    int(df['tp'].values[last_index] * 1e6)
                )
                fm.on_trade_tick(trade)

                _count += 1
                last_trade = trade
                last_ts = df['tp'].values[i]
            print(f'{date} data({last_ts}): end: {fm.count} {df.shape[0]} {_count}  res: {len(fm.feature_res[list(feature_res.keys())[0]])}')

        fm.on_stop()
        print(f'generating features for {exchange}/{symbol} (from {begin_date} to {end_date}) complished... {end_ts}')


