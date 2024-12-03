import sys
import time
import logging
import pandas as pd
import logging
import os
import argparse
import multiprocessing
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal

from nautilus_trader.backtest.node import BacktestNode 
from nautilus_trader.backtest.models import FillModel

from nautilus_trader.core.datetime import dt_to_unix_nanos  
from nautilus_trader.config import BacktestRunConfig, BacktestVenueConfig, BacktestDataConfig, BacktestEngineConfig  
from nautilus_trader.config import ImportableStrategyConfig  
from nautilus_trader.config import LoggingConfig  
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import LatencyModel
from nautilus_trader.model.instruments import *  
from nautilus_trader.model.data import QuoteTick, OrderBookDelta, TradeTick, Bar
from nautilus_trader.model.identifiers import InstrumentId, Venue  
from nautilus_trader.persistence.catalog import ParquetDataCatalog  

from etl.runner import ChronoRunner 
from etl.catalog import ChronoDataCatalog 
from etl.config import ChronoDataConfig 

import strategystats
from strategystats.stat_func import load_symbol_info
from strategystats.stat_data import SymbolType

from utils.log_method import initlog


# if '../' not in sys.path:
#     sys.path.append('../')

# backtest single factor
def run_backtest(config):
    asset = config["token"].lower()
    period = f'{config["start_time"].strftime("%Y%m%d%H:%M")}_{config["end_time"].strftime("%Y%m%d%H:%M")}'
    time_now = pd.Timestamp.fromtimestamp(time.time(), tz="HONGKONG").strftime("%Y%m%d%H%M")
    # log_name = f"{config['token']}_{config['exchange']}_{config['symbol_type']}_{config['strategy_name']}_{config['factor_type']}_{period}-{time_now}.log"
    # initlog('./log', log_name, log_level=logging.INFO)

    is_latency = config['is_latency']
    starting_balance = [f"100 {config['token']}", "1000000 USDT"]

    if config['factor_type'] == 'vinci_reg':
        factors = ['reg_preds_t3.2s_d15', 'reg_preds_t3.2s_d16']

    elif config['factor_type'] == 'vinci_cls':
        factors = ['cls_preds_t3.2s_d0', 'cls_preds_t3.2s_d1']

    elif config['factor_type'] == 'LobImbalance':
        factors = ['LobImbalance_0_5']

    elif config['factor_type'] == 'vinci_maker_label':
        factors = ['maker_cls_up_d3', 'maker_cls_down_d3']

    elif config['factor_type'] == 'vinci_mixer_cls':
        factors = ['cls_preds_t12.8s_d0', 'cls_preds_t12.8s_d1']

    elif config['factor_type'] == 'vinci_mixer_reg':
        factors = ['reg_preds_t12.8s_d0', 'reg_preds_t12.8s_d1']

    elif config['factor_type'] == 'vinci_mixer_reg_label':
        factors = ['reg_preds_t12.8s_d0', 'reg_preds_t12.8s_d1']

    elif config['factor_type'] == 'xuefeng':
        factors = [f'{asset}_pred']

    elif config['factor_type'] == 'vinci_tsmixer_cls':
        factors = ['pred_cls_t3.2s_d0', 'pred_cls_t3.2s_d1']

    elif config['factor_type'] == 'vinci_tsmixer_reg':
        factors = ['pred_reg_t3.2s_d0', 'pred_reg_t3.2s_d1']

    elif config['factor_type'] == 'vinci_maker_label_mixer':
        factors = ['maker_cls_up_d1', 'maker_cls_down_d1']

    elif config['factor_type'] == 'xuefeng_0926':
        factors = ['SignalModel01']
    
    elif config['factor_type'] == 'xuefeng_1016_600s':
        factors = ['return_predict']

    if config['factor_type'] == 'vinci_reg' or config['factor_type'] == 'vinci_cls':
        if config['symbol_type'] == "SPOT_NORMAL":
            factor_path = f'/data/dp-data/vinci_data/ready_for_use/ticker_label_202404_202407_0814/20240501_20240630/{asset}_usdt_binance'

        elif config['symbol_type'] == "SWAP_FOREVER":
            factor_path = f'/data/dp-data/vinci_data/ready_for_use/ticker_label_202404_202407_0814/20240501_20240630/{asset}_usdt_binance_perp'

    elif config['factor_type'] == 'LobImbalance':
        if config['symbol_type'] == "SPOT_NORMAL":
            factor_path = f'/data/dp-data/ModelDatas/2023_07_2024_07/{asset}_usdt_binance'

        elif config['symbol_type'] == "SWAP_FOREVER":
            factor_path = f'/data/dp-data/ModelDatas/2023_07_2024_07/{asset}_usdt_uswap_binance'

    elif config['factor_type'] == 'vinci_maker_label':
        if config['symbol_type'] == "SPOT_NORMAL":
            factor_path = f'/data/dp-data/vinci_data/ready_for_use/maker_label_202404_202405_0910/20240401_20240531/{asset}_usdt_binance'

        elif config['symbol_type'] == "SWAP_FOREVER":
            factor_path = f'/data/dp-data/vinci_data/ready_for_use/maker_label_202404_202405_0910/20240401_20240531/{asset}_usdt_binance_perp'

    elif config['factor_type'] == 'vinci_mixer_reg' or config['factor_type'] == 'vinci_mixer_cls':
        if config['symbol_type'] == "SPOT_NORMAL":
            factor_path = f'/data/dp-data/vinci_data/ready_for_use/ticker_label_itranFUFMixer_202404_0913/20240401_20240630/{asset}_usdt_binance'

        elif config['symbol_type'] == "SWAP_FOREVER":
            factor_path = f'/data/dp-data/vinci_data/ready_for_use/ticker_label_itranFUFMixer_202404_0913/20240401_20240630/{asset}_usdt_binance_perp'  

    elif config['factor_type'] == 'xuefeng':
        if config['symbol_type'] == "SPOT_NORMAL":
            factor_path = f'/data/dp-data/xuefeng_data/ready_for_use/baseline_20240913/20240401_20240630/{asset}_usdt_binance'

        elif config['symbol_type'] == "SWAP_FOREVER":
            factor_path = f'/data/dp-data/xuefeng_data/ready_for_use/baseline_20240913/20240401_20240630/{asset}_usdt_binance_perp'
            
    elif config['factor_type'] == 'vinci_tsmixer_reg' or config['factor_type'] == 'vinci_tsmixer_cls':
        if config['symbol_type'] == "SPOT_NORMAL":
            factor_path = f'/data/dp-data/vinci_data/ready_for_use/ticker_label_TSMixer_btc_202404/20240401_20240630/{asset}_usdt_binance'

        elif config['symbol_type'] == "SWAP_FOREVER":
            factor_path = f'/data/dp-data/vinci_data/ready_for_use/ticker_label_TSMixer_btc_202404/20240401_20240630/{asset}_usdt_binance_perp'

    elif config['factor_type'] == 'vinci_maker_label_mixer':
        if config['symbol_type'] == "SPOT_NORMAL":
            factor_path = f'/data/dp-data/vinci_data/ready_for_use/maker_label_itranFUFMixer_202403_202404_0926/20240301_20240430/{asset}_usdt_binance'

        elif config['symbol_type'] == "SWAP_FOREVER":
            factor_path = f'/data/dp-data/vinci_data/ready_for_use/maker_label_itranFUFMixer_202403_202404_0926/20240301_20240430/{asset}_usdt_binance_perp'

    elif config['factor_type'] == 'xuefeng_0926':
        if config['symbol_type'] == "SPOT_NORMAL":
            factor_path = f'/data/dp-data/xuefeng_data/ready_for_use/Test202404_Submit20240926/20240401_20240630/{asset}_usdt_binance'

        elif config['symbol_type'] == "SWAP_FOREVER":
            factor_path = f'/data/dp-data/xuefeng_data/ready_for_use/Test202404_Submit20240926/20240401_20240630/{asset}_usdt_binance_perp'
    
    elif config['factor_type'] == 'xuefeng_1016_600s':
        if config['symbol_type'] == "SPOT_NORMAL":
            factor_path = f'/data/dp-data/xuefeng_data/ready_for_use/600s_stagebaseline_202404_1016/20240401_20240630/{asset}_usdt_binance'

        elif config['symbol_type'] == "SWAP_FOREVER":
            factor_path = f'/data/dp-data/xuefeng_data/ready_for_use/Test202404_Submit20240926/20240401_20240630/{asset}_usdt_binance_perp'

    # factor optimal holding time(s)
    holding_time = 10

    # factor threshold
    # # taker long if achieve above value and taker short if achieve below value
    # if config['factor_type'] == 'vinci_reg':
    #     threshold = {'above': config['threshold_value'], 'below': -config['threshold_value']}
    # elif config['factor_type'] == 'vinci_cls':
    #     threshold = {'above': config['threshold_value'], 'below': config['threshold_value']}
    #     # threshold = {'above': 0.02, 'below': 0.02}
    # elif config['factor_type'] == 'vinci_maker_label':
    #     threshold = {'above': config['threshold_value'], 'below': config['threshold_value']}
    # elif config['factor_type'] == 'LobImbalance':
    #     threshold = {'above': '97%', 'below': '3%'}
    # elif config['factor_type'] == 'vinci_mixer_reg':
    #     threshold = {'above': config['threshold_value'], 'below': -config['threshold_value']}
    # elif config['factor_type'] == 'vinci_mixer_cls':
    #     threshold = {'above': config['threshold_value'], 'below': config['threshold_value']}
    # elif config['factor_type'] == 'xuefeng' or config['factor_type'] == 'xuefeng_0926':
    #     threshold = {'above': config['threshold_value'], 'below': -config['threshold_value']}
    # elif config['factor_type'] == 'vinci_tsmixer_reg':
    #     threshold = {'above': config['threshold_value'], 'below': -config['threshold_value']}
    # elif config['factor_type'] == 'vinci_tsmixer_cls':
    #     threshold = {'above': config['threshold_value'], 'below': config['threshold_value']}
    # elif config['factor_type'] == 'vinci_maker_label_mixer':
    #     threshold = {'above': config['threshold_value'], 'below': config['threshold_value']}

    if config['factor_type'] == 'LobImbalance':
        threshold = {'above': '97%', 'below': '3%'}

    elif 'reg' in config['factor_type'] or 'xuefeng' in config['factor_type']:
        threshold = {'above': config['threshold_value'], 'below': -config['threshold_value']}

    else:
        threshold = {'above': config['threshold_value'], 'below': config['threshold_value']}

    if 'vinci' in config['factor_type'] or 'xuefeng' in config['factor_type']:
        threshold_data = pd.DataFrame()
    else:
        if config['symbol_type'] == "SPOT_NORMAL":
            threshold_data = pd.read_csv(f'../distribution_result/{asset}_usdt_binance/2024_0501_0630/threshold_data.csv')

        elif config['symbol_type'] == "SWAP_FOREVER":
            threshold_data = pd.read_csv(f'../distribution_result/{asset}_usdt_uswap_binance/2024_0501_0630/threshold_data.csv')

    # plot config
    plot_config = {
        "net_value": True,
        "position": True,
        "stat_metrics": True,
        # "order_mark": {
        #     "type_list": ["buy_to_open", "sell_to_close", "sell_to_open", "buy_to_close"]
        # },
        "holding_spot_value": False,
    }
    # # symbol config
    # MODULE_PATH = os.path.dirname(strategystats.__file__)
    # CONFIG_PATH = os.path.join(MODULE_PATH, "conf/") 
    CONFIG_PATH = os.path.join(os.getcwd(), 'strategystats', 'conf')
    symbol_info = load_symbol_info(config['token'], "USDT", getattr(SymbolType, config['symbol_type']), config['exchange'], CONFIG_PATH)
    symbol_config = f"{config['token']}USDT.{config['exchange']}"

    main_dir = f"./results/{config['save_fold']}/{symbol_config}_{config['symbol_type']}"
    os.makedirs(main_dir, exist_ok=True)
    save_path = os.path.join(main_dir, f"{config['strategy_name']}{config['latency_name']}_{period}_{config['factor_type']}_{threshold['above']}_{threshold['below']}_{config['stop_loss_type']}_{config['stop_loss_taker_threshold']}_{config['threshold_value']}")
    os.makedirs(save_path, exist_ok=True)

    catalog = ChronoDataCatalog("./catalog", show_query_paths=False)
    
    # NOTE: instrument is the trading object for your strategy
    # instrument = catalog.clone_instrument("/data/catalog", CurrencyPair, "BTCUSDT.BINANCE", maker_fee="-0.00005", taker_fee="0.00018")
    if symbol_info.symbol_type == SymbolType.SPOT_NORMAL:
        if config["set_instrument"] == "yes": 
            instrument = catalog.clone_instrument("/data/catalog", CurrencyPair, 
                                                symbol_config, rewrite=True,
                                                size_increment="0.00001", size_precision=5,
                                                maker_fee="-0.00006", taker_fee="0.00000",
                                                min_quantity="0.00001",
                                                min_notional=f"0.00001 {symbol_info.token}",
                                                # price_precision=2,
                                                # price_increment="0.01",
                                                )
        else:
            instrument = catalog.clone_instrument("/data/catalog", CurrencyPair, symbol_config)

    elif symbol_info.symbol_type == SymbolType.SWAP_FOREVER:
        if config["set_instrument"] == "yes": 
            instrument = catalog.clone_instrument("/data/catalog", CryptoPerpetual, 
                                                f'{symbol_config}_PERP', rewrite=True,
                                                size_increment="0.00001", size_precision=5,
                                                maker_fee="-0.00006", taker_fee="0.00018",
                                                min_quantity="0.00001",
                                                min_notional=f"0.00001 {symbol_info.token}",
                                                # price_precision=2,
                                                # price_increment="0.01",
                                                )
        else:
            instrument = catalog.clone_instrument("/data/catalog", CryptoPerpetual, f'{symbol_config}_PERP')

    print(f'instrument: {instrument}')

    data_configs = [
        # NOTE: here are the types of market datas that your startegy will use
        # ChronoDataConfig(
        #     catalog_path=str(catalog.path),
        #     data_cls=OrderBookDelta,
        #     instrument_id=instrument.id,
        #     start_time=start_time,
        #     end_time=end_time,
        #     file_batch_size=5,
        # ),
        ChronoDataConfig(
            catalog_path=str(catalog.path),
            data_cls=QuoteTick,
            instrument_id=instrument.id,
            start_time=config['start_time'],
            end_time=config['end_time'],
            file_batch_size=5,
        ),
        # ChronoDataConfig(
        #     catalog_path=str(catalog.path),
        #     data_cls=TradeTick,
        #     instrument_id=instrument.id,
        #     start_time=start_time,
        #     end_time=end_time,
        #     file_batch_size=5,
        # )
    ]
    
    book_type = "L1_MBP"  # Ensure data book type matches venue book type

    if symbol_info.symbol_type == SymbolType.SPOT_NORMAL:
        name = symbol_info.market_info.exchange
        account_type = "CASH"
    elif symbol_info.symbol_type == SymbolType.SWAP_FOREVER:
        name = f'{symbol_info.market_info.exchange}_PERP'
        account_type = "MARGIN"

    print(f'account_type: {account_type}')
        
    venues_configs = [
        # NOTE: here are your starting account(s) and assets in account(s) for backtesting
        # NOTE: "MARGIN" for trades with leverate (eg: Future contract); "CASH" for trades with Spot
        BacktestVenueConfig(
            name=name,
            oms_type="NETTING",
            account_type=account_type,
            base_currency=None,
            starting_balances=starting_balance,
            book_type=book_type, # Venues book type
            )
    ]

    strategy_path = f"{config['strategy_name']}:MyStrategy"
    config_path = f"{config['strategy_name']}:MyStrategyConfig"

    print(f"strategy_path: {strategy_path} | config_path: {config_path}")
    strategies = [
        ImportableStrategyConfig(
            strategy_path=strategy_path,
            config_path=config_path,
            config=dict(
                instrument_id=instrument.id,
                starting_balance=starting_balance,
                start_time=config['start_time'],
                end_time=config['end_time'],
                save_path=save_path,
                factor_type=config['factor_type'],
                factors=factors,
                threshold=threshold,
                factor_path=factor_path,
                threshold_data=threshold_data,
                holding_time=holding_time,
                order_amount=config['order_amount'],
                plot_config=plot_config,
                symbol_info=id(symbol_info),
                stop_loss_rate=config['stop_loss_rate'],
                stop_win_rate=config['stop_win_rate'],
                price_interval=config['price_interval'],
                stop_loss_taker_threshold=config['stop_loss_taker_threshold'],
                stop_loss_type=config['stop_loss_type'],
            ),
        ),
    ]

    bt_config = BacktestRunConfig(
        engine=BacktestEngineConfig(
            trader_id=f"{symbol_info.token}-{config['strategy_name']}", #"BACKTESTER-001",
            strategies=strategies,
            logging=LoggingConfig(log_level="ERROR", log_level_file="WARN", log_file_name=config['log_name'], log_directory="./log/"),
        ),
        data=data_configs,
        venues=venues_configs,
    )

    latency_model = {
        f"{config['exchange']}": LatencyModel(
                    base_latency_nanos = 0,
                    insert_latency_nanos = 3*1e6,# nanoseconds.
                    update_latency_nanos = 0,
                    cancel_latency_nanos = 3*1e6,
        )
    }
    # 设置成交概率，延迟模型参数的样例
    fill_models = {
        f"{config['exchange']}": FillModel(prob_fill_on_limit=1, prob_fill_on_stop=0.5, prob_slippage=0.0)
    }
    

    # print(f"strategy_name: {config['strategy_name']}")
    print(f"latency: {is_latency}")

    if is_latency:
        node = ChronoRunner(configs=[bt_config],
                            latency_models=latency_model,
                            fill_models=fill_models,
                        )
    else:
        node = ChronoRunner(configs=[bt_config], fill_models=fill_models,)
    
    
    result = node.run()
    print(f'result: {result}')

    # Generate reports
    engine: BacktestEngine = node.get_engine(bt_config.id)
    order_fills_report = engine.trader.generate_order_fills_report()
    positions_report = engine.trader.generate_positions_report()
    account_report = engine.trader.generate_account_report(Venue("BINANCE"))

    # Save reports to files
    order_fills_report.to_csv(os.path.join(save_path, "order_fills_report.csv"), index=False)
    positions_report.to_csv(os.path.join(save_path, "positions_report.csv"), index=False)
    account_report.to_csv(os.path.join(save_path, "account_report.csv"), index=False)
    print(f"Reports saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default="BTC")
    parser.add_argument('--exchange', type=str, default="BINANCE")
    parser.add_argument('--symbol_type', type=str, default="SPOT_NORMAL")
    parser.add_argument('--start_str', type=str, default="2024-04-02T00:00")
    parser.add_argument('--end_str', type=str, default="2024-04-30T00:00")
    parser.add_argument('--strategy_name', type=str, default="taker_stra")  # 'maker' 'taker'
    parser.add_argument('--is_latency', type=str, default='no')
    parser.add_argument('--stop_win_rate', type=float, default=0.0002)
    parser.add_argument('--stop_loss_rate', type=float, default=0.0001)
    parser.add_argument('--factor_type', type=str, default="xuefeng_1016_600s")  # 'vinci_reg' 'vinci_cls' 'LobImbalance' 'vinci_maker_label'
    parser.add_argument('--order_amount', type=float, default=0.0001)  # 最小下单精度
    parser.add_argument('--set_instrument', type=str, default='no')  # 是否设置精度，初始化某个币对或者修改交易手续费时需要更新
    parser.add_argument('--save_fold', type=str, default='xuefeng_1016_600s')  # 结果保存位置
    parser.add_argument('--price_interval', type=float, default=1000*60*10)  # price记录间隔（默认1ms）
    parser.add_argument('--stop_loss_taker_threshold', type=float, default=0)  # taker止损阈值
    parser.add_argument('--stop_loss_type', type=str, default='maker')  # 止损类型说明 maker / NONE / taker
    parser.add_argument('--threshold_value', type=float, default=0.00001)  # 信号阈值

    args, _ = parser.parse_known_args()
    config = vars(args)
    config['start_time'] = pd.Timestamp(args.start_str, tz="HONGKONG")
    config['end_time'] = pd.Timestamp(args.end_str, tz="HONGKONG")
    config['is_latency'] = True if config['is_latency'] == 'yes' else False
    config['latency_name'] = '_latency' if config['is_latency'] else ''

    # period = f'{config["start_time"].strftime("%Y%m%d%H:%M")}_{config["end_time"].strftime("%Y%m%d%H:%M")}'
    # time_now = pd.Timestamp.fromtimestamp(time.time(), tz="HONGKONG").strftime("%Y%m%d%H%M")
    # log_name = f"{config['token']}_{config['exchange']}_{config['symbol_type']}_{config['strategy_name']}{config['latency_name']}_{config['factor_type']}_{config['stop_loss_type']}_{config['stop_loss_taker_threshold']}_{config['threshold_value']}_{period}-{time_now}.log"
    # initlog('./log', log_name, log_level=logging.DEBUG)
    # config['log_name'] = log_name

    # print(config)

    # run_backtest(config)

    parameter_list = []
    start_time = pd.Timestamp(args.start_str, tz="HONGKONG")
    end_time = pd.Timestamp(args.end_str, tz="HONGKONG")

    period = f'{start_time.strftime("%Y%m%d%H:%M")}_{end_time.strftime("%Y%m%d%H:%M")}'
    time_now = pd.Timestamp.fromtimestamp(time.time(), tz="HONGKONG").strftime("%Y%m%d%H%M")
    # log_name = f"{config['token']}_{config['exchange']}_{config['symbol_type']}_{config['strategy_name']}_{config['factor_type']}_{period}-{time_now}.log"
    log_name = f"{config['token']}_{config['exchange']}_{config['symbol_type']}_{config['strategy_name']}{config['latency_name']}_{config['factor_type']}_{config['stop_loss_type']}_{config['stop_loss_taker_threshold']}_{period}-{time_now}.log"
    
    initlog('./log', log_name, log_level=logging.INFO)
    config['log_name'] = log_name

    freq = 1
    for d in pd.date_range(start=start_time, end=end_time+timedelta(days=freq), freq=f'{freq}D'):
        start = d
        end = min(start + timedelta(days=freq), end_time)
        if start < end:
            print(start, end)
            config['start_time'] = start
            config['end_time'] = end
            parameter_list.append(config.copy())

    print(f'parameter_list: {parameter_list}')
    print(f'--------------- len(parameter_list): {len(parameter_list)}')

    def run_backtest_process(config):
        run_backtest(config)

    pool = multiprocessing.Pool(processes=7)
    try:
        results = pool.map(run_backtest_process, parameter_list)
    finally:
        pool.close()
        pool.join()