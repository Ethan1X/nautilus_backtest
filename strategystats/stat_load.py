# 根据回测订单和价格记录重新生成balance和hedge，并计算指标和画图

import pandas as pd
from .stra_stat import StrategyStatsRecord
# import strategystats
from .stat_func import load_symbol_info
from .stat_data import SymbolType, SymbolInfo, ts_to_millisecond, SymbolBalance
from .stra_stat import plot_from_npz
from .stat_plot import plot_func
import shutil
import numpy as np
# import tomi
import os
import datetime

MODULE_PATH = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(MODULE_PATH, "conf/")
# CONFIG_PATH = "conf/"


# NOTE: 调用initial_balance之前，调用此函数转换初始资金列表
def get_capital_list(init_balances: list[str]):
    capitals = {}
    for balance_str in init_balances:
        _num_str, _token = balance_str.split(" ")
        capitals[_token] = float(_num_str)
    return capitals


def initial_balance(symbol: SymbolInfo, capital_list, start_ts) -> SymbolBalance:
    return SymbolBalance(symbol, capital_list, start_ts)


def load_stra_record(token, load_path, start_time, end_time, starting_balance, cache_path, quote="USDT", exchange="BINANCE", symbol_type="SPOT_NORMAL"):
    # 获取stra_record 
    symbol_info = load_symbol_info(token, quote, getattr(SymbolType, symbol_type), exchange, CONFIG_PATH)
    
    init_balance = initial_balance(symbol_info, starting_balance, start_time.value)

    stra_record = StrategyStatsRecord.from_npz(init_balance, load_path, start_time, end_time)
    
    # indicator_info = stra_record.indicators

    stra_record.statistics(cache_path)

    # os.makedirs(cache_path, exist_ok=True)
    
    # stra_record.to_npz(cache_path)


def load_order_strategy_type(load_path, cache_path, start_time, end_time):
    # 获取order_strategy_type

    start_ts = ts_to_millisecond(start_time.value)
    end_ts = ts_to_millisecond(end_time.value)


    if os.path.exists(f'{load_path}/order_strategy_type.npz'):
        order_strategy_type_npz = np.load(f'{load_path}/order_strategy_type.npz', allow_pickle=True)
        
        order_ts_list, order_id_list, strategy_type_list, strategy_id_list, order_info_list = [], [], [], [], []
        for order_ts, order_id, strategy_type, strategy_id, order_info in zip(order_strategy_type_npz["order_ts"], 
                                           order_strategy_type_npz["order_id"], 
                                           order_strategy_type_npz["strategy_type"], 
                                           order_strategy_type_npz["strategy_id"], 
                                           order_strategy_type_npz["order_info"], 
                                          ):
            if start_ts > order_ts:
                continue
            if end_ts < order_ts:
                break
            order_ts_list.append(order_ts)
            order_id_list.append(order_id)
            strategy_type_list.append(strategy_type)
            strategy_id_list.append(strategy_id)
            order_info_list.append(order_info)
    
        np.savez(f"./{cache_path}/order_strategy_type.npz", order_ts=order_ts_list, 
                 order_id=order_id_list, strategy_type=strategy_type_list, 
                 strategy_id=strategy_id_list, order_info=order_info_list)
    else:
        return


def load_batch_result(batch_file_paths, save_path):
    """
    整合多个回测结果
    :param batch_file_paths: 多个回测结果存储路径
    :param save_path: 结果缓存路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    # file_names = ['market_price.npz', 'orders.npz', 'signal.npz', 'order_strategy_type.npz']
    file_names = ['market_price.npz', 'orders.npz', 'signal.npz',]
    for file_name in file_names:

        files = [os.path.join(f_path, file_name) for f_path in batch_file_paths]

        all_arrays = {}
        
        for file in files:
            if not os.path.exists(file):
                continue
            data = np.load(file, allow_pickle=True)
            for key in data:
                if key not in all_arrays:
                    all_arrays[key] = [data[key]]
                else:
                    all_arrays[key].append(data[key])
            data.close()
    
        combined_data = {key: np.concatenate(arrays) for key, arrays in all_arrays.items()}
    
        np.savez(os.path.join(save_path, file_name), **combined_data)
    return
    

def load_period_result(token, load_path, cache_path, start_time, end_time, starting_balance, plot_config, quote="USDT", exchange="BINANCE", symbol_type="SPOT_NORMAL", plot_path=None, plot_title=None, interval=1, use_s3=False):
    
    """
    从已有回测结果中提取
    :param token: 币种
    :param load_path: 回测结果路径
    :param start_time: 提取开始时间
    :param end_time: 提取结束时间
    :param starting_balance: 指定初始balance
    :param cache_path: 结果缓存路径
    """
    os.makedirs(cache_path, exist_ok=True)
    load_stra_record(token, load_path, start_time, end_time, starting_balance, cache_path, quote, exchange, symbol_type)

    # load_order_strategy_type(load_path, cache_path, start_time, end_time)

    if use_s3:
        plot_func(plot_config, cache_path, start_time, end_time, cache_path, plot_title, interval)
    else:
        period = f'{start_time.strftime("%Y%m%d")}_{end_time.strftime("%Y%m%d")}'
        plot_from_npz(plot_config, cache_path, cache_path, period, plot_title, interval)
        

def load_stra_stat(token: str, load_path: str, cache_path: str, batch_file_paths: list,
                  start_time: pd.Timestamp, end_time: pd.Timestamp, starting_balance: list, plot_config: dict,
                  quote="USDT", exchange="BINANCE", symbol_type="SPOT_NORMAL", plot_path=None, plot_title=None, interval=1, use_s3=False):
    """
    获取指定config下的statistic
    :param token: 币种
    :param load_path: batch result的存储位置
    :param cache_path: statistic的存储位置
    :param start_time: 统计开始时间
    :param end_time: 统计结束时间
    :param starting_balance: 初始balance
    :param plot_config: 画图config
    :param batch_file_paths: 需要整合回测结果的文件路径列表
    """
    # load多组回测结果整合
    load_batch_result(batch_file_paths, load_path)

    # load指定时段结果并统计
    load_period_result(token, load_path, cache_path, start_time, end_time, starting_balance, 
                       plot_config, quote, exchange, symbol_type, plot_path, plot_title, interval, use_s3)
    

