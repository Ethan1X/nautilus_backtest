# ***** 整合结果 *****

import pandas as pd
from strategystats.stat_load import load_stra_stat, load_period_result, get_capital_list
from strategystats.stat_analysis import *


def load_batch_file_paths(start_time: pd.Timestamp, end_time: pd.Timestamp, base_path: str, factor_name: str, strategy_name="maker_stra", batch_period_days=7):
    # 获取需要整合回测结果的文件路径列表
    batch_file_paths = []

    for d in pd.date_range(start=start_time, end=end_time+datetime.timedelta(days=1), freq=f'{batch_period_days}D'):
        # start = max(d, start_time)
        start = d
        end = min(d + datetime.timedelta(days=batch_period_days), end_time)
        batch_file_paths.append(f'{base_path}/{strategy_name}_{start.strftime("%Y%m%d%H:%M")}_{end.strftime("%Y%m%d%H:%M")}_{factor_name}')

    return batch_file_paths


# symbol信息和初始balance
token = "BTC"
quote = "USDT"
exchange = "BINANCE"
symbol_type = "SPOT_NORMAL"
starting_balance = [f"10 {token}", f"10 {quote}"]
factor_name = "xuefeng_0926_1e-05_-1e-05_maker_0_1e-05"
strategy_name = "taker_stra"
# 原始存储位置和缓存report位置
base_path = f"./results/backtest_1129/{token}{quote}.{exchange}_{symbol_type}"
# cache_path = f"./results/cache_taker_stra/{token}{quote}.{exchange}_{symbol_type}"

bacth_start_time = pd.Timestamp("2024-04-02 00:00:00", tz="HONGKONG")
batch_end_time = pd.Timestamp("2024-04-04 00:00:00", tz="HONGKONG")
batch_file_paths = load_batch_file_paths(bacth_start_time, batch_end_time, base_path, factor_name, strategy_name, batch_period_days=1)
print(batch_file_paths)
path_str = f'{token}{quote}.{exchange}_{symbol_type}/{strategy_name}_{bacth_start_time.strftime("%Y%m%d%H:%M")}_{batch_end_time.strftime("%Y%m%d%H:%M")}_{factor_name}'

# 指定时间段
start_time = pd.Timestamp("2024-04-02 00:00:00", tz="HONGKONG")
end_time = pd.Timestamp("2024-04-04 00:00:00", tz="HONGKONG")
trading_days = (end_time - start_time).total_seconds() / 3600/ 24

# 整合结果存储文件夹路径
load_path = f'./results/backtest_batch/{path_str}'
# load_path = './results/cache/BTCUSDT.BINANCE_SPOT_NORMAL/maker_stra_2024050200:00_2024063000:00_vinci_maker_label_0.5_0.5'

# 指定时段结果缓存路径
cache_path = f'./results/cache_taker_stra/{path_str}'

plot_config = {
    "net_value": True,
    "position": True,
    "stat_metrics": True,
    # "order_mark": {
    #     "type_list": ["buy_to_open", "sell_to_close", "sell_to_open", "buy_to_close"]
    # },
    "holding_spot_value": False,
}

# 统计回测结果
t1 = time.time()
capital = get_capital_list(starting_balance)
# load_period_result(token, load_path, cache_path,
#                start_time, end_time, starting_balance, 
#                plot_config,quote, exchange, symbol_type)

load_stra_stat(token, load_path, cache_path, 
               batch_file_paths, start_time, end_time, 
               capital, plot_config, quote, exchange, 
               symbol_type, interval=100,)

print(f'统计耗时：{time.time()-t1:.2f}s')

# t2 = time.time()

# get_stat_analysis(cache_path, trading_days, ts_period=3)
# print(f'分析耗时：{time.time()-t2:.2f}s')