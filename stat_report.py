# ***** 不用整合结果 *****

import time
import pandas as pd
from strategystats.stat_load import *
from strategystats.stat_analysis import *

# symbol信息和初始balance
token = "BTC"
quote = "USDT"
exchange = "BINANCE"
symbol_type = "SPOT_NORMAL"
starting_balance = [f"10 {token}", f"10 {quote}"]
# factor_name = "xuefeng_5.5e-05_-5.5e-05_None_0"
# strategy_name = "taker_stra"

# 原始存储位置和缓存report位置
base_path = f"./results/backtest_1129/{token}{quote}.{exchange}_{symbol_type}"
cache_path = f"./results/cache_1129/{token}{quote}.{exchange}_{symbol_type}"
path_str = 'taker_stra_2024040200:00_2024040900:00_xuefeng_0926_1e-05_-1e-05_maker_0_1e-05'

# 指定时间段
start_time = pd.Timestamp("2024-04-02 00:00:00", tz="HONGKONG")
end_time = pd.Timestamp("2024-04-09 00:00:00", tz="HONGKONG")
trading_days = (end_time - start_time).total_seconds() / 3600/ 24
# path_str = f'{token}{quote}.{exchange}_{symbol_type}/{strategy_name}_{start_time.strftime("%Y%m%d%H:%M")}_{end_time.strftime("%Y%m%d%H:%M")}_{factor_name}'

# 整合结果存储文件夹路径
load_path = f'{base_path}/{path_str}'

# load_path = './results/backtest_0918/BTCUSDT.BINANCE_SPOT_NORMAL/maker_stra_v3_2024110600:00_2024111100:00_vinci_mixer_cls_0.5_0.5_maker_0_0.5'
# load_path = './results/cache_0913/BTCUSDT.BINANCE_SPOT_NORMAL/maker_stra_2024040100:00_2024040200:00_vinci_maker_label_0.5_0.5'

cache_path = f'{cache_path}/{path_str}'
# cache_path = f'{cache_path}/maker_stra_v3_2024110600:00_2024111100:00_vinci_mixer_cls_0.5_0.5_maker_0_0.5'

plot_config = {
    "market_prices": {
        # "prices_list": ["Ask Price", "Bid Price"]  # 还可以加'Mid Price'
        "prices_list": ["Mid Price"]  # 还可以加'Mid Price'
    },
    "net_value": True,
    "position": True,
    "stat_metrics": True,
    # "hedge": {
    #     "type_list": ["buy_to_open", "sell_to_close", "sell_to_open", "buy_to_close"]
    # },
    # "orders": {
    #     "status_list": ['OPENED', 'CANCELED', 'FILLED', 'REJECTED']
    # },
}

# 统计回测结果
t1 = time.time()
capital = get_capital_list(starting_balance)
load_period_result(token, load_path, cache_path, start_time, end_time, capital, 
                   plot_config, quote, exchange, symbol_type, interval=1, use_s3=False)  # interval画图采样
print(f'统计耗时：{time.time()-t1:.2f}s')

# t2 = time.time()
# get_stat_analysis(cache_path, trading_days, ts_period=3)
# print(f'分析耗时：{time.time()-t2:.2f}s')