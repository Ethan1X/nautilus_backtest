from recover_resample_depth import resample_depth_daily
from multiprocessing import Pool
import utils
import time


dictionary = {"binance": ["btc_usdt", "eth_usdt", "sol_usdt", "bnb_usdt", "xrp_usdt", "arb_usdt", "avax_usdt"],
              "okex": ["btc_usdt", "eth_usdt", "sol_usdt", "bnb_usdt", "xrp_usdt", "arb_usdt", "avax_usdt"]
             }
# dictionary = {"binance": ["avax_usdt"]
#              }

if __name__ == '__main__':

    exchange_symbol_list = [(exchange, symbol) for exchange, symbols in dictionary.items() for symbol in symbols]
    
    num_days = 30
    start = '2024-01-31'
    step = 1
    date_list = utils.get_date_list(start, num_days)
    with Pool(3) as pool:
        args_list = [(e, s, date, step) for e,s in exchange_symbol_list for date in date_list]
        pool.starmap(resample_depth_daily, args_list)
        print('module_one_finish')


