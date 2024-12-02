from feature_label_generation import label_generation
from feature_label_generation import feature_generation
from multiprocessing import Pool
import utils
import time


dictionary = {"binance": ["btc_usdt", "eth_usdt", "sol_usdt", "bnb_usdt", "xrp_usdt", "arb_usdt", "avax_usdt"],
              "okex": ["btc_usdt", "eth_usdt", "sol_usdt", "bnb_usdt", "xrp_usdt", "arb_usdt", "avax_usdt"]
             }

# dictionary = {"binance": ["avax_usdt"]
#              }

exchange_symbol_list = [(exchange, symbol) for exchange, symbols in dictionary.items() for symbol in symbols]

if __name__ == '__main__':


    
    num_days = 30
    start = '2024-01-31'
    date_list = utils.get_date_list(start, num_days)
    
    
    with Pool(3) as pool:
        args_list = [(exchange, symbol, date) for exchange, symbol in exchange_symbol_list for date in date_list]
        pool.starmap(feature_generation, args_list)
    