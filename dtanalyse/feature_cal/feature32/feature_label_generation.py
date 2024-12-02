"""
author: wushaofan
说明：
生成样本数据，输出命名文件。分别输出feature，label
使用方式：
如果要从外部import本代码，如果生成feature请只从外部引用feature_generation函数，如果生成label请只从外部引用label_generation函数，其他函数不要引用，输出数据格式、日志文件

本文件只在module_two中引用。
本文件只在module_two中引用。
本文件只在module_two中引用。
"""


import boto3
import io
import os
import numpy as np
import pandas as pd
import logging
import utils
import json

BUCKET = 'dp4intern'
S3_PATH = 'wu/test_data'

def load_npz_from_s3(folder_name:str, file_name: str):
    '''
    注意folder_name默认应该传入是'depths'，之下存储module_1恢复的depths，S3_PATH是全局常量，存储恢复depths的上级目录
    e.g. load_npz_from_s3('depths','binance_btc_usdc/2024010101.npz')
    '''
    file_path = f'{S3_PATH}/{folder_name}/{file_name}'
    try:
        client = boto3.client('s3')
        buffer = io.BytesIO()
        client.download_fileobj(BUCKET, file_path, buffer)
        buffer.seek(0)
    except Exception as e:
        print(e)
        return []
    return np.load(buffer, allow_pickle=True)['data']


def depths_transform_from_npz_to_df_hourly(depths_npz_decompressed):
    '''
    只供depths_transform_from_npz_to_df_daily直接引用的辅助函数
    '''
    entries = []
    
    for depths_at_a_time in depths_npz_decompressed:
        entry = [depths_at_a_time['time']]
        for _ in range(20):
            bids = depths_at_a_time['bids']
            entry.extend([bids[_]['p'], bids[_]['s']])
        for _ in range(20):
            asks = depths_at_a_time['asks']
            entry.extend([asks[_]['p'], asks[_]['s']])
        entries.append(entry)

    return entries

def depths_transform_from_npz_to_df_daily(exchange, symbol, date):
    '''
    生成一天的depths，返回形式为DataFrame
    e.g. depths_transform_from_npz_to_df_daily('binance', 'btc_usdc', '20240101')
    '''
    columns = ['timestamp'] + [f'bid_{c}_{i+1}' for i in range(20) for c in ('p','s')] + [f'ask_{c}_{i+1}' for i in range(20) for c in ('p','s')]
    entries = []
    for i in range(24):
        datehour = date + (str(i) if i >= 10 else '0' + str(i))
        file_name = f'{exchange}_{symbol}/{datehour}.npz'
        entries.extend(depths_transform_from_npz_to_df_hourly(load_npz_from_s3('depths', file_name)))
    return pd.DataFrame(entries, columns=columns).drop_duplicates(['timestamp']) #这样保存的depths是严格的间隔1000ms的


def funds_unit_feature(df):
    '''
    接受一天depths的dataframe，生成funds_unit处理的feature，注意funds_units在这里设置起始为1000，但可调
    '''
    funds_unit = [2 ** pow * 1000 for pow in range(0,16)]
    
    def helper(price_amount_tuples):
        cumulative_values = np.cumsum([price * amount for price, amount in price_amount_tuples])
        cumulative_amount = np.cumsum([amount for _, amount in price_amount_tuples])
        average_price_list = []
        previous_order_amount, previous_order_value, i = 0, 0, 0
        
        for order_quantity in funds_unit:
            while cumulative_values[i] < order_quantity:
                previous_order_value = cumulative_values[i]
                previous_order_amount = cumulative_amount[i]
                if i < len(price_amount_tuples)-1:
                    i+=1
                else:
                    break

            remaining_order_quantity = order_quantity - previous_order_value
            next_order_price = price_amount_tuples[i][0]
            total_order_amount = previous_order_amount + (remaining_order_quantity / next_order_price)
            average_price = order_quantity / total_order_amount
            average_price_list.append(average_price)
            
        return average_price_list
    
    func = lambda entry: helper([(entry[f'bid_p_{i+1}'], entry[f'bid_s_{i+1}']) for i in range(20)]) + helper([(entry[f'ask_p_{i+1}'], entry[f'ask_s_{i+1}']) for i in range(20)])
            
    res = df.apply(func, axis=1, result_type='expand')
    res.rename(lambda x: f'feature{x}', axis=1, inplace=True)
    
    return pd.concat([df.timestamp, res], axis=1)


def feature_generation(exchange, symbol, date, logger_name = 'logger_feature'):
    '''
    e.g. '2024-01-01+binance+btc_usdc.csv'
    '''
    date_no = ''.join(date.split('-'))
    FEATURE_PATH = os.path.join(os.getcwd(), 'Feature')
    
    if not os.path.exists(FEATURE_PATH):
        os.makedirs(FEATURE_PATH)
        
    funds_unit_feature(depths_transform_from_npz_to_df_daily(exchange, symbol, date_no)).to_csv(f'Feature/{date}+{exchange}+{symbol}.csv')
    logger = logging.getLogger(logger_name)
    logger.info(f'saved: Feature/{date}+{exchange}+{symbol}.csv')

    
    
    
    
    

def trade_clock(exchange, symbol, datehour):
    '''
    根据trade原始数据生成未来12s, 30s, 60s的平均成交价格的label数据
    '''
    trades = utils.get_data('dp-trade', f'{datehour}/{exchange}/{symbol}.log.gz')
    
    df = pd.DataFrame([json.loads(x) for x in trades if x != ''])
    df = df.drop(['t','T','e'], axis = 1).set_index('tp').astype({'a':int,'b':int})
    
    df['pq'] = df['p'] * df['q']

    df.index = pd.to_datetime(df.index, unit='ms')
    resampled_df = df.resample('100ms').sum()

    future_avg_values = {}
    for i in [12, 30, 60]:
        future_avg_values[f'avg_price_future_{i}s'] = resampled_df['pq'].rolling(f'{i}s').sum().shift(-i)/resampled_df['q'].rolling(f'{i}s').sum().shift(-i)

    result_df = pd.DataFrame(future_avg_values, index=resampled_df.index).dropna()
    result_df = result_df.reset_index()
    result_df['timestamp'] = result_df['tp'].apply(lambda x: int(pd.Timestamp(x).timestamp() * 1000))
    result_df = result_df.drop('tp', axis=1)
    
    return result_df
        
    
    
def trade_transaction(exchange, symbol, datehour):
    '''
    根据trade原始数据生成未来5个，10个，15个订单的平均成交价格的label数据
    '''
    trades = utils.get_data('dp-trade', f'{datehour}/{exchange}/{symbol}.log.gz')
    
    df = pd.DataFrame([json.loads(x) for x in trades if x != ''])
    
    df = df.drop(['t','T','e','a','b'], axis = 1).set_index('tp')
    
    df['pq'] = df['p'] * df['q']

    df.index = pd.to_datetime(df.index, unit='ms')
    resampled_df = df.resample('100ms').sum()

    
    future_avg_values = {}
    for i in [5, 10, 15]:
        future_avg_values[f'avg_price_future_{i}txn'] = df['pq'].rolling(i).sum().shift(-i)/df['q'].rolling(i).sum().shift(-i)
    
    result_df = pd.DataFrame(future_avg_values, index=df.index).dropna()
    result_df = result_df[~result_df.index.duplicated()].resample('100ms').bfill()
    
    result_df = result_df.reset_index()
    result_df['timestamp'] = result_df['tp'].apply(lambda x: int(pd.Timestamp(x).timestamp() * 1000))
    result_df = result_df.drop('tp', axis=1)

    return result_df


def trade_label(exchange, symbol, date):
    df = depths_transform_from_npz_to_df_daily(exchange, symbol, date)
    midprice = pd.concat([(df['bid_p_1']+df['ask_p_1'])/2, df.timestamp], axis=1).rename({0:'midprice'}, axis=1)
    
    # 这里只用trade_clock
    trades = []
    for i in range(0,24):    
        trades.append(trade_clock(exchange, symbol, date+(str(i) if i >= 10 else '0'+str(i))))
    trades = pd.concat(trades, axis=0)
    trades = trades.merge(midprice, on='timestamp')
    for i in [12, 30, 60]:
        trades[f'avg_price_future_{i}s'] = trades[f'avg_price_future_{i}s']/trades['midprice'] - 1
    return trades.drop(['midprice'], axis=1)
    
    

    
def label_generation(exchange_symbol_list, date, logger_name = 'logger_label'):
    '''
    e.g. '2024-01-01+label.csv'
    '''
    date_no = ''.join(date.split('-'))
    LABEL_PATH = os.path.join(os.getcwd(), 'Label')
    
    if not os.path.exists(LABEL_PATH):
        os.makedirs(LABEL_PATH)
    
    df = []
    for exchange, symbol in exchange_symbol_list:
        trade_label(exchange, symbol, date_no).to_csv(f'Label/{date}+{exchange}+{symbol}+label.csv')
        logger = logging.getLogger(logger_name)
        logger.info(f'saved: Label/{date}+{exchange}+{symbol}+label.csv')
        
    
    

    
def setup_logger(logger_name, log_file, level=logging.INFO):
    LOG_PATH = os.path.join(os.getcwd(), 'logs')
    
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    log_file_path = os.path.join(LOG_PATH, log_file)
    if not os.path.exists(log_file_path):
        open(log_file_path, 'a').close()
        
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file_path, mode='w')
    fileHandler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fileHandler)

    
if __name__ != '__main__':
    # 注意这里是不等号
    setup_logger('logger_feature', 'feature_log_file.log')
    setup_logger('logger_label', 'label_log_file.log')