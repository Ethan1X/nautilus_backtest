import time
import datetime
import json
import copy
import boto3
import gzip
from joblib import Parallel, delayed
from botocore.config import Config
import os, sys, datetime, logging
import plotly.graph_objs as go
import pandas as pd
import numpy as np
if '../' not in sys.path:
    sys.path.append('../')

from util.s3_method import * 
from util.time_method import *
from util.plot_method import easy_plot
from util.hedge_log import initlog
from util.plot_method import timeline_sample_kv, get_plot_diff_data
from util.recover_depth import recoveryDepth
from util.statistic_method_v2 import describe_series



def get_data(bucket, file_key):
    """
    直接使用文件路径下载bucket上的文件，适用任意文件路径
    """
    records = []
    
    my_config = Config(
        region_name = 'ap-northeast-1'
    )
    client_s3 = boto3.client('s3', config = my_config)
    
    try:
        res_one = client_s3.get_object(
            Bucket=bucket,
            Key=file_key,
        )
    except Exception as e:
        if "The specified key does not exist" in str(e):
            print(f"{file_key} 不在当前bucket中")
        else:
            print(f"{file_key} 从bucket中获取数据出现异常 {e}")
            
        return records
    
    #下载到本地
    content = res_one['Body'].read()
    #解压缩出来
    ct_dzip = gzip.decompress(content).decode()
    records = ct_dzip.split('\n')
    
    return records

def originDepth(exchange, symbol, date_hour, head_num=20, pre_data=None, bucket='depths'):
    print(f"begin to recover depth at date {date_hour}: {datetime.datetime.now()}")
    file_key = "{}/{}/{}.log.gz".format(date_hour, exchange, symbol)
    origin_data = get_data(bucket, file_key)
    if not origin_data:
        return [], []
    
    raw_data_list = [json.loads(data) for data in origin_data if data != ""]
    return raw_data_list

def get_cex_depth_origin(begin_time, end_time, exchange, symbol):
    '''
        获取cex相关depth（恢复前的）
    '''
    cex_depth = []
    begin_time_ = begin_time
    prev_data = None
    s3_dt = dt2str(begin_time_, DATETIME_FORMAT4)

    while s3_dt <= dt2str(end_time, DATETIME_FORMAT4):
        logging.info(f"开始恢复 {exchange}_{symbol} 的depth数据")
        depth_list = originDepth(exchange, symbol, s3_dt, 5)
        cex_depth.extend(depth_list)
        begin_time_ += datetime.timedelta(hours=1)
        s3_dt = dt2str(begin_time_, DATETIME_FORMAT4)
    return cex_depth
    #return query_dict_list(cex_depth, 'time', gte=dt2unix(begin_time, 1000), lte=dt2unix(end_time, 1000))
    
def convert_depth_format3(depth):   # Higher performance !!!
    """ Convert the depth data to a pandas dataframe with each row representing bids and asks together
    Parameters
    ----------
    depth: [{'bids': [{'p': 42556.06, 's': 1.24069, '_': '_'},
                {'p': 42555.82, 's': 0.00062, '_': '_'}],
                'asks': [{'p': 42556.07, 's': 9.04977, '_': '_'},
                {'p': 42556.6, 's': 0.00062, '_': '_'}],
                'time': 1702746052036}]
    Returns
    -------
                    bid1	    bid2		bid_qty1	bid_qty2	ask1	    ask2	    ask_qty1    ask_qty2	
    TimeStamp																				
    1702746052036	42556.06	42556.00	1.24069	    0.22604	    42556.07	42556.34	9.04977	    0.00062	
    """
    def get_p_s(x):
        y = []
        for i in range(10):
            y.append(x[i]['p'])
            y.append(x[i]['s'])
        return pd.Series(y)
    depth_df = pd.DataFrame(depth)
    depth_df[['bid1', 'bid_qty1','bid2', 'bid_qty2','bid3', 'bid_qty3','bid4', 'bid_qty4','bid5', 'bid_qty5',
           'bid6', 'bid_qty6','bid7', 'bid_qty7','bid8', 'bid_qty8','bid9', 'bid_qty9','bid10', 'bid_qty10']] = depth_df['bids'].apply(get_p_s).apply(pd.Series)
    depth_df[['ask1', 'ask_qty1','ask2', 'ask_qty2','ask3', 'ask_qty3','ask4', 'ask_qty4','ask5', 'ask_qty5',
             'ask6', 'ask_qty6','ask7', 'ask_qty7','ask8', 'ask_qty8','ask9', 'ask_qty9','ask10', 'ask_qty10']] = depth_df['asks'].apply(get_p_s).apply(pd.Series)
    del depth_df['bids'], depth_df['asks']
    depth_df.set_index(pd.to_datetime(depth_df["time"], unit='ms'), inplace=True)
    return depth_df    


def compare_dicts(dict1, dict2, first_price_previous_dict):
    changes = []
    keys1, keys2 = set(dict1.keys()), set(dict2.keys())
    common_prices = keys1.intersection(keys2)

    for price in common_prices:
        if dict1[price] != dict2[price]:
            price_difference = price - first_price_previous_dict
            changes.append({'price_difference': price_difference, 'amount_change': dict2[price] - dict1[price]})
    
    unique_to_dict1 = keys1 - keys2
    unique_to_dict2 = keys2 - keys1
    
    for price in unique_to_dict1:
        price_difference = price - first_price_previous_dict
        changes.append({'price_difference': price_difference, 'amount_change': -dict1[price]})
    for price in unique_to_dict2:
        price_difference = price - first_price_previous_dict
        changes.append({'price_difference': price_difference, 'amount_change': dict2[price]})
    
    return changes

#拟合用到的损失函数
def model_amount(x, a, b, c):
    return c*10**(x)**(a-1) * np.exp(- 10**(x / b))


def loss_amount(params,x,y):
    a, b, c = params
    predicted = model_amount(x, a, b, c)
    return np.sum((y - predicted)**2)

def model_delta(x, a, b, c):
    return c*a**b/x**(b)


def loss_delta(params,x,y):
    a, b, c = params
    predicted = model_delta(x, a, b, c)
    return np.sum((y - predicted)**2)