"""
author: lifangyu, wushaofan, liuyu
说明：
生成样本数据，输出命名文件。接受resample等可变参数输出包含feature、label的样本
使用方式：
如果要从外部import本代码，请只从外部引用generate_samples函数，其他函数不要引用，输出数据格式、日志文件

本文件只在module_two中引用。
本文件只在module_two中引用。
本文件只在module_two中引用。
"""

import io
import os
import random
import utils
import datetime
import boto3
import logging
import json
import numpy as np
import statistics
import math

from collections import deque
from typing import List, NamedTuple, Tuple
from multiprocessing import Pool


NUM_SAMPLES_MEMORY_LIMIT = 1  # 35%
NUM_SAMPLES_PER_FILE = 1
RESAMPLE_FREQUENCY = 1

BUCKET = 'dp4intern'
S3_PATH = 'wu/test_data'


class Depth(NamedTuple):
    # 每个list里面的元素是一个二元组，存放价格和数量
    bids: list
    asks: list
    

class JointDepth(NamedTuple):
    # 所有币对的Depth拼到一起
    timestamp: int
    depth: list


class TradePrices(NamedTuple):
    # 过去成交均价，feature
    bids_price: list
    asks_price: list
    

class JointTradePrices(NamedTuple):
    timestamp: int
    prices: list
    

class Ticker(NamedTuple):
    # bid 和 ask 是四元组，存放未来 10s （label_len） 内的开盘，最高，最低，闭盘数据
    bid: tuple
    ask: tuple
    

class JointTicker(NamedTuple):
    # 所有币对的Ticker拼到一起
    timestamp: int
    ticker: list


class MaxPct(NamedTuple):
    # 未来 10s 的 “累计” 最大涨跌幅
    down_pct: list
    up_pct: list
    

class JointMaxPct(NamedTuple):
    timestamp: int
    pct: list


class Sample(NamedTuple):
    feature: np.array
    new_feature: np.array
    reg_label: np.array
    cls_label: np.array
    timestamp: np.array


class DataPointer(object):
    def __init__(self, iterator):
        self._iter = iterator
        self.current = iterator.__next__()
        self.next = iterator.__next__()

    def step(self):
        self.current = self.next
        self.next = self._iter.__next__()


def save_npz_to_s3(folder_name: str, file_name: str, output_data, logger_name):
    logger = logging.getLogger(logger_name)
    file_path = f'{S3_PATH}/{folder_name}/{file_name}'
    if not output_data:
        logger.warning(f"!!!EMPTY data: {file_path}!!!")
        return
    client = boto3.client('s3')
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=np.array(output_data, dtype=object))
    buffer.seek(0)
    client.upload_fileobj(buffer, BUCKET, file_path)
    logger.info(f"saved: {file_path}, size: {len(output_data)}")


def load_npz_from_s3(folder_name:str, file_name: str):
    file_path = f'{S3_PATH}/{folder_name}/{file_name}'
    try:
        client = boto3.client('s3')
        buffer = io.BytesIO()
        client.download_fileobj(BUCKET, file_path, buffer)
        buffer.seek(0)
    except:
        return []
    return np.load(buffer, allow_pickle=True)['data']
    

def setup_logger(logger_name, log_file, level=logging.INFO):
    LOG_PATH = os.path.join(os.getcwd(), 'logs')
    
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    log_file = os.path.join(LOG_PATH, log_file)
    
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fileHandler)
    return logger


def load_depth_iter(date, exchange, symbol):
    """
    必须往后多读一小时数据，否则会丢失最后几秒的数据
    """
    logger = logging.getLogger(f'sample_{date}')
    for hour in range(0, 25):
        hour_time = date + datetime.timedelta(hours=hour)
        hour_str = utils.dt2str(hour_time, dt_form='%Y%m%d%H')
        file_name = f'{exchange}_{symbol}/{hour_str}.npz'
        hour_depth = load_npz_from_s3('depths', file_name)
        if len(hour_depth) > 0:
            logger.info(f"load depth: {file_name}")
        else:
            logger.warning(f"!!!EMPTY data: {file_name}!!!")
        for depth in hour_depth:
            yield depth


def load_ticker_iter(date, exchange, symbol):
    logger = logging.getLogger(f'sample_{date}')
    for hour in range(0, 25):
        hour_time = date + datetime.timedelta(hours=hour)
        hour_str = hour_time.strftime("%Y%m%d%H")
        hour_ticker = utils.get_data('dpticker', f'{hour_str}/{exchange}/{symbol}.log.gz')
        if len(hour_ticker) > 0:
            logger.info(f"load ticker: {hour_str}/{exchange}/{symbol}.log.gz")
        else:
            logger.warning(f"!!!EMPTY data: {hour_str}/{exchange}/{symbol}.log.gz!!!")
        for ticker in hour_ticker:
            if ticker:
                yield json.loads(ticker)


def get_average_prices_of_given_queries(price_amount_tuples: list, funds_unit: list):
   # 计算均价 （feature） 
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

    return np.array(average_price_list)

            
                
def joint_depth_to_price(joint_depth: JointDepth, funds_unit: list):
    price_list = []
    
    for depth in joint_depth.depth:
        asks_price = get_average_prices_of_given_queries(depth.asks, funds_unit)
        bids_price = get_average_prices_of_given_queries(depth.bids, funds_unit)
        price_list.append(TradePrices(bids_price, asks_price))
    
    return JointTradePrices(joint_depth.timestamp, price_list)


def joint_ticker_queue_to_max_pct(queue: List[JointTicker]):
    """
    获得1-10s内的最大涨跌幅
    """
    max_pct_list = []
    num_channels = len(queue[0].ticker)
    
    for i in range(num_channels):
        open_bid_p = queue[0].ticker[i].bid[0]
        open_ask_p = queue[0].ticker[i].ask[0]
        low_bid_p_list = [joint_ticker.ticker[i].bid[2] for joint_ticker in queue]
        high_ask_p_list = [joint_ticker.ticker[i].ask[1] for joint_ticker in queue]
        cummin_bid_p_list = np.minimum.accumulate(low_bid_p_list)
        cummax_ask_p_list = np.maximum.accumulate(high_ask_p_list)
        max_down_pct_list = (cummin_bid_p_list - open_bid_p) / open_bid_p
        max_up_pct_list = (cummax_ask_p_list - open_ask_p) / open_ask_p
        max_pct_list.append(MaxPct(max_down_pct_list, max_up_pct_list))
    
    return JointMaxPct(queue[0].timestamp, max_pct_list)
    

def depth_generator(date, feature_len, label_len, step, exchange_symbol_list):
    """
    将所有币对的depth拼到一起，并计算平均成交价；depth已经是降采样过的数据
    """
    pointers = [DataPointer(load_depth_iter(date, exchange, symbol)) for exchange, symbol in exchange_symbol_list]
    total_len = feature_len + label_len
    queue = deque(maxlen=total_len)
    
    start_time = date + datetime.timedelta(seconds=1*step)
    end_time = date + datetime.timedelta(days=1) + datetime.timedelta(seconds=(total_len-1)*step)
    time_list = utils.datetime_range(start_time, end_time, datetime.timedelta(seconds=step), include_end=True)
    
    try:
        for curr_tp in time_list:
            lastest = max(pointer.current["time"] for pointer in pointers)
            if curr_tp < lastest:
                continue
            current_depth_list = []
            for pointer in pointers:
                while pointer.next["time"] <= curr_tp:
                    pointer.step()
                if pointer.current["time"] != curr_tp:
                    break
                
                bids = [(d["p"], d["s"]) for d in pointer.current["bids"]]
                asks = [(d["p"], d["s"]) for d in pointer.current["asks"]]
                current_depth_list.append(Depth(bids, asks))
            
            # 这里else和上面for loop对应，表示如果非正常循环终止（break）的语句
            else:
                if queue and curr_tp != queue[-1].timestamp + step * 1_000:
                    queue.clear()

                joint_depth = JointDepth(curr_tp, current_depth_list)
                queue.append(joint_depth)
                
                if len(queue) >= total_len:
                    yield list(queue)
                    
    except StopIteration:
        pass

    


def ticker_generator(date, feature_len, label_len, step, exchange_symbol_list):
    """
    将所有币对的ticker拼到一起，并计算1-10s内的最大涨跌幅；ticker是未降采样的数据
    """
    pointers = [DataPointer(load_ticker_iter(date, exchange, symbol))
        for exchange, symbol in exchange_symbol_list]
    queue = deque(maxlen=label_len)
    
    start_time = date + datetime.timedelta(seconds=feature_len*step)
    end_time = date + datetime.timedelta(days=1) + datetime.timedelta(seconds=(feature_len+label_len-1)*step)
    time_list = utils.datetime_range(start_time, end_time, datetime.timedelta(seconds=step), include_end=True)
    
    try:        
        file_missing = False
        first = True
        for curr_tp in time_list:
            lastest = max(pointer.current["tp"] for pointer in pointers)
            if curr_tp < lastest:
                continue
            if file_missing or first:
                file_missing, first = False, False
                for pointer in pointers:
                    while pointer.next["tp"] <= curr_tp:
                        pointer.step()
                continue
            
            current_ticker_list = []   
            for pointer in pointers: 
                if pointer.next['tp'] - pointer.current['tp'] > 40 * 60 * 1000:  # 缺失文件
                    pointer.step()
                    file_missing = True
                    break
                
                interval_bid_p = [pointer.current["bp"]]
                interval_ask_p = [pointer.current["ap"]]
                
                while pointer.next["tp"] <= curr_tp:  # 使当前指针是t时刻前的最后一条数据
                    pointer.step()
                    interval_bid_p.append(pointer.current["bp"])
                    interval_ask_p.append(pointer.current["ap"])
                    
                open_bid_p, close_bid_p = interval_bid_p[0], interval_bid_p[-1]
                open_ask_p, close_ask_p = interval_ask_p[0], interval_ask_p[-1]
                
                high_bid_p, low_bid_p = max(interval_bid_p), min(interval_bid_p)
                high_ask_p, low_ask_p = max(interval_ask_p), min(interval_ask_p)
                
                bid = (open_bid_p, high_bid_p, low_bid_p, close_bid_p)
                ask = (open_ask_p, high_ask_p, low_ask_p, close_ask_p)
                current_ticker_list.append(Ticker(bid, ask))
            
            else:  
                if queue and curr_tp != queue[-1].timestamp + step * 1_000:
                    queue.clear()
                
                queue.append(JointTicker(curr_tp, current_ticker_list))
                
                if len(queue) >= label_len:
                    yield joint_ticker_queue_to_max_pct(queue)
            
    except StopIteration:
        pass     


def process_reg_data(joint_depth_window: List[JointDepth], feature_len: int, label_len: int, funds_unit):
    """
    对feature和回归的label进行分割和normalization：分割为128s（feature_len）和10s (label_len)
    """
    
    joint_prices_window=[]
    timestamp_list=[] # 存放时间戳数据
    for i in range(len(joint_depth_window)):
        joint_prices_window.append(joint_depth_to_price(joint_depth_window[i],funds_unit))
        
    for i in range(len(joint_prices_window)):
        timestamp_list.append(joint_prices_window[i].timestamp)
        
        
    
    timestamp_list=np.array(timestamp_list)
    window_array = np.array([[np.concatenate((np.flip(depth.bids_price), depth.asks_price)) 
                              for depth in joint_prices.prices] for joint_prices in joint_prices_window])
    feature_array = window_array[:feature_len,:,:]
    label_array = window_array[feature_len:feature_len+label_len]

    last_midprice = np.mean([quote[0] for depth in joint_prices_window[feature_len-1].prices 
                             for quote in depth])
    
    feature_mean_removed_window = feature_array - last_midprice
    min_value = np.min(feature_mean_removed_window)
    range_value = np.max(feature_mean_removed_window) - min_value
    feature_normalized_window = (feature_mean_removed_window - min_value) / range_value

    # label calculation
    label_mean_removed_window = label_array - last_midprice
    label_normalized_window = (label_mean_removed_window - min_value) / range_value
    
    return feature_normalized_window, label_normalized_window, timestamp_list





def process_new_def_data(joint_depth_window: List[JointDepth], feature_len: int, label_len: int, exchange_symbol_list):
    
    new_feature = []
    
    for exchange, symbol in exchange_symbol_list:
        index=exchange_symbol_list.index((exchange, symbol)) # 调取某个交易所的某个币种在list中的index
        loop=[]
        for i in range(label_len):
            price_current_bids=joint_depth_window[i+feature_len-1].depth[index].bids[0][0]
            price_current_asks=joint_depth_window[i+feature_len-1].depth[index].asks[0][0]
            price_next_bids=joint_depth_window[i+feature_len].depth[index].bids[0][0]
            price_next_asks=joint_depth_window[i+feature_len].depth[index].asks[0][0]
            d_bids=(price_next_bids-price_current_bids)/price_current_bids
            d_asks=(price_next_asks-price_current_asks)/price_current_asks
            loop.append([d_bids,d_asks])
        
        new_feature.append(loop)
    
    new_feature=np.array(new_feature)
    
    return np.transpose(new_feature,(1,0,2))
        



def generate_samples(folder_name, date, file_index, feature_len, label_len, step, exchange_symbol_list, funds_unit):
    """
    按照时间戳将两个label拼在一起，并将sample保存为npz文件存入s3
    """
    
   
    date = utils.str2dt(date, dt_form='%Y-%m-%d')
    setup_logger(f'sample_{date}', f'sample_{date}.log')
    
    depth_pointer = DataPointer(depth_generator(date, feature_len, label_len, step, exchange_symbol_list))
    tick_pointer = DataPointer(ticker_generator(date, feature_len, label_len, step, exchange_symbol_list))
    sample_list = []
    sample_folder_name = f'{folder_name}_samples'
    count = 0
    
    
    try:
        while True:
            # 迭代
            curr_depth_data = depth_pointer.current
            curr_tick_data = tick_pointer.current
            # 时间戳对齐
            if curr_depth_data[feature_len].timestamp < curr_tick_data.timestamp:
                depth_pointer.step()
            elif curr_depth_data[feature_len].timestamp > curr_tick_data.timestamp:
                tick_pointer.step()
            else:
                # 生成feature and label
                reg_data = process_reg_data(curr_depth_data, feature_len, label_len, funds_unit) # 返回128s和10s的feature并返回时间戳
                new_label_data = process_new_def_data(curr_depth_data, feature_len, label_len, exchange_symbol_list)
                cls_data = np.array([(ticker.down_pct, ticker.up_pct) for ticker in curr_tick_data.pct]).transpose(2, 0, 1) # 返回10s的是涨还是跌
                
                count += 1
                if count == RESAMPLE_FREQUENCY:
                    sample_list.append(Sample(reg_data[0], reg_data[1], cls_data, new_label_data, reg_data[2]))
                    count = 0
                

                
                if len(sample_list) >= NUM_SAMPLES_MEMORY_LIMIT:
                    random.shuffle(sample_list)
                    output_data = sample_list[:NUM_SAMPLES_PER_FILE]
                    file_name = f'{folder_name}_{file_index}_samples'
                    save_npz_to_s3(sample_folder_name, file_name, output_data, logger_name=f'sample_{date}')
                    file_index += 1
                    del sample_list[:NUM_SAMPLES_PER_FILE]
                
                tick_pointer.step()
                depth_pointer.step()
                
    except StopIteration:
        pass
    
    def shuffle_and_slice_samples(sample_list: list, size: int):
        random.shuffle(sample_list)
        return [sample_list[i:i+size] for i in range(0, len(sample_list), size)]

    
    for samples in shuffle_and_slice_samples(sample_list, NUM_SAMPLES_PER_FILE):
        file_name = f'{folder_name}_{file_index}_samples'
        save_npz_to_s3(sample_folder_name, file_name, samples, logger_name=f'sample_{date}')
        file_index += 1
        

