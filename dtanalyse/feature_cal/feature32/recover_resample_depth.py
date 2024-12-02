"""
author: lifangyu, wushaofan, liuyu
说明：
恢复全量数据，输出命名文件。接受file可变参数输出文件到对应, 接受step可变参数对数据进行重采样降频
使用方式：
如果要从外部import本代码，请只从外部引用resample_depth_daily函数，其他函数不要引用，输出数据格式、日志文件

本文件只在module_one中引用。
本文件只在module_one中引用。
本文件只在module_one中引用。
"""


import io
import os
import random
import utils
import datetime
import logging
import boto3
import gzip
from botocore.config import Config
import json
import numpy as np
import utils
import datetime



def update_order_depth(order_depth_delta, current_order_depth, order_type:str):
    """
    用增量数据更新当前全量数据
    """
    for entry in order_depth_delta:
        target_price = entry['p']
        # 使用二分查找
        left, right = 0, len(current_order_depth) - 1
        while left <= right:
            mid = (left + right) // 2
            if current_order_depth[mid]['p'] == target_price:
                if entry['s'] == 0:
                    del current_order_depth[mid]
                else:
                    current_order_depth[mid]['s'] = entry['s']
                break
            if order_type == 'bid':
                if current_order_depth[mid]['p'] > target_price:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                if current_order_depth[mid]['p'] < target_price:
                    left = mid + 1
                else:
                    right = mid - 1
        if left > right and entry['s'] != 0:
            current_order_depth.insert(left, entry)
    

def append_depth(depth_update, data_one):
    """
    将新数据（data_one)添加到depth_update中，准备用于更新depth
    """
    if data_one.get('e'):
        del data_one['e']
    del data_one['tp']
    if data_one['t'] == 'buy':
        depth_update['bids'].append(data_one)
    else:
        depth_update['asks'].append(data_one)
    del data_one['t']


def match_depth(depth):
    """
    如果存在盘口重叠（买价高于卖价），则进行撮合，用买一卖一进行消除，size小的订单删除，大的订单更新size
    """
    if len(depth['asks']) == 0 or len(depth['bids']) == 0:
        return True

    curr_ask, curr_bid = 0, 0
    while depth['asks'][curr_ask]['p'] <= depth['bids'][curr_bid]['p']:
        ask_s = depth['asks'][curr_ask]['s']
        bid_s = depth['bids'][curr_bid]['s']
        if ask_s == bid_s:
            curr_ask += 1
            curr_bid += 1
        elif ask_s < bid_s:
            curr_ask += 1
            depth['bids'][curr_bid]['s'] = bid_s - ask_s
        else:
            depth['asks'][curr_ask]['s'] = ask_s - bid_s
            curr_bid += 1
        if curr_ask >= len(depth['asks']) or curr_bid >= len(depth['bids']):
            break

    del depth['asks'][:curr_ask]
    del depth['bids'][:curr_bid]
    return True


def recover_depth_hourly(exchange, symbol, date_hour, step, head_num=20, pre_data=None, bucket='depths'):
    """
    作为resample_depth_daily引用的辅助函数，细化生成每个小时数据
    """

    file_key = "{}/{}/{}.log.gz".format(date_hour, exchange, symbol)
    origin_data = utils.get_data(bucket, file_key)
    if not origin_data:
        logger = logging.getLogger('logger')
        logger.warning('file_key is empty')
        return [], []

    raw_data_list = [json.loads(data) for data in origin_data if data != ""]
    # 如果有e字段且不为0，则以e为标准；否则按tp
    if 'e' in raw_data_list[0] and raw_data_list[0].get('e') != 0:
        time_type = 'e'
    else:
        time_type = 'tp'
    raw_data_list = sorted(raw_data_list, key=lambda x: x[time_type])
    
    res = []
    depth_delta = {'bids': [], 'asks': []}  # 用于存储每个tp的增量数据
    depth_snapshot = {'bids': [], 'asks': []}  # 用于存储每个tp的全量数据
    # pre_data为空，表示是第一个小时的数据，要先跳过数据最前面不是全量的部分，找到第一条全量推送
    # 否则，pre_data传入的是上一个小时最后一个时刻的depth恢复全量的数据
    
    # 若是一天中第一个小时，则start是第一个全量数据的index，否则是0
    start = 0  
    if not pre_data:
        while '_' not in raw_data_list[start]:
            start += 1    # 找到第一条全量推送
        pre_data = {'bids': [], 'asks': []}
    depth_time = 0  # 当前数据时间

    # 按照step，起始时间，终止时间，生成时间戳序列，每一个时间戳记录一个盘口快照，单位是秒
    start_one_tp = raw_data_list[start][time_type]
    end_one_tp = raw_data_list[-1][time_type]
    time_list = utils.datetime_range(utils.unix2dt(start_one_tp//1000), 
                                     utils.unix2dt(end_one_tp//1000+1), 
                                     datetime.timedelta(seconds=step))
    time_idx = 1

    for i in range(start, len(raw_data_list)):
        if not raw_data_list[i]:
            continue
        try:
            data_one = raw_data_list[i]
        except Exception as e:
            print(e)
            print(origin_data[i])
            continue
        # 到下一个时间戳
        if data_one[time_type] != depth_time:
            # 将上一个tp累计的数据更新进去，如果有全量就用全量代替，没有就用增量更新
            if len(depth_snapshot['bids']) > 10 and len(depth_snapshot['asks']) > 10:
                pre_data = depth_snapshot
            else:
                update_order_depth(depth_delta['asks'], pre_data['asks'], order_type='ask')
                update_order_depth(depth_delta['bids'], pre_data['bids'], order_type='bid')

            # 检查盘口重叠
            match_depth(pre_data)

            # 每个时间戳记录之前的最后一条快照
            if (pre_data['asks'] or pre_data['bids']) and (depth_time <= time_list[time_idx] < data_one[time_type]):
                # 如果两个时间戳之间没有增量数据，则用上一个时间戳的全量数据填充，确保每个时间戳都有数据记录
                while time_idx < len(time_list)-1 and time_list[time_idx] < data_one[time_type]:
                    one = {
                        'bids': [{'p': i['p'], 's': i['s']} for i in pre_data['bids'][:head_num]],
                        'asks': [{'p': i['p'], 's': i['s']} for i in pre_data['asks'][:head_num]],
                        'time': time_list[time_idx],
                    }
                    res.append(one)
                    time_idx += 1

            # 更新
            depth_time = data_one[time_type]
            depth_delta = {'bids': [], 'asks': []}
            depth_snapshot = {'bids': [], 'asks': []}

        # 累积该数据
        if '_' in data_one:
            append_depth(depth_snapshot, data_one)
        else:
            append_depth(depth_delta, data_one)

    update_order_depth(depth_delta['asks'], pre_data['asks'], order_type='ask')
    update_order_depth(depth_delta['bids'], pre_data['bids'], order_type='bid')
    # 检查盘口重叠
    match_depth(pre_data)
    one = {
        'bids': [{'p': i['p'], 's': i['s']} for i in pre_data['bids'][:head_num]],
        'asks': [{'p': i['p'], 's': i['s']} for i in pre_data['asks'][:head_num]],
        'time': time_list[time_idx]
    }
    res.append(one)

    return res, pre_data

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

def save_npz_to_s3(folder_name: str, file_name: str, output_data, logger_name, s3_path, bucket):
    logger = logging.getLogger(logger_name)
    file_path = f'{s3_path}/{folder_name}/{file_name}'
    if not output_data:
        logger.warning(f"!!!EMPTY data: {file_path}!!!")
        return
    client = boto3.client('s3')
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=np.array(output_data, dtype=object))
    buffer.seek(0)
    client.upload_fileobj(buffer, bucket, file_path)
    logger.info(f"saved: {file_path}, size: {len(output_data)}")

    
def resample_depth_daily(exchange, symbol, date, step=1, folder_name='depths', s3_path='wu/test_data', bucket='dp4intern'):
    """
    引用辅助函数recover_depth_hourly，生成每日深度数据（24小时，每小时一个文件），频率指定。
    step参数可变，代表降采样resample的频率，默认为1不降采样；
    注意格式，如'binance', 'btc_usdt', '2023-03-04'，s3_path和bucket参数可变，表文件存放位置: bucket / s3_path / folder_name 
    """
    pre_data = None
    for hour in range(-1, 25):
        hour_str = utils.dt2str(utils.str2dt(date, dt_form='%Y-%m-%d') + datetime.timedelta(hours=hour), dt_form='%Y%m%d%H')
        hourly_data, pre_data = recover_depth_hourly(exchange, symbol, hour_str, step=step, head_num=20, pre_data=pre_data)
        if hour == -1: continue
        save_npz_to_s3(folder_name, f'{exchange}_{symbol}/{hour_str}.npz', hourly_data, 'logger', s3_path, bucket)
        print(f'one hour{hour}')
if __name__ != '__main__':
    # 注意这里是不等号
    setup_logger('logger', 'depth_log_file.log')

