# -*- coding: utf-8 -*-
"""
恢复旧数据的ticker
author: lifangyu
"""

import sys
from s3_utils import *
from ..util.recover_depth import recoveryDepth

def recoverTicker(bucket, exchange, symbol, start_hour, end_hour, pre_data=None):
    """
    从恢复的depths数据中提取盘口数据并转存到指定的文件中
    """
    
    new_bucket = 'recoverticker'
    
    wrong_files = []
    hourlist = get_hourlist(start_hour, end_hour)
    for hour in hourlist:
        try:
            origin_tick_list, pre_data = recoveryDepth(exchange, symbol, hour, head_num=1, pre_data=pre_data, bucket=bucket)
            
            # 如果depth文件为空，则跳过
            if not origin_tick_list:
                pre_data = None
                logger.info(f"!!{hour}/{exchange}/{symbol} not found in {bucket}!!")
                continue
            
            # 统一数据结构
            new_tick_list = format_tick_data(origin_tick_list)   
            
            # 写入s3
            new_records = pd.Series(new_tick_list).to_json(orient='records', lines=True)
            tick_file_key = f"{hour}/{exchange}/{symbol}.log.gz"
            write_suc = write_s3(new_bucket, new_records, tick_file_key)
            logger.info(f"{tick_file_key[:-7]} write2 s3 suc: {write_suc}")
            if not write_suc:
                wrong_files.append(hour)
                
        except Exception as e:
            pre_data = None
            logger.error(f'{hour}/{exchange}/{symbol}: Error: {e}')
    
    return wrong_files
      
            
def format_tick_data(origin_tick_list):
    """
    将depth提取出的盘口和tick数据的结构统一
    """
    new_tick_list = []

    for data in origin_tick_list:
        new_tick = {
            'tp': data['time'],
            'ap': data['asks'][0]['p'],
            'aa': data['asks'][0]['s'],
            'bp': data['bids'][0]['p'],
            'ba': data['bids'][0]['s'],
            'e': 0,
        }
        new_tick_list.append(new_tick)
    
    return new_tick_list


if __name__ == '__main__':

    bucket, exchange, symbol, start_hour, end_hour = sys.argv[1:]
    
    logger = init_log(f'ticker/{exchange}_{symbol}_{start_hour}_{end_hour}.log')
    wrong_files = recoverTicker(bucket, exchange, symbol, start_hour, end_hour)
    logger.info(f"Final Wrong files: {wrong_files}")
        
        
        
    
    
    
    
    
    
    