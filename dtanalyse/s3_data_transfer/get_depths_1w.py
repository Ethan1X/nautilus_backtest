# -*- coding: utf-8 -*-
"""
提取盘口1w的平均价格和总Volume
author: lifangyu
"""

import sys
from s3_utils import *
from ..util.recover_depth import recoveryDepth

def cal_depths1w_stats(orders):
    """
    得到盘口1w的平均价格和总量，两者相乘就是总的VOLUME
    """
    
    total_size = 0
    total_vol = 0
    for order in orders:
        p = order['p']
        s = order['s']
        if total_vol + s*p >= 10_000:
            total_size += (10_000 - total_vol) / p
            total_vol = 10_000
            break

        total_size += s
        total_vol += s*p
        
    avg_price = total_vol / total_size
    
    return avg_price, total_size
        
def agg_depths1w(depths):
    """
    循环计算每个时刻卖单和买单盘口1w的平均价格和总量（都保留8位小数），结构和tick数据统一
    """
    depths_1w_list = []
    for data in depths:
        tp = data['time']
        ask_p, ask_s = cal_depths1w_stats(data['asks'])
        bid_p, bid_s = cal_depths1w_stats(data['bids'])
        depths_1w = {
            'tp': tp,
            'ap': round(ask_p, 8),
            'aa': round(ask_s, 8),
            'bp': round(bid_p, 8),
            'ba': round(bid_s, 8),
        }
    
        depths_1w_list.append(depths_1w)
    
    return depths_1w_list
    
    
def getDepths1w(bucket, exchange, symbol, start_hour, end_hour, pre_data=None):
    
    new_bucket = 'depths1w'
    
    wrong_files = []
    hourlist = get_hourlist(start_hour, end_hour)
    for hour in hourlist:
        try:
            depths, pre_data = recoveryDepth(exchange, symbol, hour, head_num=10, pre_data=pre_data, bucket=bucket)
            
            # 如果depth文件为空，则跳过
            if not depths:
                pre_data = None
                logger.info(f"!!{hour}/{exchange}/{symbol} not found in {bucket}!!")
                continue
            
            # 得到盘口1w的平均价格和总量
            depths_1w_list = agg_depths1w(depths)   
            
            # 写入s3
            new_records = pd.Series(depths_1w_list).to_json(orient='records', lines=True)
            file_key = f"{hour}/{exchange}/{symbol}.log.gz"
            write_suc = write_s3(new_bucket, new_records, file_key)
            logger.info(f"{file_key[:-7]} write2 s3 suc: {write_suc}")
            if not write_suc:
                wrong_files.append(hour)
                
        except Exception as e:
            pre_data = None
            logger.error(f'{hour}/{exchange}/{symbol}: Error: {e}')
    
    return wrong_files

if __name__ == '__main__':

    bucket, exchange, symbol, start_hour, end_hour = sys.argv[1:]

    logger = init_log(f'depths1w/{exchange}_{symbol}_{start_hour}_{end_hour}.log')
    wrong_files = getDepths1w(bucket, exchange, symbol, start_hour, end_hour)
    logger.info(f"Final Wrong files: {wrong_files}")