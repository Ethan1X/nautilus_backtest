# -*- coding: utf-8 -*-
"""
检验旧数据并清洗转存主文件
author: lifangyu
"""

from recover_sig import *
from s3_utils import *
import sys
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning)
new_bucket = 'cleanolddepths'


def process_one_hour(bucket: str, hour: str, exchange: str, symbol: str, to_exchange=None, to_symbol=None):
    """
    处理1h数据的完整过程
    【只转存没问题的数据】
    params:
        bucket: 原始数据所在的bucket
        hour: 1h的时间戳
        exchange: 交易所
        symbol: 交易对
    return:
        True: 处理文件成功
        False: 添加全量标识/修改撤单逻辑有问题，记录对应文件名
    
    可能出现的问题：
    1. 数据少于1w条，不存 
    2. 数据量不足，包含的时间少于45min数据，不存
    3. 文件名对不上，按照数据对应的东八区时间存，可能会存在被覆盖问题，无法进一步处理
    4. 找到的全量标识不足4个
    5. 修改撤单逻辑失败（目前不存在）
    """
    if to_exchange is None:
        to_exchange = exchange
    if to_symbol is None:
        to_symbol = symbol
    
    has_sig, has_s0 = False, False

    # 读取原始数据
    old_file_key = f"{hour}/{exchange}/{symbol}.log.gz"
    records = get_data_froms3(bucket, old_file_key)
    if not records:
        logger.info(f"!!{old_file_key[:-7]} not found in {bucket}!!")
        return True
    print(f"Current processing: {old_file_key}")
    
    # 转换为list[dict]
    raw_data_ls = records_to_ls(records)
    
    # 如果原始数据少于1w条，不存
    if len(raw_data_ls) < 10000:
        logger.info(f"!!{old_file_key[:-7]} has less than 10000({len(raw_data_ls)}) records!!")
        return True
    
    # 保证数据起码有45min-1h的量
    total_min = (raw_data_ls[-1]['tp'] - raw_data_ls[0]['tp']) // (60*1000)
    if total_min < 45:
        logger.info(f"!!{old_file_key[:-7]} has less than 45min({total_min}) records!!")
        return True
    
    # 判断是否有s=0和全量标识
    for data in raw_data_ls:
        try:
            if data['s'] == 0:
                has_s0 = True
            if '_' in data.keys():
                has_sig = True
            if has_s0 and has_sig:
                break
        except Exception as e:
            logger.info(f"{old_file_key[:-7]} read Error: {e}; Data: {data}")
            continue
        
    # 判断时间和命名是否对的上
    curr_hour = get_hour_easteight(raw_data_ls[len(raw_data_ls)//2]['tp'])   # 取中间的时间戳
    if curr_hour != hour:
        logger.error(f"!!recordedHour: {hour}, actualHour: {curr_hour}!!")
    new_file_key = f"{curr_hour}/{to_exchange}/{to_symbol}.log.gz"
    
    # rec_hour_dt = str2dt(hour)
    # curr_hour_dt = unix2dt(raw_data_ls[len(raw_data_ls)//2]['tp'], mill_type=1e3).replace(minute=0, second=0, microsecond=0) \
    #     + timedelta(hours=8)
    # if rec_hour_dt != curr_hour_dt:
    #     diff_hour = (curr_hour_dt - rec_hour_dt).total_seconds() / 3600
    #     logger.error(f"!!recordedHour: {dt2str(rec_hour_dt)}, actualHour: {dt2str(curr_hour_dt)}, diff: {diff_hour}!!")
    # new_file_key = f"{dt2str(curr_hour_dt, '%Y%m%d%H')}/{to_exchange}/{to_symbol}.log.gz"
    
    # 原文件只需要统一数据格式【引号】【去掉u前缀】【其他的格式问题】即可转存
    if has_s0 and has_sig:
        new_records = pd.Series(raw_data_ls).to_json(orient='records', lines=True)
        write_suc = write_s3(new_bucket, new_records, new_file_key)
        logger.info(f"{new_file_key[:-7]} has s=0 & sig, no need to process, suc: {write_suc}")
        return write_suc
    
    # 处理数据，添加全量标识&修改撤单逻辑
    df = pd.DataFrame(raw_data_ls)
    if not has_sig:
        df, tp_count = add_sig(df, has_s0)
        if tp_count not in [4, 5, 6]:
            logger.error(f"!!{new_file_key[:-7]} only has {tp_count} tp with sig!!")
            return False
    if not has_s0:
        if '_' not in df.columns:
            logger.error(f"!!{new_file_key[:-7]} has no sig after processing!!")
            return False
        df = reset_s0(df)
        
    # 转为json并写入s3
    new_records = df_to_records(df)
    write_suc = write_s3(new_bucket, new_records, new_file_key)
    logger.info(f"{new_file_key[:-7]} write2 s3 suc: {write_suc}， has_sig: {has_sig}, has_s0: {has_s0}")

    return write_suc


if __name__ == '__main__':    
    
    bucket = 'deeptradingbooks'
    # exchange = 'binance'
    # symbol = 'eth_usdt'
    # start_hour = '2021010100'
    # end_hour = '2021010101'
    # 获取操作文件的绝对目录file_path = sys.argv[0]
    exchange, symbol, start_hour, end_hour, to_exchange, to_symbol = sys.argv[1:]
    
    logger = init_log(f'{exchange}_{symbol}_{start_hour}_{end_hour}.log')

    wrong_file_ls = []
    hourlist = get_hourlist(start_hour, end_hour)
    for hour in hourlist:
        suc = process_one_hour(bucket, hour, exchange, symbol, to_exchange, to_symbol)
        if not suc:
            wrong_file_ls.append(hour)
            
    logger.info(f"Final Wrong files: {wrong_file_ls}")
    
    
    
    