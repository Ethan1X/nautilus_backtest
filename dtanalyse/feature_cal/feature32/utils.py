import pytz
import boto3
import os
import io
import gzip
import logging
from datetime import datetime, timedelta
from botocore.config import Config


def unix2dt(tp, mill_type=1, tz=pytz.timezone('Asia/Shanghai')):
    return datetime.fromtimestamp(tp/mill_type, tz=tz)


def dt2str(dt, dt_form='%Y%m%d%H'):
    return dt.strftime(dt_form)


def str2dt(dt_str, dt_form='%Y%m%d%H', tz=pytz.timezone('Asia/Shanghai')):
    return tz.localize(datetime.strptime(dt_str, dt_form))


def dt2unix(dt):
    return int(dt.timestamp() * 1000)


def datetime_range(time_begin, time_end, step, include_end=True):
    ret = []
    while time_begin <= time_end:
        ret.append(int(time_begin.timestamp() * 1000))
        time_begin = time_begin + step
    if not include_end:
        ret.pop()
    return ret


def get_date_list(start_date, days):
    date_list = []
    start_date = str2dt(start_date, dt_form='%Y-%m-%d')
    for i in range(days):
        date_list.append(dt2str(start_date + timedelta(days=i), dt_form='%Y-%m-%d'))
    return date_list



def get_data(bucket, file_key):
    """
    直接使用文件路径下载bucket上的文件，适用任意文件路径
    """
    records = []

    my_config = Config(
        region_name='ap-northeast-1'
    )
    client_s3 = boto3.client('s3', config=my_config)

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

    # 下载到本地
    content = res_one['Body'].read()
    # 解压缩出来
    ct_dzip = gzip.decompress(content).decode()
    records = ct_dzip.split('\n')

    return records
