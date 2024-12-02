import logging
import boto3
import gzip
import json
import pandas as pd
from botocore.config import Config
from datetime import datetime, timedelta
import pytz


# bucket name
BN_DEPTH = "depths"
BN_TICKER = "dpticker"
BN_TRADE = "dp-trade"
BN_INTERN = "dp4intern"


def get_data_froms3(bucket: str, file_key: str):
    
    my_config = Config(
        region_name = 'ap-northeast-1',
    )
    client_s3 = boto3.client('s3', config=my_config)

    try:
        res = client_s3.get_object(Bucket=bucket, Key=file_key)
    except Exception as e:
        if 'The specified key does not exist.' in str(e):
            print(f"{file_key} not found in {bucket}")
        else:
            print(f"{file_key} report error: {e}")
        return []

    content = res['Body'].read()
    # unzip
    ct_dzip = gzip.decompress(content).decode('utf-8')
    
    return ct_dzip.split('\n')


def write_s3(bucket: str, file_content: str, filename_s3: str):
    client_s3 = boto3.client('s3')
    res = client_s3.put_object(
        Bucket=bucket,
        Body=gzip.compress(file_content.encode()),
        Key=filename_s3,
        )
    return res['ResponseMetadata']['HTTPStatusCode'] == 200
    
    
def records_to_ls(records):
    """
    转换json数据为list[dict]
    """
    raw_data_ls = []
    for line in records:
        if line:
            try:
                data = json.loads(line)
            except Exception:
                try:
                    data = eval(line)
                except Exception as e:    # 有些可能会有SyntaxError
                    print(f"Error: {e}; Data: {line}")
                    continue
            try:
                data['tp'] = int(data['tp'])  # 有些数据没有tp字段
            except Exception:
                continue
            raw_data_ls.append(data)
        
    return raw_data_ls


def df_to_records(df):
    """
    将dataframe转回数据存储的json格式
    """
    # dict_series = df.apply(lambda x: x.dropna().to_dict(), axis=1)  # 如果'_'缺失则去掉该字段
    # records = dict_series.to_json(orient='records', lines=True).split('\n')
    
    records = []
    df = df[['p', 's', 't', 'tp', '_']]   # 保证顺序
    for row in df.itertuples(index=False):
        if pd.isna(row._4):
            dict_data = dict(p=row.p, s=row.s, t=row.t, tp=row.tp)
        else:
            dict_data = row._asdict()
            dict_data['_'] = dict_data.pop('_4')
        records.append(dict_data)
    records = pd.Series(records)
    
    return records.to_json(orient='records', lines=True)


def get_hourlist(start_time: str, end_time: str):
    """
    返回输入的时间段内的每个小时的时间str（含左不含右）
    """
    time_format = '%Y%m%d%H'
    start_datetime = datetime.strptime(start_time, time_format)
    end_datetime = datetime.strptime(end_time, time_format)

    hourlist = []
    current_datetime = start_datetime
    while current_datetime < end_datetime:
        hourlist.append(current_datetime.strftime(time_format))
        current_datetime += timedelta(hours=1)

    return hourlist


def get_hour_easteight(tp):
    """
    param: tp: timestamp in ms and UTC
    return: hour str in UTC+8
    """
    dt = datetime.utcfromtimestamp(tp/1e3)
    dt_eight = dt + timedelta(hours=8)
    
    return dt_eight.strftime('%Y%m%d%H')

# def unix2dt(tp, mill_type=1, tz=pytz.utc):
#     return datetime.fromtimestamp(tp/mill_type, tz=tz)

# def dt2str(dt, dt_form='%Y%m%d%H'):
#     return dt.strftime(dt_form)

# def str2dt(dt_str, dt_form='%Y%m%d%H', tz=pytz.utc):
#     return tz.localize(datetime.strptime(dt_str, dt_form))

def init_log(log_file):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    return logger


def _load_json_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_json(file_path, lines=True)
    except Exception as e:
        print(f'get data error: {file_path} for {e}')
        return None

    return df


def load_data_source_from_s3(fn, bucket=BN_TICKER):
    file_path = f's3://{bucket}/{fn}'
    df = _load_json_data(file_path)
    return df


