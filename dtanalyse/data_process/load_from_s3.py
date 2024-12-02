import boto3
from botocore.config import Config
import os
from tqdm import tqdm


# 初始化S3资源
s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

# 列出S3中指定前缀的所有文件
def list_files_in_s3(bucket, prefix):
    objects = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for object in objects.get('Contents', []):
        key = object.get('Key')
        if key[-1] == '/':  # 跳过目录
            continue
        yield key
 
# 下载S3文件到本地
def download_files(bucket, prefix, local_dir):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    for key in list_files_in_s3(bucket, prefix):
        local_path = os.path.join(local_dir, os.path.basename(key))
        s3_client.download_file(bucket, key, local_path)
        print(f'Downloaded {key} to {local_path}')


if __name__ == "__main__":
    root_path = "/data/dp-data/"
    local_path = "ModelDatas/"
    s3_path = "ModelDatas/"
    data_paths = [
            # "2024_0701_0731_m/",
            # "2024_0601_0630_m/",
            # "2024_0501_0531_m/",
            "2024_0401_0430_m/",
            # "2024_0301_0331_m/",
            # "2024_0201_0229_m/",
            # "2024_0101_0131_m/",
            # "2023_1201_1231_m/",
            # "2023_1101_1130_m/",
            # "2023_1001_1031_m/",
            # "2023_0901_0930_m/",
            # "2023_0801_0831_m/",
            # "2023_0701_0731_m/",
    ]

    bucket = "dpaimodels"
    trade_str = "_uswap"
    exchange = "binance"
    symbol_list = [
        "btc_usdt", 
        "eth_usdt",
        # "sol_usdt",
        # "op_usdt",
        # "fil_usdt",
        # "matic_usdt",
        "bnb_usdt",
        "avax_usdt",
        "bch_usdt",
        "arb_usdt",
    ]

    for data_path in data_paths:
        for symbol in symbol_list:
            _data_path = f'{data_path}{symbol}{trade_str}_{exchange}'
            prefix = f'{s3_path}{_data_path}'
            local = f'{root_path}{local_path}{_data_path}'
            download_files(bucket, prefix, local)
            print(f'files to {local} are loaded from {bucket}')
            
