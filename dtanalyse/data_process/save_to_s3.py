import boto3
from botocore.config import Config
import os
from tqdm import tqdm


# 初始化S3资源
s3 = boto3.resource('s3')


def get_files_and_paths(file_path):
    """
    返回指定目录下的所有文件及其路径的字典。
    """
    files_and_paths = {}
    print(file_path)
    for root, dirs, files in os.walk(file_path):
        # print(root, files, dirs)
        for file in files:
            file_path = os.path.join(root, file)
            files_and_paths[file_path] = file
            # print(f'get path: {root}, {file}, {dirs}')
    return files_and_paths


# 上传文件的函数
def upload_file_to_s3(file_name, bucket, object_name=None):
    """上传文件到Amazon S3

    参数:
    file_name -- 要上传的文件的路径
    bucket -- S3中的目标存储桶
    object_name -- S3中的目标文件名，如果没有提供，使用文件的名字
    """

    # 如果object_name没有提供，使用文件的名字
    if object_name is None:
        object_name = file_name.split('/')[-1]

    # 上传文件
    try:
        s3.Bucket(bucket).upload_file(file_name, object_name)
        print(f"{file_name} 已上传至 {bucket}/{object_name}")
    except Exception as e:
        print(f"上传失败: {e}")


def upload_path_to_s3(root_path, local_path, s3_path, file_path, bucket):
    # print(f'{root_path}{local_path}{file_path}')
    files_and_paths = get_files_and_paths(f'{root_path}{local_path}{file_path}')
    # for file, path in files_and_paths.items():
    for _file_path in tqdm(files_and_paths.keys()):
        file = files_and_paths[_file_path]
        _path = _file_path.replace(f'{root_path}{local_path}', "")
        # print(f'uploading: {file}, {_path}, {bucket}')
        if _path == f'{_file_path}{file}':
            continue
        _path = f'{s3_path}{_path}'
        print(f'uploading: {_file_path} ==> {bucket}, {_path}')
        upload_file_to_s3(_file_path, bucket, _path)

        # break
    return


if __name__ == "__main__":
    root_path = "/data/dp-data/"
    local_path = "ModelDatas/"
    s3_path = "ModelDatas/"
    data_paths = [
            # "2024_0701_0731_m/",
            # "2024_0601_0630_m/",
            # "2024_0501_0531_m/",
            # "2024_0401_0430_m/",
            # "2024_0301_0331_m/",
            # "2024_0201_0229_m/",
            # "2024_0101_0131_m/",
            # "2023_1201_1231_m/",
            # "2023_1101_1130_m/",
            # "2023_1001_1031_m/",
            # "2023_0901_0930_m/",
            # "2023_0801_0831_m/",
            "2023_0701_0731_m/",
    ]

    bucket = "dpaimodels"

    index_list = [i for i in range(0, len(data_paths))]
    symbol_path = [
        "sol_usdt_uswap_binance/", 
        "op_usdt_uswap_binance/",
        "fil_usdt_uswap_binance/", 
        "matic_usdt_uswap_binance/",
    ]
    for index in index_list:
        for sp in symbol_path:
            upload_path_to_s3(root_path, local_path, s3_path, f'{data_paths[index]}{sp}', bucket)
    print(f'files for {root_path}{data_paths[index]} are saved into {bucket}')

