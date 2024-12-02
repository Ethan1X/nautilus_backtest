import os
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import tomli
from tqdm import tqdm

from time_method import *


# 拆解特征文件名
# def extract_feature_name(filename):
#     feat_info, file_type = filename.split()
#     pass

# 生成日期列表
def get_date_list(start, days):
    date_list = []
    _start_date = str2dt(start, dt_form='%Y%m%d')
    for i in range(days+1):
        date_list.append(dt2str(_start_date + timedelta(days=i), dt_form='%Y%m%d'))
    return date_list


# 合并多日（多个指定日期的文件）特征
def merge_features_for_date(start, days, filepath, res_filepath="", file_type="feather", resample_freq="100ms", exchange="", symbol_name=""):
    date_list = get_date_list(start, days)
    # print(f'date_list: {date_list}')
    for day in range(days):
        _start_date = date_list[day]
        _end_date = date_list[day+1]
        merge_features(_start_date, _end_date, filepath, res_filepath, file_type, resample_freq, exchange, symbol_name)


# 合并单日（指定日期的单个文件）特征
def merge_features(start, end, filepath, res_filepath="", file_type="feather", resample_freq="100ms", exchange="", symbol_name=""):
    # 获取对应的文件列表
    date_str = f'{start}'    # _{end}'
    start_ts = str2unix(date_str, DATETIME_FORMAT7, mill_type=1e9, tz=TZ_8)
    symbol_str = ""
    if len(exchange+symbol_name) > 0:
        symbol_str = f'{exchange}_{symbol_name}_'
    file_list = os.listdir(filepath)
    file_list = [file_name for file_name in file_list if date_str in file_name]
    if len(symbol_str) > 0:
        if "uswap" in symbol_name:
            file_list = [file_name for file_name in file_list if symbol_str in file_name]
        else:
            file_list = [file_name for file_name in file_list if (symbol_str in file_name and "uswap" not in file_name)]
    file_list = sorted(file_list)
    print(f'{symbol_str} {len(file_list)} {file_list}')
    if len(file_list) == 0:
        return None, None
        
    feat_file_list = []
    label_file_list = []
    # label_columns = [
    #     "price_change_rate_0.5s", "price_change_rate_1s", "price_change_rate_3s",
    #     "price_change_Trend_0.5s", "price_change_Trend_1s", "price_change_Trend_3s"]
    # time_period_str = ["0.5", "1", "3"]
    time_period_str = ["0.5", "1", "3", "10", "30", "60", "180", "360", "600", "1800", "3600"]
    label_columns = []
    for period in time_period_str:
        label_columns.append(f'price_change_rate_{period}s')
        label_columns.append(f'price_change_trend_{period}s')
        
    rows = 864000
    ms_list = [int(i * 100 * 1e6 + start_ts) for i in range(rows)]
    null_df = pd.DataFrame(index=range(rows))
    null_df = null_df.fillna(np.nan)
    null_df["timestamp"] = ms_list
    null_df.reset_index(inplace=True)
    # print(null_df.shape, null_df)
    
    for file_name in file_list:
        if any(label_name in file_name for label_name in label_columns): 
            label_file_list.append(file_name)
        else:
            feat_file_list.append(file_name)
    print(f'{date_str} {symbol_str}')
    # print(f'feats: {feat_file_list}')
    # print(f'labels: {label_file_list}')

    if len(res_filepath) > 0:
        if not os.path.exists(res_filepath):
            os.makedirs(res_filepath)

    merged_feat_df = None
    # feat_file_list = []
    for file_name in feat_file_list:
        df = read_and_resample(filepath + "/" + file_name, resample_freq)
        if merged_feat_df is None:
            merged_feat_df = df
        else:
            print(f'before merge: {merged_feat_df.keys()} {merged_feat_df.shape[0]}')
            merged_feat_df = pd.merge(merged_feat_df, df, on='timestamp', how="inner")
            print(f'after merge: {merged_feat_df.keys()} {merged_feat_df.shape[0]}')
    if merged_feat_df is not None:        
            merged_feat_df.reset_index(inplace=True)
            # merged_feat_df = merged_feat_df.drop('index', axis=1)
            if file_type == "parquet":
                merged_feat_df.to_parquet(f'{res_filepath}/{symbol_str}feature_{start}.parquet',index = False)
            elif file_type == "feather":
                merged_feat_df.to_feather(f'{res_filepath}/{symbol_str}feature_{start}.feather')
            # else:
            #     merged_feat_df.to_csv(f'{res_filepath}/{symbol_str}feature_{start}.csv',index = False)
            print(f'features merged: {merged_feat_df.columns} {merged_feat_df.shape}')

    merged_label_df = None
    # label_file_list = []
    for file_name in label_file_list:
        df = read_and_resample(filepath + "/" + file_name, resample_freq)
        if merged_label_df is None:
            merged_label_df = df
        else:
            print(f'before merge: {merged_label_df.keys()} {merged_label_df.shape[0]}')
            merged_label_df = pd.merge(merged_label_df, df, on='timestamp', how="inner")
            print(f'after merge: {merged_label_df.keys()} {merged_label_df.shape[0]}')
            # merged_label_df.dropna(inplace=True)
            # print(f'after drop nan: {merged_label_df.keys()} {merged_label_df.shape[0]}')
    if merged_label_df is not None:
        merged_label_df.reset_index(inplace=True)
        # merged_label_df = refine_labels(merged_label_df)
        # merged_label_df.dropna(inplace=True)
        if len(res_filepath):
            full_df = pd.merge(null_df, merged_label_df, on="timestamp", how="left")
            merged_label_df = full_df.drop('index', axis=1)
            # merged_label_df = full_df
            # merged_label_df.to_csv(f'{res_filepath}/{symbol_str}label_test.csv',index = False)
            if file_type == "parquet":
                #merged_label_df.reset_index(inplace=True)
                merged_label_df.to_parquet(f'{res_filepath}/{symbol_str}label_{start}.parquet',index = False)
            elif file_type == "feather":
                merged_label_df.to_feather(f'{res_filepath}/{symbol_str}label_{start}.feather')
            # else:
            # merged_label_df.to_csv(f'{res_filepath}/{symbol_str}label_{start}.csv',index = False)
        if merged_label_df.shape[0] > 0:
            print(f'labels merged: {merged_label_df.columns} {merged_label_df.shape} '
                f'{float(merged_label_df["timestamp"].iloc[0])} {float(merged_label_df["timestamp"].iloc[-1])} \n')
    return merged_feat_df, merged_label_df


# 更新某个特征的多日数据
def update_features_for_date(start, days, feature_list, resample_freq="100ms", feat_path="", merged_feat_path="", file_type="feather", exchange="", symbol_name=""):
    date_list = get_date_list(start, days-1)
    symbol_str = ""
    if len(exchange+symbol_name) > 0:
        symbol_str = f'{exchange}_{symbol_name}_'
        
    for date in date_list:
        merged_feature_fp = f'{merged_feat_path}/{symbol_str}feature_{date}.{file_type}'
        dst_type = os.path.splitext(merged_feature_fp)[-1]
        if dst_type == '.parquet':
            merged_df = pd.read_parquet(merged_feature_fp)
        elif dst_type == '.feather':
            merged_df = pd.read_feather(merged_feature_fp)
        elif dst_type == '.csv':
            merged_df = pd.read_csv(merged_feature_fp)
        else:
            continue
        
        for feat in feature_list:
            feat_fn = f'{feat_path}/{symbol_str}{feat}_{date}.csv'
            if os.path.exists(feat_fn):
                merged_df = update_feature(feat, feat_fn, merged_df, resample_freq)

        if dst_type == '.parquet':
            merged_df.to_parquet(merged_feature_fp)
        elif dst_type == '.feather':
            merged_df.to_feather(merged_feature_fp)
        elif dst_type == '.csv':
            merged_df.to_csv(merged_feature_fp)

    return
    

# 更新某个特征
def update_feature(feature_name, feature_csv, merged_df, resample_freq="100ms"):
    feature_df = read_and_resample(feature_csv, resample_freq)
    if feature_df is None or merged_df is None:
        return merged_df

    if feature_name not in feature_df.columns.values:
        return merged_df    
    
    if feature_name in merged_df.keys():
        # 已有特征，去除旧数据
        del merged_df[feature_name]
    merged_df = pd.merge(merged_df, feature_df, on='timestamp', how="inner")

    return merged_df


# 去除多日数据的特征
def remove_features_for_date(start, days, feature_list, merged_feat_path="", file_type="feather", exchange="", symbol_name=""):
    date_list = get_date_list(start, days-1)
    symbol_str = ""
    if len(exchange+symbol_name) > 0:
        symbol_str = f'{exchange}_{symbol_name}_'
        
    for date in date_list:
        merged_feature_fp = f'{merged_feat_path}/{symbol_str}feature_{date}.{file_type}'
        dst_type = os.path.splitext(merged_feature_fp)[-1]
        if dst_type == '.parquet':
            merged_df = pd.read_parquet(merged_feature_fp)
        elif dst_type == '.feather':
            merged_df = pd.read_feather(merged_feature_fp)
        elif dst_type == '.csv':
            merged_df = pd.read_csv(merged_feature_fp)
        else:
            continue

        merged_df = remove_features(feature_list, merged_df)
                
        if dst_type == '.parquet':
            merged_df.to_parquet(merged_feature_fp)
        elif dst_type == '.feather':
            merged_df.to_feather(merged_feature_fp)
        elif dst_type == '.csv':
            merged_df.to_csv(merged_feature_fp)
    return


# 去除特征
def remove_features(feature_names, merged_df):
    for feature_name in feature_names:
        if feature_name in merged_df.columns.values:
            del merged_df[feature_name]
    return merged_df
    

# 读取特征，并按指定频率采样
def read_and_resample(filepath, resample_freq='100ms'):
    '''
        resample_freq e.g. '0.5s', '1s', '3s'
    '''
    df = pd.read_csv(filepath)
    print(f'{df.keys()} {df.shape} {float(df["time"].iloc[0])} {float(df["time"].iloc[-1])}')
    df['time']  = df['time'].astype('int') // 10**6
    df = df[~df['time'].duplicated(keep='last')]
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
    print(f'resampling: {df.columns} {df.shape} {float(df["time"].iloc[0])} {float(df["time"].iloc[-1])}')
    df = df.resample(resample_freq, on='timestamp').last()
    df.index = df.index.astype(int)
    # df.dropna(inplace=True)
    print(f'resampled: {df.columns} {df.shape} {float(df["time"].iloc[0])} {float(df["time"].iloc[-1])}')
    # print(f'resampled: {df.columns} {df.shape}')
    return df.drop('time',axis=1).ffill()


# 去除无效标注
def refine_labels(label_df, seperated=False, rate=0.9):    
    # 设置保留概率(非保留的全部设置为nan)
    nan_probability = rate
    nan_value = np.nan
    # nan_columns = ["price_change_rate_0.5s", "price_change_rate_1s", "price_change_rate_3s"]
    nan_columns = [col for col in label_df.columns if "price_change_" in col]
    
    # 降采样
    if not seperated:
        zero_mask = (label_df.drop('timestamp', axis = 1) == 0).all(axis=1)
        nan_mask = zero_mask & (np.random.rand(len(label_df)) < nan_probability)
        label_df.loc[nan_mask, nan_columns] = nan_value
    else:
        for col in nan_columns:
            # zero_mask = (label_df.drop('timestamp', axis = 1) == 0).all(axis=1)
            zero_mask = label_df.filter(like=col).eq(0).any(axis=1)
            nan_mask = zero_mask & (np.random.rand(len(label_df)) < nan_probability)
            label_df.loc[nan_mask, col] = nan_value

    return label_df


# 对标注降采样，多日（多个指定日期的文件）
def refine_labels_for_date(start, days, seperated=False, rate=0.9, filepath="", res_filepath="", file_type="feather", exchange="", symbol_name=""):
    date_list = get_date_list(start, days-1)
    # print(f'date_list: {date_list}')
    symbol_str = ""
    if len(exchange+symbol_name) > 0:
        symbol_str = f'{exchange}_{symbol_name}_'

    if not os.path.exists(res_filepath):
        os.makedirs(res_filepath)
        
    for date in date_list:
        _fn = f'{symbol_str}label_{date}'
        _label_file = f'{filepath}/{_fn}.{file_type}'
        if file_type == "parquet":
            label_df = pd.read_parquet(_label_file)
        elif file_type == "feather":
            label_df = pd.read_feather(_label_file)
        else:
            break

        label_df = refine_labels(label_df, seperated, rate)
        _res_file = f'{res_filepath}/{_fn}'
        if file_type == "parquet":
            label_df.to_parquet(f'{_res_file}.{file_type}', index=False)
        elif file_type == "feather":
            label_df.to_feather(f'{_res_file}.{file_type}')
        else:
            label_df.to_csv(f'{_res_file}.csv', index = False)
        print(f'{date} {_res_file} is done')


# 利用回归标注生成分类标注
def convert_labels_for_date(start, days, filepath="", res_filepath="", file_type="feather", exchange="", symbol_name=""):
    date_list = get_date_list(start, days-1)
    # print(f'date_list: {date_list}')
    symbol_str = ""
    if len(exchange+symbol_name) > 0:
        symbol_str = f'{exchange}_{symbol_name}_'

    if not os.path.exists(res_filepath):
        os.makedirs(res_filepath)

    PRICE_FLUCTUATION = 1e-4
    for date in date_list:
        _fn = f'{symbol_str}label_{date}'
        _label_file = f'{filepath}/{_fn}.{file_type}'
        if file_type == "parquet":
            label_df = pd.read_parquet(_label_file)
        elif file_type == "feather":
            label_df = pd.read_feather(_label_file)
        else:
            break

        # print(label_df.columns, label_df)
        # label_df.drop('index', axis=1)
        label_df = label_df.set_index("timestamp")
        mask = label_df > PRICE_FLUCTUATION
        print(mask)
        label_df_up = mask.astype(int)
        label_df_up.columns = [s.replace("rate", "trend") + "_up" for s in label_df.columns]
        print(label_df_up.sum())
        label_df_up.reset_index(inplace=True)
        mask = label_df < -PRICE_FLUCTUATION
        label_df_down = mask.astype(int)
        label_df_down.columns = [s.replace("rate", "trend") + "_down" for s in label_df.columns]
        label_df_down.reset_index(inplace=True)
        # print(label_df_down)

        label_df = pd.merge(label_df_up, label_df_down, on="timestamp", how="inner")
        _res_file = f'{res_filepath}/{_fn}'
        if file_type == "parquet":
            label_df.to_parquet(f'{_res_file}.{file_type}', index=False)
        elif file_type == "feather":
            label_df.to_feather(f'{_res_file}.{file_type}')
        else:
            label_df.to_csv(f'{_res_file}.csv', index = False)
        print(label_df)
        print(f'{date} {_res_file} is done')


timetable = "./tt.toml"

if __name__ == "__main__":
    
    feature_path = ""
    res_path = ""
    symbol_str = ""
    token = ""
    month_str = ""
    starting_date = ""
    period = ""

    # exchange = 'binance'
    
    exchange = ""
    file_type = "parquet"
    # file_type = "feather"
    if len(exchange) > 0: 
        file_type = "feather"
        
    symbol_list = [ "",
                    "btc_usdt_uswap", "eth_usdt_uswap", "arb_usdt_uswap", "matic_usdt_uswap", 
                    "sol_usdt_uswap", "op_usdt_uswap", "fil_usdt_uswap",
                    "bnb_usdt_uswap", "avax_usdt_uswap", "bch_usdt_uswap",
                    "btc_usdt", "eth_usdt", 
                    "sol_usdt", "op_usdt", "fil_usdt",
                    "bnb_usdt", "avax_usdt", "bch_usdt",
                    "arb_usdt", "matic_usdt",
    ]

    merge = False
    refine = False
    seperated = True
    refine_rate = 0.9
    refine_str = ""
    convert = False
    remove = update = False

    with open(timetable, "rb") as fp:
        tt_data = tomli.load(fp)
        feature_path = tt_data.get("feature_path", "")
        res_path = tt_data.get("res_path", "")
        symbol_str = tt_data.get("symbol_str", "")
        token = symbol_str.split("_")[0]
        month_str = tt_data.get("month_str", "")
        starting_date = tt_data.get("start_date", starting_date)
        period = tt_data.get("days", period)
        merge = tt_data.get("merge", False)
        refine = tt_data.get("refine", False)
        if refine:
            refine_rate = float(tt_data.get(["refine_rate"], 0))
            if refine_rate != 0:
                refine_str = f'_{refine_rate}'
            seperated = tt_data.get("seperated", False)
        convert = tt_data.get("convert", False)
        remove = tt_data.get("remove", False)
        update = tt_data.get("update", False)
    print(tt_data)

    # merge_features("20240301", "20240302", feature_path, res_path)
    # right_list = [5, 10, 30, 60, 120, 300, 600, 1000, 2000, 3600, 7200, 14400]
    # feats = [f"hfoiv_{right}" for right in right_list]
    feats = ['LobImbalanceSTD_0_5', 'LobImbalanceSTD_10240_20480', 'LobImbalanceSTD_10_20', 'LobImbalanceSTD_1280_2560',
             'LobImbalanceSTD_160_320', 'LobImbalanceSTD_20_40', 'LobImbalanceSTD_2560_5120', 'LobImbalanceSTD_320_640',
             'LobImbalanceSTD_40_80', 'LobImbalanceSTD_5120_10240', 'LobImbalanceSTD_5_10', 'LobImbalanceSTD_640_1280',
             'LobImbalanceSTD_80_160', 'LobImbalance_0_5', 'LobImbalance_10240_20480', 'LobImbalance_10_20', 'LobImbalance_1280_2560',
             'LobImbalance_160_320', 'LobImbalance_20_40', 'LobImbalance_2560_5120', 'LobImbalance_320_640', 'LobImbalance_40_80',
             'LobImbalance_5120_10240', 'LobImbalance_5_10', 'LobImbalance_640_1280', 'LobImbalance_80_160', 'QuotedSpread_0_5',
             'QuotedSpread_10240_20480', 'QuotedSpread_10_20', 'QuotedSpread_1280_2560', 'QuotedSpread_160_320', 'QuotedSpread_20_40',
             'QuotedSpread_2560_5120', 'QuotedSpread_320_640', 'QuotedSpread_40_80', 'QuotedSpread_5120_10240', 'QuotedSpread_5_10',
             'QuotedSpread_640_1280', 'QuotedSpread_80_160']
    
    for sn in symbol_list[0:1]:
        # csv目录
        _feature_path = f'{feature_path}/{token}_{month_str}'
        # parquet目录
        _res_path = f'{res_path}/{month_str}_m/{symbol_str}'
        # label parquet目录
        label_path = _res_path

        if merge:
            print(f'merging feats for: {token}/{symbol_str} of {month_str}, from {_feature_path} to {_res_path}')
            merge_features_for_date(starting_date, period, _feature_path, _res_path, 
                                    file_type=file_type, exchange=exchange, symbol_name=sn)

        if refine and refine_rate > 0:
            refine_labels_for_date(starting_date, period, seperated=seperated, rate=refine_rate, 
                                   filepath=label_path, res_filepath=f'{_res_path}/label_process{refine_str}',
                                   file_type=file_type, exchange=exchange, symbol_name=sn)

        if update and len(feats) > 0:
            update_features_for_date(starting_date, period, feature_list=feats, resample_freq="100ms", 
                                    feat_path=_feature_path, merged_feat_path=_res_path,
                                    file_type=file_type, exchange=exchange, symbol_name=sn)

        if remove and len(feats) > 0:
            remove_features_for_date(starting_date, period, feature_list=feats, merged_feat_path=_res_path,
                                    file_type=file_type, exchange=exchange, symbol_name=sn)
    
        if convert:
            convert_labels_for_date(starting_date, period,
                                   filepath=label_path, res_filepath=_res_path,
                                   file_type=file_type, exchange=exchange, symbol_name=sn)


