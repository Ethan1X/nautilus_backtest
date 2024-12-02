import sys
import json
import boto3
import gzip
import logging
import traceback
from botocore.config import Config
import pandas as pd
from collections import defaultdict

if '../' not in sys.path:
    sys.path.append('../')

from util.time_method import *
from conf.servers import cnn_mongo_client

'''
    一些获取数据的方法通用方法：从S3获取或者从公共的mongo库中获取
'''
BUCKET_LIST = ['depths','dpticker','dp-trade', 'dpotherdatas']

def get_data_frombucket(bucket,datetime,exchange,symbol,wrong=True,cache=None):
    """
    author ： 王培
    data : 2022/07/20
    input : 数据桶，下载数据的文件信息，时间、交易所、币对
    output : records list格式内部是数据dict {"p":20547.09,"s":0,"t":"sell","tp":1657861195960}
    """
    # 先从缓冲中查看，如果缓冲中有，不需要拉取
    if cache is not None:
        key = get_cache_key(bucket, exchange, symbol, datetime)
        ca_data = get_data_from_cache(key, cache)
        if ca_data is not None:
            # print(f"本次数据从缓冲中获取: {bucket} {exchange} {symbol} {datetime}")
            return ca_data

    records = []
    file_key = "{}/{}/{}.log.gz".format(datetime,exchange,symbol)
    
    my_config = Config(
        region_name = 'ap-northeast-1'
    )
    client_s3 = boto3.client('s3', config = my_config)
    
    try:
        res_one = client_s3.get_object(
            Bucket=bucket,
            Key=file_key,
        )
    except Exception as e:
        if wrong:
            if "The specified key does not exist" in str(e):
                print(f"{file_key} 不在当前bucket中")
            else:
                print(f"{file_key} 从bucket中获取数据出现异常 {e}")
            
        return records
    
    #下载到本地
    content = res_one['Body'].read()
    #解压缩出来
    ct_dzip = gzip.decompress(content).decode()
    records = ct_dzip.split('\n')
    ret = ret_list_trans(records)
    # 保存数据到缓存中
    if cache is not None:
        cache[key] = ret 
    return ret

def get_cache_key(data_type, exchange, symbol, s3_dt):
    return f"{data_type}-{exchange}-{symbol}-{s3_dt}"

def get_data_from_cache(key, cache):
    # 每访问一次cache，查看删除冗余缓存
    del_cache_due_data(key.split('-')[-1], cache)
    return cache.get(key)

def del_cache_due_data(s3_dt, cache):
    '''
        删除距离当前（s3_dt）较远的s3数据缓冲
    '''
    due_keys = []
    for k in cache.keys():
        if str2dt(s3_dt, DATETIME_FORMAT4) - str2dt(k.split('-')[-1], DATETIME_FORMAT4) > datetime.timedelta(hours=2):
            due_keys.append(k)
    # print(f"删除缓冲中的过期key: {due_keys}")

    for key in due_keys:
        del cache[key]
    return

def ret_list_trans(ret_list):
    '''
        将结果从str转变为dict
    '''
    ret = []
    for row in ret_list:
        if row != '':
            try:
                ret.append(json.loads(row))
            except Exception as e:
                print(f"遇到异常跳过: {row}")
    #return [json.loads(row) for row in ret_list if row != '']
    return ret

def trans_list2df(ret_list):
    '''
        将从s3拉到的数据保存为dataframe格式
    '''
    return pd.DataFrame([json.loads(row) for row in ret_list if row != ''])
    
def query_dict_list(data_list, name, gte=None, lte=None, eq=None):
    '''
        从返回的list[dict, dict]中查找等于某个
        params:
            name: 检索字段
            # tp: str 如果检索字符串类型 num 如果检索数字类型
    '''
    ret = []
    for row in data_list:
        if eq is not None:
            if row[name] == eq:
                ret.append(row)
            continue
        elif gte is not None and lte is not None:
            if row[name] >= gte and row[name] <= lte:
                ret.append(row)
            continue
        elif gte is not None:
            if row[name] >= gte:
                ret.append(row)
            continue
        elif lte is not None:
            if row[name] <= lte:
                ret.append(row)
            continue
    return ret

def get_symbol_list(bucket,time,exchange='binance'):
    """
    author:刘雨嫣
    time：20220823
    作用：得到规定时间、交易所下的所有数据文件路径
    input:
        1. bucket:如果是depth用'depths'，是ticker用‘dpticker'
        2. time :如‘2022082009’
        3. exchange：交易所，
    output：
        该时间、交易所底下的所有symbol的list 
        >>>  ['1inch_usdt','aave_btc', 'aave_eth','aave_usdt', 'abt_usdt']
    输入实例：get_symbol_list('depths',"2022082209",'okex')
    """
    list_name=[]
    prefix = "{0}/{1}/".format(time,exchange)
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(name=bucket)
    FilesNotFound = True
    for obj in bucket.objects.filter(Prefix=prefix):
        list_name.append(obj.key)
        #print('{0}'.format(obj.key))
    FilesNotFound = False
    if FilesNotFound:
        print("ALERT", "No file in {0}/{1}".format(bucket, prefix))
        print(prefix)
    symbol=[]
    for i in list_name:
        x=i.split("/")
        m=x[2].split(".")
        symbol.append(m[0])
    return symbol


def trade_merge(trade_data):
    '''
        对trade数据进行聚合，将时间相同的订单聚合到一起，方便绘图的时候看清真实的交易情况
          展示聚合了多少笔交易，聚合后的price_avg 与 聚合后的amount_sum，及本次聚合中price的范围
    '''
    tk_data_new = []
    last_tk_info = {'T': None, 'm': None, 'price_min': None, 'price_max': None, 'amt_sum': 0, 'price_avg': None, 'b': None, 'a': None, 'tp': None, 't_num': 0}

    for row in trade_data:
        # 初始化
        if last_tk_info['T'] is None:

            last_tk_info = {
                'T': row['T'],
                'm': row['m'],
                'price_min': row['p'],
                'price_max': row['p'],
                'amt_sum': row['q'],
                'price_avg': row['p'],
                'b': row['b'],
                'a': row['a'],
                'tp': row['tp'],
                't_num': 1
            }
            continue

        if row['T'] != last_tk_info['T'] or row['m'] != last_tk_info['m']:
            tk_data_new.append(last_tk_info)
            last_tk_info = {
                'T': row['T'],
                'm': row['m'],
                'price_min': row['p'],
                'price_max': row['p'],
                'amt_sum': row['q'],
                'price_avg': row['p'],
                'b': row['b'],
                'a': row['a'],
                'tp': row['tp'],
                't_num': 1,
            }
        else:
            last_tk_info['price_min'] = min(last_tk_info['price_min'], row['p'])
            last_tk_info['price_max'] = max(last_tk_info['price_max'], row['p'])
            last_tk_info['amt_sum'] = last_tk_info['amt_sum'] + row['q']
            last_tk_info['price_avg'] = (last_tk_info['price_avg'] * last_tk_info['amt_sum'] + row['p'] * row['q']) / (last_tk_info['amt_sum'] + row['q'])
            last_tk_info['tp'] = row['tp']
            last_tk_info['t_num'] += 1

    if len(tk_data_new) > 0 and last_tk_info['T'] is not None and (tk_data_new[-1]['T'] != last_tk_info['T'] or tk_data_new[-1]['m'] != last_tk_info['m']):
        tk_data_new.append(last_tk_info)

    return tk_data_new


def classify_depth_delete(depth_old, recover_depth):
    '''
        对depth中每档数据进行划分，一共有三类：1）前一个tp就存在的depth档位；2）本tp变化的档位；3）本tp删除的档位
        returns:
           depth_delete_ask: list[dict] 本tp删除的档位 [{'asks': 3.2,'time':int(time.time()*1000)}, {'asks': 3.2,'time':int(time.time()*1000)}] asks: 保存的是price
    '''
    # 初始化结果数据集
    depth_delete_ask = []
    depth_delete_bid = []
    e_list=[]
    t_list=[]
    p_list=[]
    time_list=[]
    p_ask_max_list=[]
    p_ask_min_list=[]
    p_bid_max_list=[]
    p_bid_min_list=[]
    #在depth_old中查找's'等于0的数
    for data in depth_old:   
        if data['s'] == 0:
            e_list.append(data['e'])
            t_list.append(data['t'])
            p_list.append(data['p'])                
    for item in recover_depth:
        time_list.append(item['time'])
        asks_list=item['asks']
        bids_list=item['bids']
        p_ask_max_list.append(asks_list[4]['p'])
        p_ask_min_list.append(asks_list[0]['p'])
        p_bid_min_list.append(bids_list[4]['p'])
        p_bid_max_list.append(bids_list[0]['p'])
    for j in range(len(time_list)):
        for i in range(len(e_list)):
            if e_list[i]==time_list[j]:  
                if t_list[i]=='sell':
                    if p_list[i]<=p_ask_max_list[j] and p_list[i]>= p_ask_min_list[j]:
                        depth_dict = {'asks': p_list[i],'time':e_list[i]}
                        depth_delete_ask.append(depth_dict)
                if t_list[i]=='buy':
                    if p_list[i]>=p_bid_min_list[j] and p_list[i]<= p_bid_max_list[j]:
                        depth_dict = {'bids': p_list[i],'time':e_list[i]}
                        depth_delete_bid.append(depth_dict)

    return depth_delete_ask,depth_delete_bid


def classify_depth_change_exist(recover_depth):
    '''
        对depth中每档数据进行划分，一共有三类：1）前一个tp就存在的depth档位；2）本tp变化的档位；3）本tp删除的档位

        returns:
           depth_change_ask: list[dict] 本tp变动的档位 [{'asks':{'p':1.23,'s';0.23},'time':int(time.time()*1000),'add':'add'}, ...] 根据tp+p 展开为list
           depth_exist_ask: list[dict] 本tp变动的档位 [{'asks':{'p':1.23,'s';0.23},'time':int(time.time()*1000)}, ...] 根据tp+p 展开为list
    '''
    # 初始化结果数据集
    depth_change_ask= []
    depth_change_bid= []
    depth_exist_ask = []
    depth_exist_bid = []
    #寻找新增的数据
    for i in range(len(recover_depth)):
        if i< len(recover_depth)-1:
            tp=recover_depth[i]['time']
            tp_next=recover_depth[i+1]['time']
            
            asks_list=recover_depth[i]['asks']
            asks_list_next=recover_depth[i+1]['asks']
            
            bids_list=recover_depth[i]['bids']
            bids_list_next=recover_depth[i+1]['bids']
            for ask_dict_next in asks_list_next:
                for ask_dict in asks_list:
                    if ask_dict['p']==ask_dict_next['p']:
                        if ask_dict['s']!=ask_dict_next['s']:
                            depth_change_ask.append({'asks':ask_dict_next,'time':tp_next,'add':'add'})
                        if ask_dict['s']==ask_dict_next['s']:
                            depth_exist_ask.append({'asks':ask_dict_next,'time':tp_next})
                        break
                else:
                    depth_change_ask.append({'asks':ask_dict_next,'time':tp_next,'add':'add'})
            for bid_dict_next in bids_list_next:
                for bid_dict in bids_list:
                    if bid_dict['p']==bid_dict_next['p']:
                        if bid_dict['s']!=bid_dict_next['s']:
                            depth_change_bid.append({'bids':bid_dict_next,'time':tp_next,'add':'add'}) 
                        if bid_dict['s']==bid_dict_next['s']:
                            depth_exist_bid.append({'bids':bid_dict_next,'time':tp_next})
                        break
                else:
                    depth_change_bid.append({'bids':bid_dict_next,'time':tp_next,'add':'add'})
    return depth_change_ask,depth_change_bid,depth_exist_ask,depth_exist_bid

