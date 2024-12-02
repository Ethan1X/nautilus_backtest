import time
import datetime
import json
import copy
import boto3
import gzip
from botocore.config import Config

def get_data(bucket, file_key):
    """
    直接使用文件路径下载bucket上的文件，适用任意文件路径
    """
    records = []
    
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
    
    return records


def update_bids(res, bids_p):
    """
    用res更新bids_p
    """
    for i in res:
        bid_price = i['p']
        # 测试使用二分查找的运行效率
        l, r = 0, len(bids_p) - 1
        while l <= r:
            mid = (l + r) // 2
            if bids_p[mid]['p'] == bid_price:
                if i['s'] == 0:     #p一致，s=0，则去掉depths中的这条数据
                    del bids_p[mid]
                else:       #p一致，s！=0，则用depth——update的这条数据替换depths中的数据
                    bids_p[mid]['s'] = i['s']
                break
            if bids_p[mid]['p'] > bid_price:
                l = mid + 1
            else:
                r = mid - 1
        if l > r and i['s'] != 0:
            bids_p.insert(l, i)

def update_asks(res, asks_p):
    """
    用res更新bids_p
    """
    for i in res:
        ask_price = i['p']
        # 测试使用二分查找的运行效率
        l, r = 0, len(asks_p) - 1
        while l <= r:
            mid = (l + r) // 2
            if asks_p[mid]['p'] == ask_price:
                if i['s'] == 0:     #p一致，s=0，则去掉depths中的这条数据
                    del asks_p[mid]
                else:       #p一致，s！=0，则用depth——update的这条数据替换depths中的数据
                    asks_p[mid]['s'] = i['s']
                break
            if asks_p[mid]['p'] < ask_price:
                l = mid + 1
            else:
                r = mid - 1
        if l > r and i['s'] != 0:
            asks_p.insert(l, i)

def update_depth(depth_update, data_one):
    """
    将新数据（data_one)添加到depth_update中，准备用于更新depth
    """
    del data_one['e']
    del data_one['tp']
    if data_one['t']=='buy':
        depth_update['bids'].append(data_one)
    else:
        depth_update['asks'].append(data_one)
    del data_one['t']


def match_depth(depth):
    """
    如果存在盘口重叠（买价高于卖价），则进行撮合，递归用买一卖一进行消除，size小的订单删除，大的订单更新size
    """
    if len(depth['asks']) == 0 or len(depth['bids']) == 0:
        return True
    
    ask_p, ask_s = depth['asks'][0]['p'], depth['asks'][0]['s']
    bid_p, bid_s = depth['bids'][0]['p'], depth['bids'][0]['s']
    if ask_p <= bid_p:
        # print(f"存在ask1<bid1的情况，ask1={ask_p}, bid1={bid_p}, tp={curr_tp}")
        if ask_s == bid_s:
            depth['asks'].pop(0)
            depth['bids'].pop(0)
        elif ask_s < bid_s:
            depth['asks'].pop(0)
            depth['bids'][0]['s'] = bid_s - ask_s
        else:
            depth['asks'][0]['s'] = ask_s - bid_s
            depth['bids'].pop(0)
        return match_depth(depth)
    return True
    

def recoveryDepth(exchange, symbol, date_hour, head_num=20, pre_data=None, bucket='depths'):
    print(f"begin to recover depth at date {date_hour}: {datetime.datetime.now()}")
    file_key = "{}/{}/{}.log.gz".format(date_hour, exchange, symbol)
    origin_data = get_data(bucket, file_key)
    if not origin_data:
        return [], []
    
    raw_data_list = [json.loads(data) for data in origin_data if data != ""]
    # 如果有e字段且不为0，则以e为标准；否则按tp
    if 'e' in raw_data_list[0] and raw_data_list[0].get('e') != 0:
        time_type = 'e'
    else:
        time_type = 'tp'
    sorted_list = sorted(raw_data_list, key=lambda x: x[time_type])
    
    start = 0
    res = []
    depth_update = {'bids':[],'asks':[]}
    # pre_data为空，表示是第一个小时的数据，要先跳过数据最前面不是全量的部分，找到第一条全量推送
    # 否则，pre_data传入的是上一个小时最后一个时刻的depth恢复全量的数据
    if not pre_data:
        while '_' not in sorted_list[start]:
            start += 1
        pre_data = {'bids':[], 'asks':[]}
    depth_time = 0

    for i in range(start, len(sorted_list)):
        if not sorted_list[i]:
            continue
        try:
            data_one = sorted_list[i]
        except Exception as e:
            print(e)
            print(origin_data[i])
            continue
        if data_one[time_type] != depth_time:
            if depth_update['asks'] and depth_update['bids'] and '_' in depth_update['asks'][-1] and '_' in depth_update['bids'][-1]:
                ask_index = bid_index = 0
                while '_' not in depth_update['asks'][ask_index]:
                    ask_index += 1
                while '_' not in depth_update['bids'][bid_index]:
                    bid_index += 1
                pre_data = {'bids': depth_update['bids'][bid_index:],
                            'asks': depth_update['asks'][ask_index:]}
            else:
                update_bids(depth_update['bids'], pre_data['bids'])
                update_asks(depth_update['asks'], pre_data['asks'])
            
            # 检查盘口重叠
            match_depth(pre_data)
            if depth_time > 0 and (pre_data['asks'] or pre_data['bids']):
                one = {'bids': copy.deepcopy(pre_data['bids'][:head_num]),
                    'asks': copy.deepcopy(pre_data['asks'][:head_num]),
                    'time': depth_time}
                res.append(one)
            depth_time = data_one[time_type]
            depth_update = {'bids':[],'asks':[]}
        update_depth(depth_update, data_one)

    update_bids(depth_update['bids'], pre_data['bids'])
    update_asks(depth_update['asks'], pre_data['asks'])
    # 检查盘口重叠
    match_depth(pre_data)
    one = {'bids': copy.deepcopy(pre_data['bids'][:head_num]),
            'asks': copy.deepcopy(pre_data['asks'][:head_num]),
            'time': depth_time}
    res.append(one)
    print(f"finish depth recovery at {datetime.datetime.now()}")
    return res, pre_data

def doCalcAP(ps_list, price):
    if not ps_list:
        return 0
    amount_sum = size = 0
    for i in ps_list:
        cur_size = i['s']
        cur_amount = i['p'] * i['s']
        if amount_sum + cur_amount < price:
            size += cur_size
            amount_sum += cur_amount
        else:
            size += (price - amount_sum) / i['p']
            break
    else:
        return amount_sum / size
    return price / size

def calcAveragePrice(depth, price):
    for depth_per_time in depth:
        ask_avg_pirce = doCalcAP(depth_per_time['asks'], price)
        bid_avg_price = doCalcAP(depth_per_time['bids'], price)
        depth_per_time['ask_ap'] = ask_avg_pirce
        depth_per_time['bid_ap'] = bid_avg_price


if __name__ == '__main__':
    t1 = time.time()
    data, _ = recoveryDepth('binance', 'btc_usdt', '2023051910', 10)
    print(f'耗时: {round(time.time() - t1, 3)}s {len(data)}')
    print(data[:3])
