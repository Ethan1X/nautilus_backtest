'''
    新架构中常用统计方法
'''
import os, sys, datetime
from collections import defaultdict
import numpy as np

sys.path.append('../')

from util.time_method import *

def get_price(db_pub, cal_time):
    '''
        将price整理为以exchange-symbol_type为key，以每个symbol对应的price为value的dict
        {
            'BINANCE-SPOT_NORMAL':{
                'DOGE/BUSD': {'ask': 0.07193, 'bid': 0.07192},
            },
            'BINANCE-SWAP_FOREVER':{
                'BTC/USDT': {'ask': 24999.3, 'bid': 24999.2},
            },
        }
    '''
    price_info = defaultdict(lambda: defaultdict(lambda: {'ask': None, 'bid': None}))
    # 现货中间价单独保存，用于币不平计算及净值结算使用：如果有币安价格就优先使用币安价格
    spot_price_info = defaultdict(float)

    price_ret = db_pub['pubinfo']['time_prices'].find_one({'time': {'$lte': cal_time}}, sort=[('time', -1)])
    # 对时效性做验证
    if price_ret is None:
        print(f"未获取到{cal_time}的price")
        return None, None
    if cal_time - price_ret['time'].replace(tzinfo=TZ_0) > datetime.timedelta(seconds=60*5):
        print(f"price 时效性{price_ret['time']}不符合要求")
        return None, None

    for p_info in price_ret['prices']:
        price_info[f"{p_info['exchange']}-{p_info['symbol_type']}"][p_info['symbol']] = {'ask': p_info['ask'], 'bid': p_info['bid']}
        if p_info['symbol_type'] == 'SPOT_NORMAL':
            if spot_price_info.get(p_info['symbol']) is None or (spot_price_info.get(p_info['symbol']) is not None and p_info['exchange'] == 'BINANCE'):
                spot_price_info[p_info['symbol']] = (p_info['ask'] + p_info['bid']) / 2
    
    # 补充稳定币价格
    spot_price_info['USDT/USDT'] = 1
    return price_info, spot_price_info

def get_forex_exchange_price(forex_token, spot_price_info, tokens=['BTC', 'ETH']):
    '''
        获取汇率的交易所间USDT价格: 1 个 forex_token 等于多少个USDT
    '''
    forex_price_list = []
    for token in tokens:
        forex_price_list.append(spot_price_info[f'{token}/USDT'] / spot_price_info[f'{token}/{forex_token}'])
    if len(forex_price_list) != 0:
        return np.mean(forex_price_list)
    else:
        return None

def describe_series(se):
    return se.describe(percentiles=[.05,.15,.25,.35,.45,.5,.55,.65,.75,.85,.95,.99])

# mongo建立索引
def create_index(db, table, col, index_sort:int):
    '''
        params:
            index_sort: 1 增序排列；-1 降序排列
    '''
    index_name = f'{col}_{str(index_sort)}'
    if not index_name in db[table].index_information():
        print(f"索引：{index_name} 不在{table}中，开始生成索引")
        db[table].create_index([(col, index_sort)])
        print(f"索引:{col} 生成完毕")
    return

# 保存数据到mongo
def save2mongo(db, table, data:list, update_num=10000, update_col='time'):
    '''
        params:
            data: 由dict组成的list
            update_num: 默认一次更新时的数量
            update_col: 更新时基于的筛选条件
    '''
    if len(data) == 0:
        return

    t1 = time.time()
    def update_mongo(update_items, first_update_col_value):
        if first_update_col_value is None:
            first_update_col_value = update_items[0][update_col]
            update_items = update_items[1:]
        last_update_col_value = update_items[-1][update_col]
        db[table].delete_many({update_col: {'$gt': first_update_col_value, '$lte': last_update_col_value}})
        # db[table].update_many({update_col: {'$gt': first_update_col_value, '$lte': last_update_col_value}}, {'$set': update_items}, upsert=True)
        db[table].insert_many(update_items)
        print(f"已更新: {first_update_col_value}-{last_update_col_value} 之间的数据")

        return last_update_col_value

    update_items = []
    first_update_col_value = None

    for item in data:
        update_items.append(item)

        if len(update_items) > update_num:
            first_update_col_value = update_mongo(update_items, first_update_col_value)
            update_items = []

    if len(update_items) > 0:
        update_mongo(update_items, first_update_col_value)
    print(f"本次保存平均每1000条数据保存耗时: {(time.time() - t1) / len(data) * 1000}s")
    return
