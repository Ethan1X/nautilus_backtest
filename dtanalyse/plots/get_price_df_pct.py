
'''
  根据不同交易所不同币对获取价差图
'''
import os, sys, datetime, logging
import plotly.graph_objs as go
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from util.s3_method import * 
from util.time_method import *
from util.plot_method import easy_plot
from util.hedge_log import initlog
from util.plot_method import timeline_sample_kv, get_plot_diff_data
from util.recover_depth import recoveryDepth
from util.statistic_method_v2 import describe_series

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
def get_price_df_pct_list(begin_time,end_time,exchange_ask="binance",exchange_bid="okex",symbol_ask="btc_usdt",symbol_bid = "eth_usdt",bucket='dpticker',is_adj_time=True):
        '''
        author：张鑫健
        input：begin_time,end_time,exchange_ask=,exchange_bid,symbol_ask,symbol_bid ,bucket,is_adj_time
        output:data_ask,data_bid
        输出格式：[{'tp':1672503599998,'aa':0.1344,'ap':1.13432,'ba':12.3321,'bp':0.2343,'e':1672503599998}]
        data_ask,data_bid数据是获取的两个交易所的所需时间的特定币对的数据
        '''
        #判断是否进行操作，解决前十分钟或后十分钟问题
        if is_adj_time:
            begin_time_new =s3_time_adj(begin_time, 'begin_time')
            end_time_new = s3_time_adj(end_time, 'end_time')
        else:
            begin_time_new, end_time_new = begin_time, end_time

        begin_time_ = begin_time_new
        s3_dt = dt2str(begin_time_, DATETIME_FORMAT4)
        data_ret_ask=[]
        data_ret_bid=[]
#         cache={}
        #读取两个交易所两个币对的数据
        while s3_dt <= dt2str(end_time_new, DATETIME_FORMAT4):
            data_ret_1 = get_data_frombucket(bucket,s3_dt,exchange_ask, symbol_ask)
            data_ret_2 = get_data_frombucket(bucket,s3_dt,exchange_bid, symbol_bid)
            logging.info(f"获取 {exchange_ask}_{symbol_ask} cex ticker数据: {s3_dt} {len(data_ret_1)}")
            logging.info(f"获取 {exchange_bid}_{symbol_bid} cex ticker数据: {s3_dt} {len(data_ret_2)}")
            if len(data_ret_1) == 0:
                logging.warning(f"cex:symbol_ask_{exchange_ask}:{symbol_ask} {begin_time_} 所在hour无depth数据")
                begin_time_ += datetime.timedelta(hours=1)
                s3_dt = dt2str(begin_time_, DATETIME_FORMAT4)
                continue      
            if len(data_ret_2) == 0:
                logging.warning(f"cex:symbol_bid_{exchange_bid}:{symbol_bid} {begin_time_} 所在hour无depth数据")
                begin_time_ += datetime.timedelta(hours=1)
                s3_dt = dt2str(begin_time_, DATETIME_FORMAT4)
                continue 
                
            begin_time_ += datetime.timedelta(hours=1)
            s3_dt = dt2str(begin_time_, DATETIME_FORMAT4)
            data_ret_ask.extend(data_ret_1)
            data_ret_bid.extend(data_ret_2)
        #对数据切片
        data_ask = query_dict_list(data_ret_ask, 'tp', gte=dt2unix(begin_time, 1000), lte=dt2unix(end_time, 1000))
        data_bid = query_dict_list(data_ret_bid, 'tp', gte=dt2unix(begin_time, 1000), lte=dt2unix(end_time, 1000))
        
        return data_ask,data_bid




def get_price_df_pct_sorted(data_ask,data_bid,plot_interval_us=1000):
    '''
    author：张鑫健
    input：data_ask,data_bid,plot_interval_us是降采样间隔时间
    output:price_df_pct
    输出格式：[{'price_df_pct':0.1234,'tp':1672503599998}]
    price_df_pct数据是每个tp下经过降采样得到的价差百分比
    '''
    price_ret_1 = []
    price_df_pct=[]
    price_df_pct_list=[]
    #将数据进行合并
    combine_list = data_ask + data_bid
    #对数据按照tp排序
    sorted_list = sorted(combine_list,key=lambda d: d['tp'])
    for item in sorted_list:
        if item in data_ask:
            item['source']='ask'
        if item in data_bid:
            item['source']='bid'
    i=len(sorted_list)-1
    #找到每个数据离得最近的不同bid/ask的数据，求出价差比
    while i>0:
        if sorted_list[i]['source']=='ask':
            for j in range(1,i):
                if sorted_list[i-j]['source']=='bid':
                    value_1=(float(sorted_list[i]['bp'])-float(sorted_list[i-j]['ap']))/(float(sorted_list[i-j]['ap'])*float(sorted_list[i-j]['aa']))
                    price_df_pct_list.append({'price_df_pct':value_1,'tp':sorted_list[i]['tp']})
                    break
                if sorted_list[i-j]['source']=='ask':
                    continue
            i=i-1
        if sorted_list[i]['source']=='bid':
            for j in range(1,i):
                if sorted_list[i-j]['source']=='ask':
                    value_2=(float(sorted_list[i-j]['bp'])-float(sorted_list[i]['ap']))/(float(sorted_list[i]['ap'])*float(sorted_list[i]['aa']))
                    price_df_pct_list.append({'price_df_pct':value_2,'tp':sorted_list[i]['tp']})
                    break
                if sorted_list[i-j]['source']=='bid':
                    continue
            i=i-1
    #对数据按照tp排序
    price_df_pct_list_sorted = sorted(price_df_pct_list,key=lambda d: d['tp'])
    #判断是否需要降采样
    if plot_interval_us is not None:
        # 添加采样时间
        for item in price_df_pct_list_sorted:
            item['tp_dt_1'] = unix2dt(item['tp'], 1000)
            price_ret_1.append(item)
                    
        price_df_pct_list_sample = timeline_sample_kv(price_ret_1,'tp_dt_1',sample_gap=plot_interval_us)
        logging.info(f"采样后数据点数量: {len(price_df_pct_list_sample)}")
        price_df_pct.extend(price_df_pct_list_sample)
    else:
        price_df_pct.extend(price_df_pct_list_sorted)
    return price_df_pct
    



def get_price_df_pct_plot(price_df_pct_list):
    '''
    author：张鑫健
    作图得到价差图
    '''
    scatters = []
    scatters.append(go.Scatter(x=[unix2dt(row['tp'], 1000) for row in price_df_pct_list],
                                           y=[row['price_df_pct'] for row in price_df_pct_list],
                                           mode='lines',
                                           yaxis='y1',
                                           hovertext=[{'price_df_pct': row['price_df_pct'],'tp':row['tp']} for row in price_df_pct_list],
                                           name=f"price_df_pct")) 
    return scatters




def get_price_df_pct(begin_time,end_time,exchange_ask="binance",exchange_bid="binance",symbol_ask="btc_usdt",symbol_bid= "btc_usdt_uswap",bucket='dpticker'):
    scatters_price=[]
    data_ask,data_bid=get_price_df_pct_list(begin_time,end_time,exchange_ask,exchange_bid,symbol_ask,symbol_bid,bucket,is_adj_time=True)
    price_df_pct_list=get_price_df_pct_sorted(data_ask,data_bid,plot_interval_us=1000000)
    scatters_price.extend(get_price_df_pct_plot(price_df_pct_list))
    #作图
    plot_title = f'plot'
    plot_title_price = f'plot_test_price'
    mylayout_price = go.Layout(title=plot_title, yaxis=dict(title='pct'), yaxis1=dict(title='price_pct', titlefont=dict(color='rgb(148, 103, 189)'), tickfont=dict(color='rgb(148, 103, 189)'),overlaying= 'y1',side='right'))
    easy_plot(scatters_price, title=plot_title_price, layout=mylayout_price, store_path="./files_price_pct/")

if __name__ == '__main__':

     begin_time = datetime.datetime(2023,1,1,0,15,tzinfo=TZ_8)
     end_time = datetime.datetime(2023,1,1,0,20,tzinfo=TZ_8)

     get_price_df_pct(begin_time,end_time,exchange_ask="binance",exchange_bid="binance",symbol_ask="btc_usdt",symbol_bid= "btc_usdt_uswap",bucket='dpticker')

