import os, sys, datetime, logging
import plotly.graph_objs as go
import pandas as pd
import threading
import queue

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from util.s3_method import *
from util.time_method import *
from util.plot_method import easy_plot
from util.hedge_log import initlog
from util.plot_method import timeline_sample_kv, get_plot_diff_data
from util.recover_depth import recoveryDepth
from util.statistic_method_v2 import describe_series
from util.load_s3_data import *
from dateutil.relativedelta import relativedelta

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class FutureData:
    '''
        加载期货相关数据并绘图
    '''

    @staticmethod
    def thread_get(function: callable, begin_time: datetime.datetime, end_time: datetime.datetime,
                   symbol: str, cex_market: str, batch=10, plot_interval_us=1000,
                   is_adj_time=True):
        '''
        并行取数, 加快取数进程，并行数默认为10，短于10个小时的数据量不会打开并行，只支持取用tick和rate数据
        '''
        result_queue = queue.Queue()
        time_dif = (end_time - begin_time).total_seconds() / 3600 # 按照小时划分任务
        time_batch = time_dif // batch
        if time_batch >= 1: # 只有大于10个小时的数据量才会打开并行
            time_list = [(begin_time + relativedelta(hours=i * time_batch),
                          begin_time + relativedelta(hours=(i + 1) * time_batch))
                         for i in range(10)]
            if time_dif - time_batch * batch != 0:
                time_list.append((begin_time + relativedelta(hours=10 * time_batch), end_time))
            threads = []
            for i, j in enumerate(time_list):
                thread = threading.Thread(target=function, args=(
                j[0], j[1], symbol, cex_market, plot_interval_us, is_adj_time, result_queue))
                threads.append(thread)
                thread.start()
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            result_list = []
            while not result_queue.empty():
                result_list.extend(result_queue.get())
        else:
            result_list = function(begin_time, end_time, symbol, cex_market, plot_interval_us,
                                                    is_adj_time)
        return result_list

    @staticmethod
    def get_cex_ticker(begin_time, end_time, symbol, cex_market, plot_interval_us=None, is_adj_time=True,
                       result_queue=None):
        '''
        并行取数的相关函数，通过result_queue在不同任务间传递数据
        '''
        result_list = LoadS3Data.get_cex_ticker(begin_time, end_time, symbol, cex_market, plot_interval_us, is_adj_time)
        if result_queue != None:
            result_queue.put(result_list)
        return result_list

    @staticmethod
    def get_fundingrate(begin_time, end_time, symbol, cex_market, plot_interval_us=None, is_adj_time=True,
                     result_queue=None):
        '''
        获取资金费率
        '''
        result_list = LoadS3Data.get_fundingrate(begin_time, end_time, symbol, cex_market, plot_interval_us, is_adj_time)
        if result_queue != None:
            result_queue.put(result_list)
        return result_list
    
    
    @staticmethod
    def get_cex_trade(begin_time, end_time, symbol, exchange, plot_interval_us=None, is_adj_time=True,
                     result_queue=None):
        '''
        获取trade数据
        '''
        result_list = LoadS3Data.get_cex_trade(begin_time, end_time, exchange, symbol, is_adj_time)
        if result_queue != None:
            result_queue.put(result_list)
        return result_list
    
    
    @staticmethod
    def get_cex_depth(begin_time, end_time, symbol, cex_market, plot_interval_us=None, is_adj_time=True, result_queue=None):
        '''
        获取recover_depth数据
        '''
        result_list = LoadS3Data.get_cex_depth(egin_time, end_time, exchange, symbol)
        if result_queue != None:
            result_queue.put(result_list)
        return result_list

    
    @staticmethod
    def sort_date(date_list: list, append_date: datetime.datetime, reverse_signal: True or False):
        '''
        工具函数：用于帮助取所有期货数据时，日期排序
        '''
        date_list.append(append_date)
        sorted_dates = sorted(date_list, reverse = reverse_signal)
        date_index = sorted_dates.index(append_date) + 1
        return sorted_dates[:date_index]

    
    @staticmethod
    def get_lifetime_future(future_end: str, start_date: datetime.datetime, end_date:datetime.datetime,
                            symbol: str, exchange: str, plot_interval_us=None, is_adj_time=True,
                            has_week_future=False):
        '''
        获取从开始时间到截止时间特定期货的全部ticker数据
        :param future_end: 期货的到期时间, %Y%m%d形式
        :param start_date: 取数的开始时间
        :param end_date: 取数的截止时间
        :return: ticker数据的字典形式
        '''
        f_symbol = symbol.split('future')[0] + 'future'
        future_end = datetime.datetime.strptime(future_end, '%Y%m%d')
        future_end_date = datetime.datetime(future_end.year, future_end.month, future_end.day, future_end.hour,
                                            tzinfo=TZ_8)
        this_quarter_month = future_end_date - relativedelta(months=2)
        this_quarter_begin = datetime.datetime(this_quarter_month.year, this_quarter_month.month, 1, tzinfo=TZ_8)
        pre_quarter_month = future_end_date - relativedelta(months=5)
        pre_quarter_begin = datetime.datetime(pre_quarter_month.year, pre_quarter_month.month, 1, tzinfo=TZ_8)
        pre_quarter_end = this_quarter_begin - relativedelta(hours=1)
        if has_week_future == True:
            this_week_begin = future_end_date - relativedelta(days=7)
            this_week_end = future_end_date + relativedelta(hours=16)
            next_week_begin = future_end_date - relativedelta(days=14)
            next_week_end = this_week_begin - relativedelta(hours=1)
            this_quarter_end = next_week_begin - relativedelta(hours=1)
            start_date_list = [pre_quarter_begin, this_quarter_begin, next_week_begin, this_week_begin]
            end_date_list = [pre_quarter_end, this_quarter_end, next_week_end, this_week_end]
            total_date = 4
        else:
            this_quarter_end = future_end_date + relativedelta(hours=16)
            start_date_list = [pre_quarter_begin, this_quarter_begin]
            end_date_list = [pre_quarter_end, this_quarter_end]
            total_date = 2
        data = []
        start_date_list = FutureData.sort_date(start_date_list, start_date, True)
        end_date_list = FutureData.sort_date(end_date_list, end_date, False)
        if (len(end_date_list) >= 1) and (len(start_date_list) >= (total_date+1-1)):
            if start_date_list[total_date-1] < end_date_list[0]: 
                data.extend(FutureData.thread_get(FutureData.get_cex_ticker, start_date_list[total_date-1], end_date_list[0], f_symbol + '_next_quarter', exchange,
                                      plot_interval_us=plot_interval_us, is_adj_time=is_adj_time))
        if (len(end_date_list) >= 2) and (len(start_date_list) >= (total_date+1-2)):
            if start_date_list[total_date-2] <= end_date_list[1]: 
                data.extend(FutureData.thread_get(FutureData.get_cex_ticker, start_date_list[total_date-2], end_date_list[1], f_symbol + '_this_quarter',
                                          exchange, plot_interval_us=plot_interval_us, is_adj_time=is_adj_time))
        if (len(end_date_list) >= 3) and (len(start_date_list) >= (total_date+1-3)) and has_week_future:
            if start_date_list[total_date-3] <= end_date_list[2]: 
                data.extend(FutureData.thread_get(FutureData.get_cex_ticker, start_date_list[total_date-3], end_date_list[2], f_symbol + '_next_week',
                                          exchange, plot_interval_us=plot_interval_us, is_adj_time=is_adj_time))
        if (len(end_date_list) >= 4) and (len(start_date_list) >= (total_date+1-4)) and has_week_future:
            if start_date_list[total_date-4] <= end_date_list[3]: 
                data.extend(FutureData.thread_get(FutureData.get_cex_ticker, start_date_list[total_date-4], end_date_list[3], f_symbol + '_this_week',
                                          exchange, plot_interval_us=plot_interval_us, is_adj_time=is_adj_time))
        return data

    @staticmethod
    def merge_ticker_data(df1, df2, source_name1 = 'f_', source_name2 = 's_',merge_key = 'tp'):
        '''合并ticker数据'''
        df1 = pd.DataFrame(df1).rename(
            columns={'ap': source_name1+'ap', 'aa': source_name1+'aa', 'bp': source_name1+'bp', 'ba': source_name1+'ba'})
        df1 = df1.sort_values('tp').drop_duplicates().reset_index(drop = True)
        df2 = pd.DataFrame(df2).rename(
            columns={'ap': source_name2+'ap', 'aa': source_name2+'aa', 'bp': source_name2+'bp', 'ba': source_name2+'ba'})
        df2 = df2.sort_values('tp').drop_duplicates().reset_index(drop = True)
        data = pd.concat([df1, df2])
        data = data.sort_values(merge_key).fillna(method='ffill').reset_index(drop=True)
        return data.to_dict(orient='records')

    @staticmethod
    def get_spot_future_ticker(start_date, end_date, symbol, exchange, f_symbol, f_exchange, future_end, plot_interval_us=None, is_adj_time=True,
                            has_week_future=False):
        '''获取期货和现货的ticker数据，返回形式为字典'''
        f_data = FutureData.get_lifetime_future(future_end, start_date, end_date, f_symbol, f_exchange, plot_interval_us, is_adj_time, has_week_future)
        s_data = FutureData.thread_get(FutureData.get_cex_ticker, start_date, end_date,
                   symbol, exchange, plot_interval_us=plot_interval_us, is_adj_time=is_adj_time)
        if len(f_data)>0:
            data = FutureData.merge_ticker_data(f_data, s_data)
            future_end_ = datetime.datetime.strptime(future_end, '%Y%m%d')
            future_end_date = datetime.datetime(future_end_.year, future_end_.month, future_end_.day, 16, tzinfo=TZ_8)
            if end_date>future_end_date:
                data = pd.DataFrame(data)
                mask = data.tp>future_end_date
                data.loc[mask, ['f_ap', 'f_aa', 'f_bp', 'f_ba']] = np.nan
            data = FutureData.get_compare_pct(data, ['f_', 's_'])
            return data
        else:
            print(f'期货输入的开始时间{start_date}、截至时间{end_date}和到期日期{future_end}不匹配，未取出数据')
            return
        

    @staticmethod
    def basic_compare_plot(data, column_list, y_axis_double, column_list_axis2):
        '''
        基础作图函数
        :param data: 字典形式数据（如tick数据的标准形式）
        :param column_list: 列表，传入需要画图的数据在字典中对应的key；例：绘制最优买价和卖价则传入['ap', 'bp']
        '''
        scatters = []
        scatters_axis2=[]
        for i in column_list:
            scatters.append(go.Scatter(x=[unix2dt(row['tp'], 1000) for row in data],
                                       y=[row[i] for row in data],
                                       mode='lines',
                                       yaxis='y1',
                                       hovertext=[{i: row[i], 'tp': row['tp']} for row in
                                                  data],
                                       name=i))
        if y_axis_double:
            for j in column_list_axis2:
                scatters_axis2.append(go.Scatter(x=[unix2dt(row['tp'], 1000) for row in data],
                                                 y=[row[j] for row in data],
                                                 mode='lines',
                                                 yaxis='y2',
                                                 hovertext=[{j: row[j], 'tp': row['tp']} for row in data],
                                                 name=j))
        return scatters, scatters_axis2

    
    @staticmethod
    def get_compare_pct(data, source_name):
        '''获得两个对比标的的价格变化'''
        data = pd.DataFrame(data)
        data['pct_change'] = (data[source_name[0]+'bp'] - data[source_name[1]+'ap'])\
                                     /(data[source_name[1]+'ap'])
        data['dif'] = data[source_name[0]+'bp'] - data[source_name[1]+'ap']
        data = data.drop_duplicates(subset=[source_name[0]+'bp',source_name[1]+'ap', 'pct_change', 'dif'])
        data.loc[data['pct_change']>1,'pct_change']=1
        data.loc[data['pct_change']<-1,'pct_change']=-1
        return data.to_dict(orient='records')
    
    @staticmethod
    def get_compare_price(begin_time, end_time, exchange_ask="binance", exchange_bid="binance", symbol_ask="btc_usdt",
                         symbol_bid="btc_usdt_uswap", plot_interval_us=1000000, is_adj_time=True, source_name = []):
        '''获得两个需要对比的标的数据'''
        data_ask = FutureData.thread_get(FutureData.get_cex_ticker, begin_time, end_time, symbol_ask, exchange_ask,
                                         plot_interval_us, is_adj_time)
        data_bid = FutureData.thread_get(FutureData.get_cex_ticker, begin_time, end_time, symbol_bid, exchange_bid,
                                         plot_interval_us, is_adj_time)
        if len(source_name) == 0:
            source_name = [symbol_ask+'_', symbol_bid+'_']
        else:
            source_name = [i+'_' for i in source_name]
        data = FutureData.merge_ticker_data(data_ask, data_bid, source_name1 = source_name[0], source_name2 = source_name[1])
        data = FutureData.get_compare_pct(data, source_name)
        return data

    @staticmethod
    def get_compare_plot(data, column_list, plot_title_price = 'plot_compare_price', y_axis_double = True, column_list_axis2= ['pct_change'], plot_interval_us = None):
        '''
        绘制AB品种价格对比图和价差百分比图
        :param data:所有的数据，数据类型是dict
        :param column_list: y1轴对应数据的标签名，数据类型是列表，例如：['ask1', 'bid1']
        :param plot_title_price: 图表名字
        :param y_axis_double: 是否使用y2轴，默认为True
        :param column_list_axis2: y2轴对应数据的标签名，默认绘制价差百分比，如果需要绘制价差，则使用['dif']
        '''
        data_ = []
        if plot_interval_us is not None:
            ticker_ret_sample = timeline_sample_kv(data, 'tp_dt', sample_gap=plot_interval_us)
            logging.info(f"采样后数据点数量: {len(ticker_ret_sample)}")
            data_.extend(ticker_ret_sample)
        else:
            data_.extend(data)
        
        print(len(data_))
        
        scatters_price, scatters_price2 = FutureData.basic_compare_plot(data_, column_list, y_axis_double, column_list_axis2)

        # 作图
        plot_title = f'plot'
        mylayout_price = go.Layout(title=plot_title, yaxis=dict(overlaying='y1'),
                                   yaxis1=dict(titlefont=dict(color='rgb(148, 103, 189)'),
                                               tickfont=dict(color='rgb(148, 103, 189)'), overlaying='y2',side='right'))
        easy_plot(scatters_price + scatters_price2, title=plot_title_price, layout=mylayout_price, store_path="./fig/")


if __name__ == '__main__':
    begin_time = datetime.datetime(2024,3,10,21,0,tzinfo=TZ_8)
    end_time = datetime.datetime(2024,3,10,22,0,tzinfo=TZ_8)

    exchange_ask="binance"
    exchange_bid="binance"
    symbol_ask="bnb_usdt"
    symbol_bid="bnb_usdt_uswap"
    plot_interval_us=1000000
    # 默认获取binance中btc_usdt和btc_usdt_uswap数据，并绘制两者的价格对比图
    compare_data = FutureData.get_compare_price(begin_time, end_time, exchange_ask, exchange_bid, symbol_ask, symbol_bid, plot_interval_us)
    FutureData.get_compare_plot(compare_data, ['bnb_usdt_ap', 'bnb_usdt_uswap_ap'])

    # Okex的btc-usdt和Binance的btc-usdt
    exchange_ask, exchange_bid, symbol_ask,symbol_bid="binance", "okex", "btc_usdt","btc_usdt"
    compare_data1 = FutureData.get_compare_price(begin_time, end_time,exchange_ask, exchange_bid, symbol_ask,symbol_bid,source_name = ['binance', 'okex'])
    FutureData.get_compare_plot(compare_data1, ['binance_ap', 'okex_ap'], 'plot_compare_exchange_price', plot_interval_us = 600000)


    # 获取期现数据
    symbol, exchange, f_symbol, f_exchange, future_end, plot_interval_us, is_adj_time = 'btc_usd_cswap','binance','btc_usd_cfuture_next_quarter','binance','20240329',1000000, True
    future_compare_data = FutureData.get_spot_future_ticker(begin_time, end_time, symbol, exchange, f_symbol, f_exchange,future_end, plot_interval_us, is_adj_time)
    # 绘制期现对比图
    FutureData.get_compare_plot(future_compare_data, ['f_ap', 's_ap'], 'future_spot_compare')

