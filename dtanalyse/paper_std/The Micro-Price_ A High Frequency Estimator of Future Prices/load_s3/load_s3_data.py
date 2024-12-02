'''
    对DEX订单进行绘图展示: 将给定to地址或者from地址的成交信息在图中展示；绘图展示CEX/DEX depth；
    后续这个类演化为与S3相关数据交互获取及绘图相关的类
'''
import os, sys, datetime, logging
import plotly.graph_objs as go
import pandas as pd

if '../' not in sys.path:
    sys.path.append('../')

from util.s3_method import * 
from util.time_method import *
from util.plot_method import easy_plot
from util.hedge_log import initlog
from util.plot_method import timeline_sample_kv, get_plot_diff_data
from util.recover_depth import recoveryDepth
from util.statistic_method_v2 import describe_series

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class LoadS3Data:
    '''
        加载S3数据到内存并绘图基类
    '''
    # 建立缓冲区，将从S3拉取过的数据保存在缓冲中，下次拉取时先从缓冲中查看
    # key: bucket_type-exchange-symbol-date
    # bucket_type 同 BUCKET_LIST
    cache = {}

    @staticmethod
    def get_cex_ticker(begin_time, end_time, symbol, cex_market, plot_interval_us=None, is_adj_time=True):
        '''
            对数据进行采样，采样时间间隔:plot_interval_us,单位微秒
        '''
        # symbol_ = '_'.join(symbol.split('_')[:2])
        if is_adj_time:
            begin_time_new = s3_time_adj(begin_time, 'begin_time')
            end_time_new = s3_time_adj(end_time, 'end_time')
        else:
            begin_time_new, end_time_new = begin_time, end_time

        ticker_ret = []
        begin_time_ = begin_time_new
        s3_dt = dt2str(begin_time_, DATETIME_FORMAT4)

        while s3_dt <= dt2str(end_time_new, DATETIME_FORMAT4):
            depth_ret = get_data_frombucket(BUCKET_LIST[1], s3_dt, cex_market, symbol, cache=LoadS3Data.cache)
            logging.info(f"获取 {cex_market}_{symbol} cex ticker数据: {s3_dt} {len(depth_ret)}")
            # depth_ret = ret_list_trans(depth_ret)
            if len(depth_ret) == 0:
                logging.warning(f"cex:symbol{cex_market}:{symbol} {begin_time_} 所在hour无depth数据")
                begin_time_ += datetime.timedelta(hours=1)
                s3_dt = dt2str(begin_time_, DATETIME_FORMAT4)

                continue
                
            # 对数据进行降采样
            if plot_interval_us is not None:
                # 添加采样时间
                depth_ret_ = []
                for item in depth_ret:
                    item['tp_dt'] = unix2dt(item['tp'], 1000)
                    depth_ret_.append(item)

                ticker_ret_sample = timeline_sample_kv(depth_ret_, 'tp_dt', sample_gap=plot_interval_us)
                logging.info(f"采样后数据点数量: {len(ticker_ret_sample)}")
                ticker_ret.extend(ticker_ret_sample)
            else:
                ticker_ret.extend(depth_ret)

            begin_time_ += datetime.timedelta(hours=1)
            s3_dt = dt2str(begin_time_, DATETIME_FORMAT4)
   
        return query_dict_list(ticker_ret, 'tp', gte=dt2unix(begin_time, 1000), lte=dt2unix(end_time, 1000))

    @staticmethod
    def get_cex_ticker_fig(cex_ticker_data, ticker_title,is_trim_data=False, yaxis='y'):
        '''
            对ticker进行绘图
            展示ticker数据
        '''
        scatters = []
        if is_trim_data:
            cex_ticker_data = get_plot_diff_data(cex_ticker_data, ['ap', 'bp'])

        scatters.append(go.Scatter(x=[unix2dt(row['tp'], 1000) for row in cex_ticker_data], 
                                   y=[float(row['ap']) for row in cex_ticker_data], 
                                   mode='lines', 
                                   hovertext=[{'price': row['ap'], 'size': row.get('aa')} for row in cex_ticker_data], 
                                   yaxis=yaxis, 
                                   name=f"{ticker_title}_ask_price")) #symbol=symbol, color=color
        scatters.append(go.Scatter(x=[unix2dt(row['tp'], 1000) for row in cex_ticker_data], 
                                   y=[float(row['bp']) for row in cex_ticker_data],
                                   mode='lines', 
                                   hovertext=[{'price': row['bp'], 'size': row.get('ba')} for row in cex_ticker_data], 
                                   yaxis=yaxis, 
                                   name=f"{ticker_title}_bid_price")) #symbol=symbol, color=color
        return scatters
    
    @staticmethod
    def get_cex_ticker_fig_size(ticker_ret_size,ticker_title,yaxis='y2'):
        '''
            获取cex相关ticker的size
        '''
        bars = []
        #symbol_ = '_'.join(symbol.split('_')[:2])
        bars.append(go.Bar(x=[unix2dt(row['tp'], 1000) for row in ticker_ret_size],
                               y=[float(row['aa'])for row in ticker_ret_size], 
                               hovertext=[{'price': row['ap'], 'amount': row.get('aa')}for row in ticker_ret_size],
                               yaxis=yaxis,
                               marker=dict(color=[ 'red','blue', 'green','yellow'],
                                           line=dict(color='red',width=1)),
                               name=f"{ticker_title}_ask_ticker_size")) 
            
        bars.append(go.Bar(x=[unix2dt(row['tp'], 1000) for row in ticker_ret_size],
                               y=[float(row['ba']) for row in ticker_ret_size],
                               hovertext=[{'price': row['bp'], 'amount': row.get('ba')}for row in ticker_ret_size], 
                               yaxis=yaxis,
                               marker=dict(color=[ 'green','blue', 'red','yellow'],
                                           line=dict(color='green',width=1)),
                               name=f"{ticker_title}_bid_ticker_size"))
        return bars

    def get_cex_ticker_plot(begin_time, end_time, symbol, cex_market='binance', plot_interval_us=None):
        symbol_ = '_'.join(symbol.split('_')[:2])
        cex_data = LoadS3Data.get_cex_ticker(begin_time, end_time, symbol_, cex_market, plot_interval_us=plot_interval_us)
        scatters = LoadS3Data.get_cex_ticker_fig(cex_data, cex_market)

        return scatters

    @staticmethod
    def get_cex_depth(begin_time, end_time, exchange, symbol, head_num=5):
        '''
            获取cex相关depth（恢复后的）
        '''
        cex_depth = []
        begin_time_ = begin_time
        prev_data = None
        s3_dt = dt2str(begin_time_, DATETIME_FORMAT4)
        
        while s3_dt <= dt2str(end_time, DATETIME_FORMAT4):
            logging.info(f"开始恢复 {exchange}_{symbol} 的depth数据")
            depth_list, prev_data = recoveryDepth(exchange, symbol, s3_dt, head_num, pre_data=prev_data)
            cex_depth.extend(depth_list)
            begin_time_ += datetime.timedelta(hours=1)
            s3_dt = dt2str(begin_time_, DATETIME_FORMAT4)
        return query_dict_list(cex_depth, 'time', gte=dt2unix(begin_time, 1000), lte=dt2unix(end_time, 1000))

    @staticmethod
    def get_cex_depth_fig(cex_depth, depth_title, depth_levels=[0,1,2,3,4], yaxis='y'):
        '''
            绘图展示cex depth图
            
        '''
        scatters = []
        for depth_level in depth_levels:
            scatters.append(go.Scatter(x=[unix2dt(row['time'], 1000) for row in cex_depth], y=[row['asks'][depth_level]['p'] for row in cex_depth], 
                                       mode='lines', 
                                       yaxis=yaxis,
                                       hovertext=[{'price': row['asks'][depth_level]['p'], 'size': row['asks'][depth_level]['s']} for row in cex_depth], 
                                       name=f"{depth_title}_ask_{depth_level}")) #symbol=symbol, color=color,
            scatters.append(go.Scatter(x=[unix2dt(row['time'], 1000) for row in cex_depth], y=[row['bids'][depth_level]['p'] for row in cex_depth], 
                                       mode='lines', 
                                       yaxis=yaxis, 
                                       hovertext=[{'price': row['bids'][depth_level]['p'], 'size': row['bids'][depth_level]['s']} for row in cex_depth], 
                                       name=f"{depth_title}_bid_{depth_level}")) #symbol=symbol, color=color,

        return scatters
    
    @staticmethod
    def get_cex_depth_fig_size(cex_depth,depth_title,depth_levels=[0,1,2,3,4],yaxis='y2'):    
        '''
         根据需要展示depth的size
        '''
        bars = []
        for depth_level in depth_levels:
                     bars.append(go.Bar(x=[unix2dt(row['time'], 1000) for row in cex_depth],
                                        y=[row['asks'][depth_level]['s'] for row in cex_depth],
                                        yaxis=yaxis, 
                                        marker=dict(color=[ 'red','blue', 'green','yellow'],
                                                    line=dict(color='red',width=1)),
                                        hovertext=[{'price': row['asks'][depth_level]['p'], 'size': row['asks'][depth_level]['s']} for row in cex_depth], 
                                        name=f"{depth_title}_ask_depth_size")) 
        for depth_level in depth_levels:         
                     bars.append(go.Bar(x=[unix2dt(row['time'], 1000) for row in cex_depth], 
                                        y=[row['bids'][depth_level]['s'] for row in cex_depth],
                                        yaxis=yaxis,
                                        marker=dict(color=[ 'green','red', 'blue','yellow'],
                                                    line=dict(color='green',width=1)), 
                                        hovertext=[{'price': row['bids'][depth_level]['p'], 'size': row['bids'][depth_level]['s']} for row in cex_depth], 
                                        name=f"{depth_title}_bid_depth_size"))    
        return bars

    @staticmethod
    def get_depth_classify_fig(begin_time, end_time, exchange, symbol, num_head=5,is_adj_time=True):
        '''
            对depth中每档数据进行划分 并绘图，一共有三类：1）前一个tp就存在的depth档位；2）本tp变化的档位；3）本tp删除的档位
        '''
        depth_old, recover_depth = LoadS3Data.get_all_depth(begin_time, end_time, exchange, symbol, num_head,is_adj_time=is_adj_time)
        depth_delete_ask,depth_delete_bid = classify_depth_delete(depth_old, recover_depth)
        depth_change_ask,depth_change_bid,depth_exist_ask,depth_exist_bid = classify_depth_change_exist(recover_depth)
        # depth_exist_ask,depth_exist_bid = classify_depth_exist(recover_depth)
        scatters = []
        scatters.append(go.Scatter(x=[unix2dt(row['time'], 1000) for row in depth_delete_ask],
                                           y=[row['asks'] for row in depth_delete_ask if 'asks' in row],
                                           mode='markers',
                                           yaxis='y1',
                                           hovertext=[{'price': row['asks']} for row in depth_delete_ask if 'asks' in row],
                                           name=f"delete_ask")) 
           
        scatters.append(go.Scatter(x=[unix2dt(row['time'], 1000) for row in depth_delete_bid], 
                                           y=[row['bids'] for row in depth_delete_bid if 'bids' in row],
                                           mode='markers',
                                           yaxis='y1',
                                           hovertext=[{'price': row['bids']} for row in depth_delete_bid if 'bids' in row],
                                           name=f"delete_bid"))
        
        scatters.append(go.Scatter(x=[unix2dt(row['time'], 1000) for row in depth_change_ask],
                                           y=[row['asks']['p'] for row in depth_change_ask if 'asks' in row],
                                           mode='markers', 
                                           yaxis='y1',
                                           hovertext=[{'price': row['asks']['p'], 'size': row['asks']['s'],'add':'add'} for row in depth_change_ask if 'asks'in row], 
                                           name=f"add_ask")) 
        
        scatters.append(go.Scatter(x=[unix2dt(row['time'], 1000) for row in depth_change_bid],
                                           y=[row['bids']['p'] for row in depth_change_bid if 'bids' in row],
                                           mode='markers', 
                                           yaxis='y1',
                                           hovertext=[{'price': row['bids']['p'], 'size': row['bids']['s'],'add':'add'} for row in depth_change_bid if 'bids'in row], 
                                           name=f"add_bid"))
      
        scatters.append(go.Scatter(x=[unix2dt(row['time'], 1000) for row in depth_exist_ask],
                                           y=[row['asks']['p'] for row in depth_exist_ask if 'asks' in row],
                                           mode='markers',
                                           yaxis='y1',
                                           hovertext=[{'price': row['asks']['p'], 'size': row['asks']['s']} for row in depth_exist_ask if 'asks' in row],
                                           name=f"exist_ask"))  
       
        scatters.append(go.Scatter(x=[unix2dt(row['time'], 1000) for row in depth_exist_bid],
                                           y=[row['bids']['p'] for row in depth_exist_bid if 'bids' in row],
                                           mode='markers',
                                           yaxis='y1',
                                           hovertext=[{'price': row['bids']['p'], 'size': row['bids']['s']} for row in depth_exist_bid if 'bids' in row],
                                           name=f"exist_bid"))    
  
           
        return scatters

    @staticmethod
    def get_cex_depth_fig_value(cex_depth, depth_title):
        '''
            将asks及bids 五档 value 求和 堆叠到一个tp展示
        '''
        bars = []
        asks = []
        bids = []
        for row in cex_depth:
            total_sum = sum([ask['s'] * ask['p'] for ask in row['asks'][:5]])#计算5档的总和
            total = sum([bid['s'] * bid['p'] for bid in row['bids'][:5]])
    
            asks.append(total_sum)
            bids.append(total)
    
        bar_asks = go.Bar(
            x=[unix2dt(row['time'], 1000) for row in cex_depth],
            y=asks,
            hovertext=[{
                'total_sum': total_sum
            } for total_sum in asks],
            yaxis='y2',
            marker=dict(
                color=[ 'red','blue', 'green','yellow'],  
                line=dict(
                    color='red',  
                    width=1 
                 )
             ),
            name=f'{depth_title}_ask_value'
         )

        bar_bids = go.Bar(
            x=[unix2dt(row['time'], 1000) for row in cex_depth],
            y=bids,
            hovertext=[{
                'total': total
            } for total in bids],
            yaxis='y2',
            marker=dict(
                color=['green', 'blue', 'red'], #设置柱的颜色
                line=dict(
                    color='green',  #设置柱边框的颜色
                    width=1  #设置宽度
                 )
             ),
            name=f'{depth_title}_bid_value'
         )
        bars.append(bar_asks)
        bars.append(bar_bids)

        return bars
    
    @staticmethod
    def get_all_depth(begin_time, end_time, exchange, symbol, num_head=5,is_adj_time=True):
        '''
            获取depth数据：同时返回S3中原始的depth 和 recover后的全量depth
            对depth中每档数据进行划分，一共有三类：1）前一个tp就存在的depth档位；2）本tp变化的档位；3）本tp删除的档位
            returns:
                depth_old: S3原始depth数据
                recover_depth: 恢复全量的depth数据
        '''
        if is_adj_time:
            begin_time_new =s3_time_adj(begin_time, 'begin_time')
            end_time_new = s3_time_adj(end_time, 'end_time')
        else:
            begin_time_new, end_time_new = begin_time, end_time
            
        depth_old= []
        recover_depth=[]
        prev_data = None
        
        recover_begin_time_ = begin_time_new
        begin_time_ = begin_time_new

        s3_dt = dt2str(begin_time_, DATETIME_FORMAT4)
        recover_s3_dt = dt2str(recover_begin_time_, DATETIME_FORMAT4)
        #恢复raw depth
        while s3_dt <= dt2str(end_time_new, DATETIME_FORMAT4):
            logging.info(f"开始寻找 {exchange}_{symbol} 的depth数据")
            depth_ret = get_data_frombucket(BUCKET_LIST[0], s3_dt, exchange, symbol, cache=LoadS3Data.cache)
            depth_old.extend(depth_ret)
            begin_time_ += datetime.timedelta(hours=1)
            s3_dt = dt2str(begin_time_, DATETIME_FORMAT4)

        #恢复recover depth  
        while recover_s3_dt <= dt2str(end_time_new, DATETIME_FORMAT4):
            logging.info(f"开始恢复 {exchange}_{symbol} 的depth数据")
            depth_list, prev_data = recoveryDepth(exchange, symbol, recover_s3_dt, num_head, pre_data=prev_data)
            recover_depth.extend(depth_list)
            recover_begin_time_+= datetime.timedelta(hours=1)
            recover_s3_dt= dt2str(recover_begin_time_, DATETIME_FORMAT4)
        #做数据切片
        depth_old=query_dict_list(depth_old, 'tp', gte=dt2unix(begin_time, 1000), lte=dt2unix(end_time, 1000))
        recover_depth=query_dict_list(recover_depth, 'time', gte=dt2unix(begin_time, 1000), lte=dt2unix(end_time, 1000))
        
        return depth_old, recover_depth
    
    @staticmethod
    def get_cex_trade(begin_time, end_time, exchange, symbol, is_adj_time=True,is_merge=True):
        '''
            获取cex相关trade,得到的trade为聚合后的数据
        '''
        if is_adj_time:
            begin_time_new = s3_time_adj(begin_time, 'begin_time')
            end_time_new = s3_time_adj(end_time, 'end_time')
        else:
            begin_time_new, end_time_new = begin_time, end_time

        trade_data = []
        begin_time_ = begin_time_new
        s3_dt = dt2str(begin_time_, DATETIME_FORMAT4)

        while s3_dt <= dt2str(end_time_new, DATETIME_FORMAT4):
            logging.info(f"开始查看cex-trade: {exchange} {symbol} {s3_dt}的数据")
            trade_data.extend(get_data_frombucket(BUCKET_LIST[2], s3_dt, exchange, symbol, cache=LoadS3Data.cache))
            begin_time_ += datetime.timedelta(hours=1)
            s3_dt = dt2str(begin_time_, DATETIME_FORMAT4)
        
        trade_ret = query_dict_list(trade_data, 'tp', gte=dt2unix(begin_time, 1000), lte=dt2unix(end_time, 1000))
        # 将实际成交数据进行聚合后再展示
        if is_merge:
            trade_ret=trade_merge(trade_ret)
        return trade_ret

    @staticmethod
    def get_cex_trade_fig(cex_trade,trade_title,yaxis='y'):
        '''
            绘图展示cex trade图
        '''
        # cex_trade=trade_merge(cex_trade)
        scatters = []
        scatters.append(go.Scatter(x=[unix2dt(row['T'], 1000) for row in cex_trade], 
                                   y=[round(row['price_min'], 5) if row['m'] == 'SELL' else round(row['price_max'], 5) for row in cex_trade], 
                                   mode='markers', 
                                   marker_color=['red' if row['m'] == 'SELL' else 'green' for row in cex_trade], 
                                   yaxis=yaxis, 
                                   marker=dict(symbol="star-open"), 
                                   name=f"{trade_title}",
                                   hovertext=[{'amt': round(row['amt_sum'], 5), 'BuyerOrderId': row['b'], 'SellerOrderId': row['a'], 'p_min': round(row['price_min'], 5), 'p_max': round(row['price_max'], 5), 'p_avg': round(row['price_avg'], 5), 't_num': row['t_num']} for row in cex_trade])) #symbol=symbol, color=color,
        return scatters
    
    @staticmethod
    def get_trade_fig_size(size_data,trade_title,yaxis='y2'):
        '''
        获取trade的size图像
        
        '''
        bars=[]
        bars.append(go.Bar(x=[unix2dt(row['T'], 1000) for row in size_data],
                           y=[float(row['q']) for row in size_data],
                           hovertext=[{'price': row['p'], 'amount':row.get('q')} for row in size_data],
                           yaxis=yaxis,
                           marker=dict(color=[ 'yellow','blue', 'green','red'],
                                       line=dict(color='yellow',width=1)), 
                                       name=f"{trade_title}_trade_size")) 
        return bars

    
    @staticmethod
    def get_time_trade_fig_value(size_data,trade_title,yaxis='y2',second=10):  
     '''
      获取trade的前10s内的value总和
     '''
     trades_sell=[]
     trades_buy=[]
         #累计循环
     start_buy=0
     start_sell=0
     for i,row in enumerate(size_data):
         j_updated=False
         current_time = unix2dt(row['T'], 1000)
         previous_time = current_time - datetime.timedelta(seconds=second)
         if row['m']=='SELL': 
             total_sell=0
             for j in range(start_sell,len(size_data)):
                 prev_row=size_data[j]
                 prev_time = unix2dt(prev_row['T'], 1000)
                 if prev_time > previous_time and prev_time <= current_time and prev_row['m']=='SELL':
                     total_sell += prev_row['q'] *prev_row['p']#根据累计时刻计算总和
                     if not j_updated:
                         start_sell=j
                         j_updated=True
                 if prev_time > current_time:
                    #  start_sell=start_sell+1
                     break
             trades_sell.append(total_sell)
             trades_buy.append(0)
         if row['m']=='BUY': 
             total_buy=0
             for j in range(start_buy,len(size_data)):
                 prev_row=size_data[j]
                 prev_time = unix2dt(prev_row['T'], 1000)
                 if prev_time > previous_time and prev_time <= current_time and prev_row['m']=='BUY':
                     total_buy+= prev_row['q'] *prev_row['p']#根据累计时刻计算总和
                     if not j_updated:
                         start_buy=j
                         j_updated=True
                 if prev_time > current_time:
                    #  start_buy=start_buy+1
                     break
             trades_buy.append(total_buy)
             trades_sell.append(0)
    
     bar_trade_sell = go.Bar(
         x=[unix2dt(row['T'], 1000) for row in size_data],
         y=trades_sell,
         hovertext=[{
             'total_sell': total_sell
         } for total_sell in trades_sell],
         yaxis=yaxis,
         marker=dict(
                 color=['red', 'blue', 'green'],  # 设置柱的颜色
                 line=dict(
                     color='red',  # 设置柱边框的颜色
                     width=1  # 设置柱边框的宽度
                 )
         ),
        name=f'{trade_title}_trade_value_sell'
     )

     bar_trade_buy = go.Bar(
         x=[unix2dt(row['T'], 1000) for row in size_data],
         y=trades_buy,
         hovertext=[{
             'total_buy': total_buy
         } for total_buy in trades_buy],
         yaxis=yaxis,
         marker=dict(
                 color=['green', 'blue', 'red'],  # 设置柱的颜色
                 line=dict(
                     color='green',  # 设置柱边框的颜色
                     width=1  # 设置柱边框的宽度
                 )
         ),
         name=f'{trade_title}_trade_value_buy'
     )
     return bar_trade_sell,bar_trade_buy
 
    @staticmethod
    def get_fundingrate(begin_time, end_time, symbol, cex_market, plot_interval_ms=None, is_adj_time=True):
        '''
            对数据进行采样，采样时间间隔:plot_interval_ms,单位毫秒
        '''
        # symbol_ = '_'.join(symbol.split('_')[:2])
        if is_adj_time:
            begin_time_new = s3_time_adj(begin_time, 'begin_time')
            end_time_new = s3_time_adj(end_time, 'end_time')
        else:
            begin_time_new, end_time_new = begin_time, end_time

        ret = []
        begin_time_ = begin_time_new
        s3_dt = dt2str(begin_time_, DATETIME_FORMAT4)

        while s3_dt <= dt2str(end_time_new, DATETIME_FORMAT4):
            depth_ret = get_data_frombucket(BUCKET_LIST[3], s3_dt, cex_market, symbol, cache=LoadS3Data.cache)
            logging.info(f"获取 {cex_market}_{symbol} cex fundingrate数据: {s3_dt} {len(depth_ret)}")
            # depth_ret = ret_list_trans(depth_ret)
            if len(depth_ret) == 0:
                logging.warning(f"cex:symbol{cex_market}:{symbol} {begin_time_} 所在hour无fundingrate数据")
                begin_time_ += datetime.timedelta(hours=1)
                s3_dt = dt2str(begin_time_, DATETIME_FORMAT4)

                continue
                
            # 对数据进行降采样
            if plot_interval_ms is not None:
                # 添加采样时间
                depth_ret_ = []
                for item in depth_ret:
                    item['tp_dt'] = unix2dt(item['tp'], 1000)
                    depth_ret_.append(item)

                ret_sample = timeline_sample_kv(depth_ret_, 'tp_dt', sample_gap=plot_interval_ms)
                logging.info(f"采样后数据点数量: {len(ret_sample)}")
            ret.extend(depth_ret)

            begin_time_ += datetime.timedelta(hours=1)
            s3_dt = dt2str(begin_time_, DATETIME_FORMAT4)
   
        return query_dict_list(ret, 'tp', gte=dt2unix(begin_time, 1000), lte=dt2unix(end_time, 1000))



if __name__ == '__main__':
    initlog(None, f"plot_demo.log", log_level=logging.INFO) # "../logs"
    
    begin_time = datetime.datetime(2023, 10, 10, tzinfo=TZ_8)
    end_time = datetime.datetime(2023, 10, 10, 1, tzinfo=TZ_8)
    exchange = "binance"
    symbol = "btc_usdt"

    scatters = []
    plot_title = f'plot_test'

    cex_trade_data = LoadS3Data.get_cex_trade(begin_time, end_time, exchange, symbol, is_adj_time=False,is_merge=True)
    scatters.extend(LoadS3Data.get_cex_trade_fig(cex_trade_data, f"{exchange}_{symbol}_trade", yaxis='y'))
    scatters.extend(LoadS3Data.get_cex_ticker_plot(begin_time, end_time, symbol, exchange, plot_interval_us=None))

    mylayout = go.Layout(title=plot_title, yaxis=dict(title='pct'), yaxis2=dict(title='quote_diff', titlefont=dict(color='rgb(148, 103, 189)'), tickfont=dict(color='rgb(148, 103, 189)'),overlaying='y', side='right'))
    easy_plot(scatters, title=plot_title, layout=mylayout)
