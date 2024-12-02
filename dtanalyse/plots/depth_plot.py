

def cex_depth_fig_plot():
    '''
        get_cex_depth_fig_value 使用示例
        输出图像:ticker折线图 、trade散点图以及depthm每个档位的value

    '''
    scatters_value.extend(LoadS3Data.get_cex_trade_fig(cex_trade_data,f"{exchange}_{symbol}_trade", yaxis='y1'))
    scatters_value.extend(LoadS3Data.get_cex_ticker_plot(begin_time, end_time, symbol, exchange, plot_interval_ms=None))
    bars_value.extend(LoadS3Data.get_cex_depth_fig_value(begin_time, end_time, exchange, symbol,depth_levels=[0,1,2,3,4],yaxis='y2'))
    bars_value.extend(scatters_value)
    mylayout_depth_value = go.Layout(title=plot_title, yaxis=dict(title='pct'), yaxis2=dict(title='value', titlefont=dict(color='rgb(148, 103, 189)'), tickfont=dict(color='rgb(148, 103, 189)'),overlaying= 'y1',side='right'),
                                     barmode='stack')
    easy_plot(bars_depth_value, title=plot_title_depth_value_time, layout=mylayout_depth_value, store_path="./files_bars_depth_value/")


    return



if __name__ == '__main__':
    cex_depth_fig_plot()


# def depth_summery_plot():
#     '''
        
#     '''
#     return

    
#    initlog(None, f"plot_demo.log", log_level=logging.INFO) 
    
#     begin_time = datetime.datetime(2023, 6,12,11,15,tzinfo=TZ_8)
#     end_time = datetime.datetime(2023, 6,12,11,20,tzinfo=TZ_8)
#     exchange = "binance"
#     symbol = "btc_usdt"

#     scatters_trade = []
#     scatters_ticker = []
#     scatters_depth = []
#     scatters_value=[]
#     scatters_clssify=[]
#     scatters_depth_value=[]
#     scatters_trade_value=[]
#     bars_value=[]
#     bars_trade=[]
#     bars_ticker=[]
#     bars_depth=[]
#     bars_depth_value=[]
#     bars_trade_value=[]
#     depth_delete=[]
#     depth_now=[]
#     depth_exist=[]
    
#     plot_title_trade = f'plot_test_trade'
#     plot_title_ticker = f'plot_test_ticker'
#     plot_title_depth = f'plot_test_depth'
#     plot_title_value= f'plot_test_value'
#     plot_title = f'plot'
#     plot_title_depth_value_time = f'plot_test_depth_time_value'
#     plot_title_trade_value_time = f'plot_test_trade_time_value'

#     cex_trade_data = LoadS3Data.get_cex_trade(begin_time, end_time, exchange, symbol, is_adj_time=False)
#     #cex_depth_data = LoadS3Data.get_cex_depth(begin_time, end_time, exchange, symbol)
#     #scatters_depth.extend(LoadS3Data.get_cex_depth_fig(cex_depth_data,f"{exchange}_{symbol}_depth"))
    
    
#     #获取trade的size
#     scatters_trade.extend(LoadS3Data.get_cex_trade_fig(cex_trade_data,f"{exchange}_{symbol}_trade", yaxis='y1'))
#     scatters_trade.extend(LoadS3Data.get_cex_ticker_plot(begin_time, end_time, symbol, exchange, plot_interval_ms=None))
#     bars_trade.extend(LoadS3Data.get_cex_trade_fig_size(begin_time, end_time, exchange, symbol,is_size_show=True, yaxis='y2',is_adj_time=True))
#     bars_trade.extend(scatters_trade)
    
#     #获取ticker的size
#     scatters_ticker.extend(LoadS3Data.get_cex_trade_fig(cex_trade_data,f"{exchange}_{symbol}_trade", yaxis='y1'))
#     scatters_ticker.extend(LoadS3Data.get_cex_ticker_plot(begin_time, end_time, symbol, exchange, plot_interval_ms=None))
#     bars_ticker.extend(LoadS3Data.get_cex_ticker_fig_size(begin_time, end_time, exchange, symbol,is_size_show=True,yaxis='y2'))
#     bars_ticker.extend(scatters_ticker)
    
#     #获取depth的size
#     scatters_depth.extend(LoadS3Data.get_cex_trade_fig(cex_trade_data,f"{exchange}_{symbol}_trade", yaxis='y1'))
#     scatters_depth.extend(LoadS3Data.get_cex_ticker_plot(begin_time, end_time, symbol, exchange, plot_interval_ms=None))
#     bars_depth.extend(LoadS3Data.get_cex_depth_fig_size(begin_time, end_time, exchange, symbol,is_size_show=True,depth_levels=[0,1,2,3,4],yaxis='y2'))
#     bars_depth.extend(scatters_depth)
    
#     #获取depth的value
#     scatters_value.extend(LoadS3Data.get_cex_trade_fig(cex_trade_data,f"{exchange}_{symbol}_trade", yaxis='y1'))
#     scatters_value.extend(LoadS3Data.get_cex_ticker_plot(begin_time, end_time, symbol, exchange, plot_interval_ms=None))
#     bars_value.extend(LoadS3Data.get_cex_depth_fig_value(begin_time, end_time, exchange, symbol,depth_levels=[0,1,2,3,4],yaxis='y2'))
#     bars_value.extend(scatters_value)
    
#     #获取trade的value（10s）
#     scatters_trade_value.extend(LoadS3Data.get_cex_trade_fig(cex_trade_data,f"{exchange}_{symbol}_trade", yaxis='y1'))
#     scatters_trade_value.extend(LoadS3Data.get_cex_ticker_plot(begin_time, end_time, symbol, exchange, plot_interval_ms=None))
#     bars_trade_value.extend(LoadS3Data.get_time_trade_fig_value(begin_time, end_time, exchange,symbol,yaxis='y2',second=10,is_adj_time=True))
#     bars_trade_value.extend(scatters_trade_value)
    
#     #获取depth的value（10s）
#     scatters_depth_value.extend(LoadS3Data.get_cex_trade_fig(cex_trade_data,f"{exchange}_{symbol}_trade", yaxis='y1'))
#     scatters_depth_value.extend(LoadS3Data.get_cex_ticker_plot(begin_time, end_time, symbol, exchange, plot_interval_ms=None))
#     bars_depth_value.extend(LoadS3Data.get_time_depth_fig_value(begin_time, end_time, exchange, symbol,depth_levels=[0,1,2,3,4],yaxis='y2'))
#     bars_depth_value.extend(scatters_depth_value)
    
#     #获取盘口删除、新挂单、一直存在的
#     depth_delete_ask,depth_delete_bid,depth_now_ask,depth_now_bid,depth_exist_ask,depth_exist_bid=LoadS3Data.get_old_depth(begin_time, end_time, exchange, symbol)
#     #scatters_clssify.extend(LoadS3Data.get_cex_trade_fig(cex_trade_data,f"{exchange}_{symbol}_trade", yaxis='y1'))
#     #scatters_clssify.extend(LoadS3Data.get_cex_ticker_plot(begin_time, end_time, symbol, exchange, plot_interval_ms=None))
#     scatters_clssify.extend(LoadS3Data.get_depth_fig(depth_delete_ask,depth_delete_bid,depth_now_ask,depth_now_bid,depth_exist_ask,depth_exist_bid))
    
#     #定义mylayout
#     mylayout_trade = go.Layout(title=plot_title, yaxis=dict(title='pct'), yaxis2=dict(title='size', titlefont=dict(color='rgb(148, 103, 189)'), tickfont=dict(color='rgb(148, 103, 189)'),overlaying= 'y1',side='right'))
#     mylayout_ticker= go.Layout(title=plot_title,yaxis=dict(title='pct'), yaxis2=dict(title='size', titlefont=dict(color='rgb(148, 103, 189)'), tickfont=dict(color='rgb(148, 103, 189)'),overlaying= 'y1',side='right'),barmode='stack')
#     mylayout_depth = go.Layout(title=plot_title, yaxis=dict(title='pct'), yaxis2=dict(title='size', titlefont=dict(color='rgb(148, 103, 189)'), tickfont=dict(color='rgb(148, 103, 189)'),overlaying= 'y1',side='right'),barmode='stack')
#     mylayout_value = go.Layout(title=plot_title, yaxis=dict(title='pct'), yaxis2=dict(title='value', titlefont=dict(color='rgb(148, 103, 189)'), tickfont=dict(color='rgb(148, 103, 189)'),overlaying= 'y1',side='right'),barmode='stack')
#     mylayout_trade_value = go.Layout(title=plot_title, yaxis=dict(title='pct'), yaxis2=dict(title='value', titlefont=dict(color='rgb(148, 103, 189)'), tickfont=dict(color='rgb(148, 103, 189)'),overlaying= 'y1',side='right'),barmode='stack')
#     mylayout_depth_value = go.Layout(title=plot_title, yaxis=dict(title='pct'), yaxis2=dict(title='value', titlefont=dict(color='rgb(148, 103, 189)'), tickfont=dict(color='rgb(148, 103, 189)'),overlaying= 'y1',side='right'),barmode='stack')
#     mylayout_depth = go.Layout(title=plot_title, yaxis=dict(title='pct'), yaxis2=dict(title='size', titlefont=dict(color='rgb(148, 103, 189)'), tickfont=dict(color='rgb(148, 103, 189)'),overlaying= 'y1',side='right'))
    
#     easy_plot(bars_trade, title=plot_title_trade, layout=mylayout_trade, store_path="./files_trade/")
#     easy_plot(bars_ticker, title=plot_title_ticker, layout=mylayout_ticker, store_path="./files_ticker/")
#     easy_plot(bars_depth, title=plot_title_depth, layout=mylayout_depth, store_path="./files_depth/")
#     easy_plot(bars_value, title=plot_title_value, layout=mylayout_value, store_path="./files_bars/")
#     easy_plot(bars_trade_value, title=plot_title_trade_value_time, layout=mylayout_trade_value, store_path="./files_bars_trade_value/")
#     easy_plot(bars_depth_value, title=plot_title_depth_value_time, layout=mylayout_depth_value, store_path="./files_bars_depth_value/")
#     easy_plot(scatters_clssify, title=plot_title_depth, layout=mylayout_depth, store_path="./files_depth_recover/")
   

# #获取多个交易所
# def get_all_exchange_data(begin_time, end_time, symbol, exchanges):
#     scatters = []

#     for exchange in exchanges:
#             cex_trade_data = LoadS3Data.get_cex_trade(begin_time, end_time, exchange, symbol, is_adj_time=False)
#             scatters.extend(LoadS3Data.get_cex_trade_fig(cex_trade_data, f"{exchange}_{symbol}_trade", yaxis='y1'))
#             scatters.extend(LoadS3Data.get_cex_trade_fig_size(begin_time, end_time, exchange, symbol,is_size_show=True, is_adj_time=True,yaxis='y2'))
#             scatters.extend(LoadS3Data.get_cex_ticker_plot(begin_time, end_time, symbol, exchange, plot_interval_ms=None))

#     return scatters

# begin_tim = datetime.datetime(2023, 10, 10,12,15,tzinfo=TZ_8)
# end_tim = datetime.datetime(2023, 10, 10, 12,20,tzinfo=TZ_8)
# exchang=["binance", "okex"]
# sym = "btc_usdt"
# plot_title=f'plot'
# sca=get_all_exchange_data(begin_tim,end_tim,sym,exchang)
# mylayout = go.Layout(title=plot_title, yaxis=dict(title='pct'), yaxis2=dict(title='size', titlefont=dict(color='rgb(148, 103, 189)'), tickfont=dict(color='rgb(148, 103, 189)'),overlaying='y', side='right'))
# easy_plot(sca, title=plot_title, layout=mylayout,  store_path="./files_trade/")

