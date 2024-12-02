#function
import pandas as pd
from util.load_s3_data import *
import json
from convert_depth_format import convert_depth_format2
import time
import matplotlib.pyplot as plt
import numpy as np

class Process_ip_data:
    
    def process_trade_data(trade_data):
        '''
        input:  trade data from LoadS3Data.get_cex_trade
        output: data frame including 4 rows: time, trade price, trade size, trade direction (0 for buy and 1 for sell)
        '''
        df = []
        try:
            for dic in trade_data:
                direction = 1 if dic['m']=="BUY" else 0
                df.append([dic['T'],dic['price_avg'],dic['amt_sum'],direction])
        except:
            print("格式错误")
            print(dic)
        return pd.DataFrame(df, columns=['time', 'ave_price', 'amount', 'direction']).sort_values(by=['time']).reset_index(drop=True)

    def process_ticker_data(ticker_data):
        '''
        input:  trade data from LoadS3Data.get_cex_ticker
        output: data frame including 4 rows: time, trade price, trade size, trade direction (0 for buy and 1 for sell)
        '''
        df = []
        try:
            last_time = 0
            for dic in ticker_data:
                if dic['tp'] == last_time:
                    #如果同一时刻（ms）有多个挂单信息，则只保留最后一个挂单信息
                    df.pop()
                df.append([dic['tp'],dic['bp'],dic['ba'],dic['ap'],dic['aa']])
                last_time = dic['tp']
        except:
            print("格式错误")
            print(dic)
        return pd.DataFrame(df, columns=['time', 'bp', 'ba', 'ap', 'aa']).sort_values(by=['time']).reset_index(drop=True)

    def find_trade(time, start_line, trade_data):
        '''
        从起始行开始查找，找出离time最近的一次交易，返回最近一次交易数据
        '''
        for i in range(start_line,len(trade_data)):
            if trade_data.loc[i,"time"] < time:
                continue
            else:
                return i, trade_data.loc[i]
    def find_change(next_trade_time, curr_price ,ticker_start_line, ticker_data):
        '''
        从起始行开始查找，找出离time最近的一次bid price变化ticker 
        Return next change time and change flag: 0: trade, 1: price up, 2: price down
        '''
        for i in range(ticker_start_line,len(ticker_data)):
            if ticker_data.loc[i,"time"] > next_trade_time:
                return i, next_trade_time, 0
            elif ticker_data.loc[i,"bp"] == curr_price:
                continue
            elif ticker_data.loc[i,"bp"] > curr_price:
                return i, ticker_data.loc[i,"time"], 1
            elif ticker_data.loc[i,"bp"] < curr_price:    
                return i, ticker_data.loc[i,"time"], 2
        return i, ticker_data.loc[i,"time"], 0  #如果数据结尾没有交易也没有价格变化，视为交易

