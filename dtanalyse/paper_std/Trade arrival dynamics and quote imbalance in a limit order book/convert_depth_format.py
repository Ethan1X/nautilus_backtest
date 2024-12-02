import pandas as pd
from joblib import Parallel, delayed

def convert_depth_format1(depth_data, test_len):
    """
    Func: Convert the depth data to a pandas dataframe with each row representing a bid or ask respecitvely
    Author: Vijay Huang, 03/05/2024
    Imput Format:
        {'bids': [{'p': 42556.06, 's': 1.24069, '_': '_'},
        {'p': 42555.82, 's': 0.00062, '_': '_'}],
        'asks': [{'p': 42556.07, 's': 9.04977, '_': '_'},
        {'p': 42556.6, 's': 0.00062, '_': '_'}],
        'time': 1702746052036}
    Output:
        	            price	    size	_	type
        time				
        1702746052036	42556.06	1.24069	_	bid
        1702746052036	42555.82	0.00062	_	bid
        1702746052036	42556.07	9.04977	_	ask
        1702746052036	42556.60	0.00062	_	ask
    """
    if test_len is None:
        test_len = len(depth_data)
    else:
        if len(depth_data) < test_len:
                test_len = len(depth_data)
        else:
            test_len = test_len

    depth_df = pd.DataFrame()
    for i in range(test_len):
        time = depth_data[i]["time"]
        bid_df, ask_df = pd.DataFrame(), pd.DataFrame()
        for bid in depth_data[i]['bids']:
            df = pd.DataFrame(bid, index=[time]).assign(type="bid")
            bid_df = pd.concat([bid_df, df])
        for ask in depth_data[i]['asks']:
            df = pd.DataFrame(ask, index=[time]).assign(type="ask")
            ask_df = pd.concat([ask_df, df])
        depth_df = pd.concat([depth_df, bid_df, ask_df])
        
    depth_df = pd.DataFrame(depth_df)
    depth_df.rename(columns={'p': 'price', 's': 'size'}, inplace=True)
    depth_df.index.name = "time"
    
    return depth_df


def convert_depth_format2(depth_data, test_len):
    """
    Func: Convert the depth data to a pandas dataframe with each row representing bids and asks altogether
    Author: Vijay Huang, 03/05/2024
    Imput Format:
        {'bids': [{'p': 42556.06, 's': 1.24069, '_': '_'},
        {'p': 42555.82, 's': 0.00062, '_': '_'}],
        'asks': [{'p': 42556.07, 's': 9.04977, '_': '_'},
        {'p': 42556.6, 's': 0.00062, '_': '_'}],
        'time': 1702746052036}
    Output:
	                bid1	    bid2		bid_qty1	bid_qty2	ask1	    ask2	    ask_qty1    ask_qty2	
    TimeStamp																				
    1702746052036	42556.06	42556.00	1.24069	    0.22604	    42556.07	42556.34	9.04977	    0.00062	
    """
    if test_len is None:
        test_len = len(depth_data)
    else:
        if len(depth_data) < test_len:
                test_len = len(depth_data)
        else:
            test_len = test_len
        

    depth_df = pd.DataFrame(columns=["bid1", "bid2", "bid3", "bid4", "bid5", 
                                    "bid_qty1", "bid_qty2", "bid_qty3", "bid_qty4", "bid_qty5"] 
                                    +  ["ask1", "ask2", "ask3", "ask4", "ask5",
                                        "ask_qty1", "ask_qty2", "ask_qty3", "ask_qty4", "ask_qty5"])
    for i in range(test_len):
        time = depth_data[i]["time"]
        bid_df = pd.DataFrame([d['p'] for d in depth_data[i]['bids']] + [d['s'] for d in depth_data[i]['bids']]).T
        bid_df.columns = depth_df.columns[:10]

        ask_df = pd.DataFrame([d['p'] for d in depth_data[i]['asks']] + [d['s'] for d in depth_data[i]['asks']]).T
        ask_df.columns = depth_df.columns[10:]
        df = pd.concat([bid_df, ask_df], axis=1)   
        df.index = [time]    
        depth_df = pd.concat([depth_df, df], axis=0)  
                
    depth_df.index.name = "TimeStamp"
    
    return depth_df
