import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

    # Convert the depth data to a pandas dataframe with each row representing a bid and/or ask, respecitvely
    
    
def convert_depth_format1(depth_data, test_len):
    """ Func: Convert the depth data to a pandas dataframe with each row representing a bid or ask
    Parameters
    ----------
    depth_data: [{'bids': [{'p': 42556.06, 's': 1.24069, '_': '_'},
                {'p': 42555.82, 's': 0.00062, '_': '_'}],
                'asks': [{'p': 42556.07, 's': 9.04977, '_': '_'},
                {'p': 42556.6, 's': 0.00062, '_': '_'}],
                'time': 1702746052036}]
    test_len: int
    
    Returns
    -------
        	            price	    size	_	type
        time				
        1702746052036	42556.06	1.24069	_	bid
        1702746052036	42555.82	0.00062	_	bid
        1702746052036	42556.07	9.04977	_	ask
        1702746052036	42556.60	0.00062	_	ask
    """
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
    """ Convert the depth data to a pandas dataframe with each row representing bids and asks together
    Parameters
    ----------
    depth_data: [{'bids': [{'p': 42556.06, 's': 1.24069, '_': '_'},
                {'p': 42555.82, 's': 0.00062, '_': '_'}],
                'asks': [{'p': 42556.07, 's': 9.04977, '_': '_'},
                {'p': 42556.6, 's': 0.00062, '_': '_'}],
                'time': 1702746052036}]
    test_len: int
    
    Returns
    -------
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

    depth_df = pd.DataFrame()
    for i in tqdm(range(len(depth_data))):
        time = depth_data[i]["time"]
        bid_df = pd.DataFrame(pd.concat([pd.DataFrame(depth_data[0]["bids"])["p"], 
                                        pd.DataFrame(depth_data[0]["bids"])["s"]], 
                                        axis=0, ignore_index=True)).T
        ask_df = pd.DataFrame(pd.concat([pd.DataFrame(depth_data[0]["asks"])["p"], 
                                        pd.DataFrame(depth_data[0]["asks"])["s"]], 
                                        axis=0, ignore_index=True)).T
        df = pd.concat([bid_df, ask_df], axis=1)   
        df.index = [time]    
        depth_df = pd.concat([depth_df, df])
    depth_df.columns=["bid1", "bid2", "bid3", "bid4", "bid5", 
                    "bid_qty1", "bid_qty2", "bid_qty3", "bid_qty4", "bid_qty5", 
                    "ask1", "ask2", "ask3", "ask4", "ask5",
                    "ask_qty1", "ask_qty2", "ask_qty3", "ask_qty4", "ask_qty5"]
    depth_df.index.name = "TimeStamp"
    
    return depth_df


def convert_depth_format3(depth):   # Higher performance !!!
    """ Convert the depth data to a pandas dataframe with each row representing bids and asks together
    Parameters
    ----------
    depth: [{'bids': [{'p': 42556.06, 's': 1.24069, '_': '_'},
                {'p': 42555.82, 's': 0.00062, '_': '_'}],
                'asks': [{'p': 42556.07, 's': 9.04977, '_': '_'},
                {'p': 42556.6, 's': 0.00062, '_': '_'}],
                'time': 1702746052036}]
    Returns
    -------
                    bid1	    bid2		bid_qty1	bid_qty2	ask1	    ask2	    ask_qty1    ask_qty2	
    TimeStamp																				
    1702746052036	42556.06	42556.00	1.24069	    0.22604	    42556.07	42556.34	9.04977	    0.00062	
    """
    def get_p_s(x):
        y = []
        for i in range(5):
            y.append(x[i]['p'])
            y.append(x[i]['s'])
        return pd.Series(y)
    depth_df = pd.DataFrame(depth)
    depth_df[['bid1', 'bid_qty1','bid2', 'bid_qty2','bid3', 'bid_qty3','bid4', 'bid_qty4','bid5', 'bid_qty5']] = depth_df['bids'].apply(get_p_s).apply(pd.Series)
    depth_df[['ask1', 'ask_qty1','ask2', 'ask_qty2','ask3', 'ask_qty3','ask4', 'ask_qty4','ask5', 'ask_qty5']] = depth_df['bids'].apply(get_p_s).apply(pd.Series)
    del depth_df['bids'], depth_df['asks']
    depth_df.set_index(pd.to_datetime(depth_df["time"], unit='ms'), inplace=True)
    return depth_df