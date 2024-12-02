# -*- coding: utf-8 -*-
"""
处理旧数据
author: lifangyu
主要内容：
1. 恢复全量标识
2. 修改撤单的记录逻辑，和新数据统一
"""


def add_sig(df, has_s0, drop_dup=False):
    """
    添加全量标识
        1. 去重并选出tp重复数>=30的
        2. 每个tp判断：1.买卖前20档是否按价格排序 2. sell是否排在buy前面，选出全量所在tp
        3. 验证tp应该有6个
        4. 在原始df中，loc全量所在tp，分离全量和增量并添加标识
    params:
        df: 原始数据转为的dataframe
        has_s0: 撤单是否为s=0
        drop_dup: 是否删除全量tp里面的增量
    """
    # 去重并选出tp重复数>=30的 
    # 不要reset_index!
    temp_df = df.drop_duplicates(
        subset=['p', 's', 't', 'tp'])
    tp_counts = temp_df['tp'].value_counts()
    temp_df['count'] = temp_df['tp'].map(tp_counts)
    temp_df = temp_df[temp_df['count'] >= 30]

    # 每个tp判断：1.【index断点前的】【买卖前20档】是否按价格排序 2. sell是否排在buy前面
    all_gps = temp_df.groupby('tp')
    tp_result = []
    for tp, gp in all_gps:
        end_idx = get_last_cts(gp.index)
        incre_gp = gp.loc[:end_idx]
        buy_df = incre_gp[incre_gp['t'] == 'buy']
        sell_df = incre_gp[incre_gp['t'] == 'sell']
        if len(buy_df) == 0 or len(sell_df) == 0:
            continue
        type_sorted = (sell_df.index[0] < buy_df.index[0]) and (sell_df.index[-1] < buy_df.index[-1])
        buy_sorted = (buy_df['p'].head(20)).is_monotonic_decreasing
        sell_sorted = (sell_df['p'].head(20)).is_monotonic_increasing
        if buy_sorted and sell_sorted and type_sorted:
            tp_result.append(tp)

    # 如果tp量不够直接返回
    if len(tp_result) not in [4, 5, 6]:
        # raise Exception(f"Error: tp count is {len(tp_result)}")
        return df, len(tp_result)

    # 分离全量和增量并添加标识
    for tp in tp_result:
        # 如果撤单为s=0，那剩余的增量全部为撤单
        if has_s0:
            idx_ls = temp_df[(temp_df['tp'] == tp) & (temp_df['s'] != 0)].index
            start_idx, end_idx = idx_ls[0], idx_ls[-1]
        # 如果撤单为重复挂单信息，那么用全量和增量中间index不连续的逻辑提取全量
        else:
            idx_ls = temp_df[temp_df['tp'] == tp].index
            start_idx = idx_ls[0]
            end_idx = get_last_cts(idx_ls)
        df.loc[start_idx: end_idx, '_'] = '_'   # 在原始df上操作

    # 删除全量tp里面的增量
    if drop_dup:
        df = df[~((df['tp'].isin(tp_result)) & df['_'].isna())]

    return df, len(tp_result)


def get_last_cts(idx_ls):
    """
    找到最后一个连续的index
    """
    not_cts = idx_ls.to_series().diff(-1) != -1
    return not_cts.idxmax() if not_cts.any() else idx_ls[-1]


def get_rows_toreset(gp):
    """
    找到每个分组（价格和方向一样）需要修改的行的index
    param: gp: 按照价格和方向分组的dataframe
    """
    s = -1
    gp_reset_idx = []
    
    for row in gp.itertuples():
        row_s = row.s
        if row_s != s:
            s = row_s
        else:
            s = 0
            gp_reset_idx.append(row.Index)
    
    return gp_reset_idx
            

def reset_s0(df):
    """
    修改撤单记录，本来的逻辑：连续第二条重复订单为撤单；修改后：s=0
    param:
        df: 原始数据转为的dataframe，且应该包含全量标识
    """
    # 如果不包含全量标识，无法区分某条是撤单还是全量
    # if '_' not in df.columns:
    #     raise Exception("Only process df with _ signal")

    # 按照价格和方向分组处理，得到所有需要修改的行的index
    temp_df = df[df['_'].isna()].sort_values(by='tp', ascending=True)
    pt_gps = temp_df.groupby(['p', 't'], group_keys=False)
    reset_idx_list = []
    for _, gp in pt_gps:
        gp_reset_idx = get_rows_toreset(gp)
        reset_idx_list.extend(gp_reset_idx)
    df.loc[reset_idx_list, 's'] = 0

    return df



# def get_rows_toreset(gp):
#     """
#     找到每个分组（价格和方向一样）需要修改的行的index
#         1. 先找到和上条数据重复的行（只有tp不同）
#         2. 重复的行中，和上面不重复的行中间间隔奇数行才是撤单行
#             如果一条数据接着重复4次，只有【2&4】才是撤单，【1&3】是挂单
#     param:
#         gp: 按照价格和方向分组的dataframe
#     """
#     temp = gp.reset_index().rename(columns={'index': 'origin_idx'})  # 重置index，origin_idx是gp在原始df中的index
#     temp['same_as_last'] = temp['s'] == temp['s'].shift(1)   # 找到和上条数据重复的行（只有tp不同）

#     # 找到前面【最近的】【没有重复的】行的index
#     cum_diff = (temp['same_as_last'] != temp['same_as_last'].shift()).cumsum()
#     temp['pre_idx'] = temp.groupby(cum_diff)['same_as_last'].transform('idxmax')-1

#     temp['is_odd'] = (temp.index - temp['pre_idx']) % 2 == 1  # 找到中间间隔奇数行的行
#     temp['rows_toreset'] = (temp['same_as_last']) & (temp['is_odd'])   # 找到需要修改的行

#     return temp[temp['rows_toreset']]['origin_idx'].to_list()





    
    
    
    
