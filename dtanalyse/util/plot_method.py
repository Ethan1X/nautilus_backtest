#encoding=utf-8

import os
import plotly
import datetime
import logging
import numpy
import plotly.graph_objs as go

# 绘图的一些配置
# 点图中点的形状
SYMBOLS = ['circle-open', 'circle', 'circle-open-dot', 'square', 'star', 'triangle-up', 'pentagon', 'diamond']
# 点图中点的颜色
COLORS = ['blue', 'orange', 'green', 'brown', 'purple', 'pink', 'darkorange', 'yellow']

def double_yaxis_layout(title, y1_title, y2_title):
    return go.Layout(title=title, yaxis={'title': y1_title}, yaxis2={'title': y2_title, 'overlaying': 'y', 'side': 'right'})


def timeline_sample(timeline, sample_gap=None, sample_number=None, plot_sample=300):
    ''' 时间线抽样 '''
    sampled = []
    if sample_number:
        step = (timeline[-1][0] - timeline[0][0]) / (sample_number - 1)
    else:
        step = datetime.timedelta(seconds=(sample_gap if sample_gap is not None else plot_sample))
    for t, v in timeline:
        if len(sampled) == 0 or t - sampled[-1][0] >= step:
            sampled.append((t, v))
    return sampled

def timeline_sample_kv(timeline, sample_col, sample_gap=None, sample_number=None, is_unixordt='dt'):
    ''' 
        时间线抽样 
        数据格式: [{sample_col: datetime.datetime(), ...}, {sample_col: datetime.datetime(), ...},]
        sample_col: 采样字段，时间，datetime格式；
        sample_gas: 采样间隔，单位microsecond： 1seconds= 10**6microsecond
        sample_number: 一共采多少个点
        sample_gap: 采样时间间隔，单位毫秒 默认1000ms一个点
         
        只支持unix到ms级别
     
    '''
    sampled = []
    if len(timeline) == 0:
        return sampled
        
    if is_unixordt == 'dt':
        if sample_number:
            step = (timeline[-1][sample_col] - timeline[0][sample_col]) / (sample_number - 1)
        else:
            step = datetime.timedelta(microseconds=(sample_gap if sample_gap is not None else 1000))
        for item in timeline:
            if len(sampled) == 0 or item[sample_col] - sampled[-1][sample_col] >= step:
                sampled.append(item)
    elif is_unixordt == 'unix':
        if sample_number:
            step = (timeline[-1][sample_col] - timeline[0][sample_col]) / (sample_number - 1)
        else:
            step = sample_gap / 1000 if sample_gap is not None else 1000
        for item in timeline:
            if len(sampled) == 0 or item[sample_col] - sampled[-1][sample_col] >= step:
                sampled.append(item)
    else:
        assert(f'异常时间格式，不支持抽样')
    return sampled

def get_plot_diff_data(raw_data:list, comp_cols:list):
    '''
        将k,v json结构的绘图数据进行瘦身，单纯剔除与上个点相同的点，以减轻绘图压力
        raw_data: kv json 构成的list
        comp_cols: 需要对比的k名称list
    '''
    plot_data = []
    last_item = None
    for item in raw_data:

        if last_item is None or not all([item[k] == last_item[k] for k in comp_cols]):
            plot_data.append(item)
            last_item = item

    return plot_data

def plot_timeline_rate(timeline, title, **kwargs):
    assert(len(timeline) > 0)
    v0 = timeline[0][1]
    assert(v0 > 0)
    return plot_timeline([(t, v / v0) for t, v in timeline], title, **kwargs)


def plot_timeline(timeline, title, sample_gap=None, sample_number=None, **kwargs):
    ''' 用时间线抽样生成 scatter '''
    return plot_pair_list(timeline_sample(timeline, sample_gap, sample_number), title, **kwargs)


def plot_pair_list(pair_list, title, mode="lines", yaxis="y1"):
    ''' 用时间线生成 scatter '''
    return go.Scatter(x=numpy.array([t for t, _ in pair_list]), y=numpy.array([mid for t, mid in pair_list]), mode=mode, name=title, yaxis=yaxis)


def plot_dict(dict_to_plot, title):
    return plot_pair_list(sorted(dict_to_plot.items()), title)


def plot_single(pair_list, title="untitled", store_path="/data/jupyter/plots/", layout=None):
    easy_plot([plot_pair_list(pair_list, title)], title, store_path, layout)


def easy_plot(scatters, title="untitled", store_path="/data/jupyter/plots/", layout=None, port=9991, plot_path='view/plots'):
    ''' 一行代码打印一张图，如果在 44 上，可以直接从互联网查看 '''
    data = {"data": scatters}
    if layout:
        data["layout"] = layout
    plotly.offline.plot(data, filename=get_plot_filename(title, store_path, port, plot_path))

def get_plot_filename(title, store_path, port=9991, plot_path='view/plots'):
    filename = '%s.html' % title
    filehtml = 'http://deeptrading.eth:%s/%s/%s' % (port, plot_path, filename)
    print(filehtml)
    logging.info(filehtml)
    fullpath = '%s%s' % (store_path, filename)
    return fullpath

if __name__ == '__main__':
    # 画图demo
    title = 'plot_demo'
    scatters = []
    # 折线图
    x = range(100)
    y = [i * i for i in x]
    scatters.append(go.Scatter(x=x, y=y, mode='lines', name='square1', yaxis='y1'))
    scatters.append(go.Scatter(x=x, y=x, mode='lines', name='square2', yaxis='y2'))
    # 设置双坐标轴名称
    layout = double_yaxis_layout(title, 'y1_name', 'y2_name')
    easy_plot(scatters, title=title, layout=layout)

