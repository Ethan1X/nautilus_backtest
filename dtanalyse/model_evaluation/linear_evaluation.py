'''
    线性模型评价
'''

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.metrics import mean_squared_error


def evaluation(y_pred, y_true, key, class_thre=0.0001, label_name = '0.5s', model_name = 'model', config = {'rate_1' : 1e-4, 'bonds':[-10*1e-4, 10*1e-4], 'bins' :[1e-4, 5e-4], 'diff_1' : 2e-5, 'diff_2' : 1e-5}):
    '''
    rate_1 : 柱状图中的分割比例
    bins : 分段mse的分割点，为列表格式，函数中自动生成区间
    diff_1 : 真实值和预测值同号时badpred阈值
    diff_2 : 真实值和预测值异号时badpred阈值
    bonds: 用于划分直方图区间边界
    key: 为0或者1，表示以y_true or y_pred 划分结果区间, key为0，表示使用y_true进行分桶, key为1，表示使用y_pred进行分桶
    class_thre: 计算分类准确率召回率使用的阈值，三分类：涨超，跌超，未涨超跌超
    '''
    rate_1 = config['rate_1']
    bins = config['bins']
    diff_1 = config['diff_1']
    diff_2 = config['diff_2']
    left_bond = config['bonds'][0]
    right_bond = config['bonds'][1]

    # 定义区间的开始、结束和大小
    start = min(y_pred.min(), y_true.min())
    end = max(y_pred.max(), y_true.max())
    size = rate_1

    # 使用numpy.histogram计算直方图的区间和计数
    x_bins = [-np.inf]+ list(np.arange(left_bond, right_bond, size)) + [np.inf]
    counts1, edges1 = np.histogram(y_pred, bins=x_bins)
    counts2, edges2 = np.histogram(y_true, bins=x_bins)

    ## 冗余检验
    y_pred = np.array(y_pred).reshape(-1)
    y_true = np.array(y_true).reshape(-1)
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError('mismatch of length of 2 array')
    
    #初始化结果
    results = {
        'interval': [],
        'count': [],
        'model_mse': [],
        'baseline_mse' : [],
        'badpred_rate_1': [],
        'badpred_rate_2': []
    }
    
    
    # 整体检验
    # 计算MSE 
    model_mse = np.mean((y_true - y_pred)**2)
    baseline_mse = np.mean((y_true - np.mean(y_true))**2)
    print("整体结果：")
    print(f"Testing data number: {len(y_pred)}")
    print(f"Model MSE: {model_mse}")
    print(f"Basedline MSE: {baseline_mse}")
    
    #分类比较
    badpreds_1 = (
    ((y_true >= 0) & (y_pred >= 0) | (y_true <= 0) & (y_pred <= 0)) & (np.abs(y_true - y_pred) >= diff_1)    
    )
    badpreds_2 = (
    ((y_true > 0) & (y_pred < 0) | (y_true < 0) & (y_pred > 0)) & (np.abs(y_true - y_pred) >= diff_2)
    )
    
    badpred_rate_1 = np.sum(badpreds_1) / len(y_pred)
    badpred_rate_2 = np.sum(badpreds_2) / len(y_pred)
    print(f"同向Bad Prediction Rate: {badpred_rate_1:.2%} (同向差值大于{diff_1})")
    print(f"不同向Bad Prediction Rate: {badpred_rate_2:.2%} (不同向差值大于{diff_2})")
    
    
    # 给出涨跌平三分类的评价
    originPing, originZhang, originDie = 0,0,0
    originPing = (abs(y_true) <= class_thre).sum()
    originZhang = (y_true > class_thre).sum()
    originDie = (y_true < -class_thre).sum()
    print("原始y_true中上涨占比" + str(round(originZhang / (len(y_true) - 1), 4)) 
          + "，下跌占比" + str(round(originDie / (len(y_true) - 1), 4)) 
          + "，平占比" + str(round(originPing / (len(y_true) - 1), 4)))

    predictPing, predictZhang, predictDie = 0,0,0
    predictPing = (abs(y_pred) <=class_thre).sum()
    predictZhang = (y_pred > class_thre).sum()
    predictDie = (y_pred < -class_thre).sum()
    print("预测y_true中上涨占比" + str(round(predictZhang / (len(y_pred) - 1), 4)) 
          + "，下跌占比" + str(round(predictDie / (len(y_pred) - 1), 4)) 
          + "，平占比" + str(round(predictPing / (len(y_pred) - 1), 4)))
    

    rightZhang, rightDie, rightPing = 0,0,0
    rightPing = ((abs(y_true) <= class_thre) & (abs(y_pred) <= class_thre)).sum()
    rightZhang = ((y_true > class_thre) & (y_pred > class_thre)).sum()
    rightDie = ((y_true < -class_thre) & (y_pred < -class_thre)).sum()
    zRec = round(rightZhang / originZhang, 4)
    dRec = round(rightDie / originDie, 4)
    pRec = round(rightPing / originPing, 4)
    zAcc = round(rightZhang / predictZhang, 4) if predictZhang != 0 else np.nan
    dAcc = round(rightDie / predictDie, 4) if predictDie != 0 else np.nan
    pAcc = round(rightPing / predictPing, 4) if predictPing != 0 else np.nan
    print("涨召回率" + str(zRec))
    print("跌召回率" + str(dRec))
    print("平召回率" + str(pRec))
    print("涨准确率" + str(zAcc))
    print("跌准确率" + str(dAcc))
    print("平准确率" + str(pAcc))
    
    
    # 给出同正同负的两分类的评价
    originPos = (y_true >= 0).sum()
    originNeg = (y_true < 0).sum()
    predictPos = (y_pred >= 0).sum()
    predictNeg = (y_pred < 0).sum()
    RightPos = ((y_true >= 0) & (y_pred >= 0)).sum()
    RightNeg = ((y_true < 0) & (y_pred < 0)).sum()

    posRec = round(RightPos / originPos, 4)
    negRec = round(RightNeg / originNeg, 4)
    posAcc = round(RightPos / predictPos, 4) if predictPos != 0 else np.nan
    negAcc = round(RightNeg / predictNeg, 4) if predictNeg != 0 else np.nan
    print("正召回率" + str(posRec))
    print("负召回率" + str(negRec))
    print("正准确率" + str(posAcc))
    print("负准确率" + str(negAcc))
    

    #写入结果
    results['interval'].append('all data')
    results['count'].append(len(y_pred))
    results['model_mse'].append(model_mse)
    results['baseline_mse'].append(baseline_mse)
    results['badpred_rate_1'].append(badpred_rate_1)
    results['badpred_rate_2'].append(badpred_rate_2)
    results['zRec'].append(zRec)
    results['dRec'].append(dRec)
    results['pRec'].append(pRec)
    results['zAcc'].append(zAcc)
    results['dAcc'].append(dAcc)
    results['pAcc'].append(pAcc)
    results['posRec'].append(posRec)
    results['negRec'].append(negRec)
    results['posAcc'].append(posAcc)
    results['negAcc'].append(negAcc)
    
    # 分段检验
    # 生成区间
    sorted_rates = sorted(bins)
    intervals = []

    # 生成负无穷到最负阈值的区间
    intervals.append((-np.inf, -sorted_rates[-1]))
    # 生成每个负阈值之间的区间
    for i in range(len(sorted_rates) - 1, 0, -1):
        intervals.append((-sorted_rates[i], -sorted_rates[i - 1]))
    # 包括零点附近的区间
    intervals.append((-sorted_rates[0], 0))
    intervals.append((0, sorted_rates[0]))
    # 生成每个正阈值之间的区间
    for i in range(len(sorted_rates) - 1):
        intervals.append((sorted_rates[i], sorted_rates[i + 1]))
    # 生成最大正阈值到正无穷的区间
    intervals.append((sorted_rates[-1], np.inf))
    intervals = [(edges1[i], edges1[i+1]) for i in range(len(edges1) - 1)]

    # 初始化结果列表

    for lower, upper in intervals:
        # 确定当前区间内的索引
        if key == 0:
            mask = (y_true >= lower) & (y_true < upper)
        elif key == 1:
            mask = (y_pred >= lower) & (y_pred < upper)
        y_true_interval = y_true[mask]
        y_pred_interval = y_pred[mask]
        
        # 计算 MSE
        model_mse = np.mean((y_pred_interval - y_true_interval) ** 2) if len(y_pred_interval) > 0 else np.nan
        baseline_mse = np.mean((y_true_interval - np.mean(y_true))**2) if len(y_pred_interval) > 0 else np.nan
        # 计算 badpred_rate
        badpreds_1 = (
        ((y_true_interval >= 0) & (y_pred_interval >= 0) | (y_true_interval <= 0) & (y_pred_interval <= 0)) & (np.abs(y_true_interval - y_pred_interval) >= diff_1)    )
        badpreds_2 = (
        ((y_true_interval > 0) & (y_pred_interval < 0) | (y_true_interval < 0) & (y_pred_interval > 0)) & (np.abs(y_true_interval - y_pred_interval) >= diff_2)
        )
        
        badpred_rate_1 = np.sum(badpreds_1) / len(y_pred_interval)
        badpred_rate_2 = np.sum(badpreds_2) / len(y_pred_interval)
        
        #打印结果
        print(f"区间{lower} to {upper}结果：")
        print(f"Testing data number: {len(y_pred_interval)}")
        print(f"Model MSE: {model_mse}")
        print(f"Basedline MSE: {baseline_mse}")
        print(f"同向Bad Prediction Rate: {badpred_rate_1:.2%} (同向差值大于{diff_1})")
        print(f"不同向Bad Prediction Rate: {badpred_rate_2:.2%} (不同向差值大于{diff_2})")
        
        # 给出涨跌平三分类的评价
        originPing, originZhang, originDie = 0,0,0
        originPing = (abs(y_true) <= class_thre).sum()
        originZhang = (y_true > class_thre).sum()
        originDie = (y_true < -class_thre).sum()
        print("原始y_true中上涨占比" + str(round(originZhang / (len(y_true) - 1), 4)) 
              + "，下跌占比" + str(round(originDie / (len(y_true) - 1), 4)) 
              + "，平占比" + str(round(originPing / (len(y_true) - 1), 4)))

        predictPing, predictZhang, predictDie = 0,0,0
        predictPing = (abs(y_pred) <= class_thre).sum()
        predictZhang = (y_pred > class_thre).sum()
        predictDie = (y_pred < -class_thre).sum()
        print("预测y_true中上涨占比" + str(round(predictZhang / (len(y_pred) - 1), 4)) 
              + "，下跌占比" + str(round(predictDie / (len(y_pred) - 1), 4)) 
              + "，平占比" + str(round(predictPing / (len(y_pred) - 1), 4)))


        rightZhang, rightDie, rightPing = 0,0,0
        rightPing = ((abs(y_true) <= class_thre) & (abs(y_pred) <= class_thre)).sum()
        rightZhang = ((y_true > class_thre) & (y_pred > class_thre)).sum()
        rightDie = ((y_true < -class_thre) & (y_pred < -class_thre)).sum()
        zRec = round(rightZhang / originZhang, 4)
        dRec = round(rightDie / originDie, 4)
        pRec = round(rightPing / originPing, 4)
        zAcc = round(rightZhang / predictZhang, 4) if predictZhang != 0 else np.nan
        dAcc = round(rightDie / predictDie, 4) if predictDie != 0 else np.nan
        pAcc = round(rightPing / predictPing, 4) if predictPing != 0 else np.nan
        print("涨召回率" + str(zRec))
        print("跌召回率" + str(dRec))
        print("平召回率" + str(pRec))
        print("涨准确率" + str(zAcc))
        print("跌准确率" + str(dAcc))
        print("平准确率" + str(pAcc))


        # 给出同正同负的两分类的评价
        originPos = (y_true >= 0).sum()
        originNeg = (y_true < 0).sum()
        predictPos = (y_pred >= 0).sum()
        predictNeg = (y_pred < 0).sum()
        RightPos = ((y_true >= 0) & (y_pred >= 0)).sum()
        RightNeg = ((y_true < 0) & (y_pred < 0)).sum()

        posRec = round(RightPos / originPos, 4)
        negRec = round(RightNeg / originNeg, 4)
        posAcc = round(RightPos / predictPos, 4) if predictPos != 0 else np.nan
        negAcc = round(RightNeg / predictNeg, 4) if predictNeg != 0 else np.nan
        print("正召回率" + str(posRec))
        print("负召回率" + str(negRec))
        print("正准确率" + str(posAcc))
        print("负准确率" + str(negAcc))


        # 将结果添加到列表中
        results['interval'].append(f'{lower:.6} to {upper:.6}')
        results['count'].append(len(y_pred_interval))
        results['model_mse'].append(model_mse)
        results['baseline_mse'].append(baseline_mse)
        results['badpred_rate_1'].append(badpred_rate_1)
        results['badpred_rate_2'].append(badpred_rate_2)
        results['zRec'].append(zRec)
        results['dRec'].append(dRec)
        results['pRec'].append(pRec)
        results['zAcc'].append(zAcc)
        results['dAcc'].append(dAcc)
        results['pAcc'].append(pAcc)
        results['posRec'].append(posRec)
        results['negRec'].append(negRec)
        results['posAcc'].append(posAcc)
        results['negAcc'].append(negAcc)

    pd.DataFrame(results).to_csv(f'eval_{model_name}_{label_name}_{key}_results.csv', index=False)



    data1 = y_pred
    data2 = y_true

    # 创建图表
    fig = go.Figure()

    tmp_x = edges1[:-1]
    tmp_x[0] = tmp_x[1]-size

    # 添加第一个直方图
    fig.add_trace(go.Bar(x=tmp_x, 
                        y=counts1, width=np.diff(edges1), 
                        name='y-predict', opacity=0.7, 
                        text=[str(i) for i in intervals]))

    # 添加第二个直方图
    fig.add_trace(go.Bar(x=tmp_x, 
                        y=counts2, width=np.diff(edges2), 
                        name='y-true', opacity=0.7,
                        text=[str(i) for i in intervals]))

    def cal_mean_squared_error(y_true, y_pred, baseline):
        if len(y_true) == len(y_pred):
            if len(y_pred) > 0:
                return mean_squared_error(y_true, y_pred), mean_squared_error(y_true, np.repeat(baseline, len(y_true)))
        return 0, 0

    # 分组计算MSE
    values_groupby_y_true = {
        edges1[i]: 
        cal_mean_squared_error(
            y_true[(y_true >= edges1[i]) & (y_true < edges1[i+1])],
            y_pred[(y_true >= edges1[i]) & (y_true < edges1[i+1])],
            y_true.mean())
        for i in range(len(edges1) - 1)
    }

    # 提取MSE数据
    x_values = list(values_groupby_y_true.keys())
    mse_values = [v[0] for v in values_groupby_y_true.values()]  # MSE y_pred vs y_true
    baseline_mse_values = [v[1] for v in values_groupby_y_true.values()]  # MSE y_true vs baseline

    # 添加MSE折线图，使用新的y轴
    fig.add_trace(go.Scatter(x=x_values, y=mse_values, mode='lines+markers', name='MSE y_pred vs y_true',
                            yaxis='y2'))
    fig.add_trace(go.Scatter(x=x_values, y=baseline_mse_values, mode='lines+markers', name='MSE y_true vs baseline',
                            yaxis='y2'))

    # 更新布局
    fig.update_layout(
        title=f"Comparison of Two Histograms with MSE Lines ({model_name})",
        xaxis_title="Value",
        yaxis_title="Count",
        yaxis2=dict(title='MSE', overlaying='y', side='right'),  # 新的y轴设置
        bargap=0.2,
        barmode='overlay'
    )
    fig.write_html(f"plot_output_{label_name}_{model_name}.html")

    return results

if __name__ == '__main__':
    # 示例使用
    key = input("输入需要以作为分数据入桶的标准，0表示y_true，1表示y_pred")
    if not key in [0,1]:
        raise ValueError('key不合法')
    index = input("输入需要对第index列进行评价，0开始计数")
    y_pred = np.load("y_prediction_0201_0207.npy",allow_pickle = True).reshape(-1,3)  # 随机生成预测值
    y_true = np.load("y_origin_0201_0207.npy", allow_pickle = True).reshape(-1,3)  # 随机生成实际值
    evaluation(y_pred[:,index], y_true[:,index], key)