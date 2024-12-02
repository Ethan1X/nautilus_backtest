import os
import numpy as np
import pandas as pd

'''
folder_path可以根据label所在的文件夹的位置进行变动
'''
folder_path = os.getcwd()  # 提取当前路径

keyword = 'label'  # 指定关键词

# 获取文件夹中的所有文件名
file_names = os.listdir(folder_path)

# 提取带有关键词的文件名
label_file_names = [file_name for file_name in file_names if keyword in file_name]

# 设置保留概率(非保留的全部设置为nan)
nan_probability = 0.9
nan_value = np.nan
nan_columns = ["price_change_rate_0.5s", "price_change_rate_1s", "price_change_rate_3s"]

# 降采样
for file_name in label_file_names:
    label = pd.read_parquet(file_name)
    zero_mask = (label.drop('timestamp', axis = 1) == 0).all(axis=1)
    nan_mask = zero_mask & (np.random.rand(len(label)) < nan_probability)
    label.loc[nan_mask, nan_columns] = nan_value
    new_name = os.path.splitext(file_name)[0] + "_process" +os.path.splitext(file_name)[1] 
    label.to_parquet(new_name)
