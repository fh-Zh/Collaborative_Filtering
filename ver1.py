# -*- coding=utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
import csv
from tkinter import messagebox

# read file
data = pd.read_csv('./ratings.csv', names=['user_id', 'item_id', 'rating', 'timestamp'])
data = data[1:]
m = []
n = []
for i in range(1, len(data.user_id)):
    m.append(int(data.user_id[i]))
    n.append(int(data.item_id[i]))
# the number of users
mm = max(m)
# 建立字典,key为序列,value为电影的id,减小计算量
d = {}
nn = 0
for j in range(0, len(n)):
    if n[j] in d.values():
        nn += 0
    else:  # 字典中不存在值,则添加新值
        d[nn] = n[j]
        nn += 1


# 返回字典value的key
def get_keys(dic, value):
    temp = [k for k, v in dic.items() if v == value]
    return temp[0]


# train dataset 0.9 ; test dataset 0.1 ; devided by user_id
#train_data, test_data = train_test_split(data,           test_size=0.1,stratify=data['user_id'])
test_data = pd.DataFrame(columns = ['user_id', 'item_id', 'rating', 'timestamp'])
for line in data.itertuples():
    if int(line[1]) >= 100 & int(line[1]) <= 125:
        #print(line.index)
        test_data.loc[line[0]] = [line[1], line[2], line[3], line[4]]
    elif int(line[1])>125:
        break;
train_data = data[~ data.index.isin(test_data.index)]
# users - items matrix
user_item_matrix = np.zeros((mm, nn))
for line in train_data.itertuples():
    user_item_matrix[int(line[1]) - 1, get_keys(d, int(line[2]))] = float(line[3])
# finding similar users by similarity measure
# user_similarity_m = pairwise_distances(user_item_matrix, metric='cosine') # cosine similarity measure
# user_similarity_m = pairwise_distances(user_item_matrix, metric='jaccard')
user_similarity_m = np.corrcoef(user_item_matrix)  # Return Pearson product-moment correlation coefficients
# 求每一行的平均, 就是每个user的rating的平均评分
mean_user_rating = user_item_matrix.mean(axis=1)
rating_diff = (user_item_matrix - mean_user_rating[:, np.newaxis])  # 增加新维度，便于实现加减操作
user_precdiction = mean_user_rating[:, np.newaxis] + user_similarity_m.dot(rating_diff) / \
                   np.array([np.abs(user_similarity_m).sum(axis=1)]).T
# 除以np.array([np.abs(item_similarity_m).sum(axis=1)]以对评分进行标准化
# 展平以计算 训练集 的均方根误差
prediction_flatten = user_precdiction[user_item_matrix.nonzero()]
user_item_matrix_flatten = user_item_matrix[user_item_matrix.nonzero()]
error_train = sqrt(mean_squared_error(prediction_flatten, user_item_matrix_flatten))
print('training set RMSE：', error_train)
# 测试集部分
test_data_matrix = np.zeros((mm, nn))
for line in test_data.itertuples():
    test_data_matrix[int(line[1]) - 1, get_keys(d, int(line[2]))] = float(line[3])
# 预测公式
rating_diff = (test_data_matrix - mean_user_rating[:, np.newaxis])
user_precdiction = mean_user_rating[:, np.newaxis] + user_similarity_m.dot(rating_diff) / np.array([np.abs(user_similarity_m).sum(axis=1)]).T
# 展平数据以计算测试集的均方根误差
prediction_flatten = user_precdiction[user_item_matrix.nonzero()]
user_item_matrix_flatten = user_item_matrix[user_item_matrix.nonzero()]
error_test = sqrt(mean_squared_error(prediction_flatten, user_item_matrix_flatten))
print('test set RMSE：', error_test)
# write prediction result in movie.csv
f = open('movie.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow(["userld", "movield"])
for i in range(100, 126):
    l = user_precdiction[i - 1][:]
    for j in range(0, len(l)):
        if user_item_matrix[i - 1][j] > 0:  # 丢弃user已经看过的电影,不推荐
            l[j] = -1000
    wr.writerow([i, d[np.argmax(l)]])

messagebox.showinfo("!!!!!!!", "mission complete")
