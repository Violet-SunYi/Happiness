#-*- coding:utf-8 -*-

from math import isnan
import pandas as pd
import numpy as np
from keras import models
from keras import layers
from keras.utils import to_categorical

print('开始清洗数据')
'''******训练集数据******'''
df = pd.read_csv( 'happiness_train_abbr.csv')
df = df.drop(['survey_time','province','city','gender','birth','nationality'], axis=1)
rawLabels = df.iloc[:, 1]

'''删去幸福感为负数的值'''
count = 0
for i in range(0,8000):
    if(rawLabels[i] < 1):
        count += 1
        df = df.drop(i, axis=0)

'''获取标签'''
labels = df.iloc[:, 1]
def label(k,labelList =  []):
    try:
        for i in range(k,8000):
            labelList.append(labels[i]-1)
    except KeyError:
        label(i+1,labelList)
    return i,labelList
i,data_labels = label(0)
data_labels = np.array(data_labels)
data_labels = to_categorical(data_labels)

'''获取训练数据'''
data = df.iloc[:,:].values
def getData(k,dataList =  []):
    try:
        for i in range(k,8000-count):
            dataList.append(data[i,2:])
    except KeyError:
        label(i+1,dataList)
    return i,dataList
i,data_train = getData(0)
data_train = np.array(data_train)

columnNumber = len(data_train[0])

'''以均值填充缺失的元素'''
for i in range(0, len(data_train)):
    for j in range(0, len(data_train[0])):
        if(isnan(data_train[i][j])):
            data_train[i][j] = int(round(np.mean(df.iloc[:,j]))) # 四舍五入取整
        data_train[i][j] = int(round((data_train[i][j] - np.mean(df.iloc[:,j])) / np.std(df.iloc[:,j])))
print('清洗数据完成')
print('开始保存清洗之后的数据')

'''保存清洗后的数据'''
np.save("data_train.npy",data_train)
np.save("data_labels.npy",data_labels)
print('数据已保存')