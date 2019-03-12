import pandas as pd
import numpy as np
from math import isnan

df = pd.read_csv( 'happiness_test_abbr.csv')
df = df.drop(['survey_time','province','city','gender','birth','nationality'], axis=1)

'''获取测试数据'''
data = df.iloc[:,:].values
dataList =  []
for i in range(2968):
    dataList.append(data[i,1:])

data_test = np.array(dataList)
columnNumber = len(data_test[0])
'''以均值填充缺失的元素'''
for i in range(0, len(data_test)):
    for j in range(0, len(data_test[0])):
        if(isnan(data_test[i][j])):
            data_test[i][j] = int(round(np.mean(df.iloc[:,j]))) # 四舍五入取整
            data_test[i][j] = int(round((data_test[i][j] - np.mean(df.iloc[:, j])) / np.std(df.iloc[:, j])))

np.save("data_test.npy",data_test)