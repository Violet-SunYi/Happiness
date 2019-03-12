#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from keras import models
from keras import layers

data_train = np.load('data_train.npy')
data_labels = np.load('data_labels.npy')

columnNumber = len(data_train[0])

import os
if(os.path.exists('my_model.h5')):
    network = models.load_model('my_model.h5')
    print('已载入网络')
else:
    print('开始架构网络')
    network = models.Sequential()
    network.add(layers.Dense(25,activation='relu',input_shape=(columnNumber,)))
    network.add(layers.Dense(18, activation='relu'))
    network.add(layers.Dense(5))

    '''优化器、损失函数、精度'''
    network.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])

network.fit(data_train, data_labels, epochs=10, batch_size=256)

'''******测试集数据******'''
data_test = np.load('data_test.npy')

'''测试'''
testPredict = network.predict(data_test)
listPredict = []
for i in range(len(testPredict)):
    listPredict.append(np.argmax(testPredict[i])+1)

'''写入csv'''
id = [i for i in range(8001,10969)]
dataframe = pd.DataFrame({ 'id': id, 'happiness': listPredict})
dataframe.to_csv("happiness_submit.csv", index=False, columns=['id','happiness'])

'''保存模型'''
#network.save('my_model.h5')

