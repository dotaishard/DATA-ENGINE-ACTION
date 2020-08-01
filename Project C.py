# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:32:08 2020

@author: zhangwenyi
"""

### PROJECT C ###
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np


#加载数据
data = pd.read_csv('CarPrice_Assignment.csv')
train = data[["car_ID","symboling","CarName", "fueltype","aspiration","doornumber","carbody","drivewheel","enginelocation","wheelbase","carlength","carwidth","carheight","curbweight","enginetype","cylindernumber","enginesize","fuelsystem","boreratio","stroke","compressionratio","horsepower","peakrpm","citympg","highwaympg","price"]]
train = train.drop(columns = ["car_ID","CarName"])
#print (train)


#LabelEncoder
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
train['symboling'] = LE.fit_transform(train['symboling'])
train['fueltype'] = LE.fit_transform(train['fueltype'])
train['aspiration'] = LE.fit_transform(train['aspiration'])
train['doornumber'] = LE.fit_transform(train['doornumber'])
train['carbody'] = LE.fit_transform(train['carbody'])
train['drivewheel'] = LE.fit_transform(train['drivewheel'])
train['enginelocation'] = LE.fit_transform(train['enginelocation'])
train['cylindernumber'] = LE.fit_transform(train['cylindernumber'])
train['enginetype'] = LE.fit_transform(train['enginetype'])
train['fuelsystem'] = LE.fit_transform(train['fuelsystem'])
#print (train)


#规范化到 [0,1] 空间
min_max_scaler = preprocessing.MinMaxScaler()
train_x = min_max_scaler.fit_transform(train)
pd.DataFrame(train_x).to_csv('temp.csv', index=False)
#print (train_x)


#使用KMeans聚类
kmeans = KMeans(n_clusters=50)
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
#合并聚类结果，插入到原数据中
result = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
result.rename({0:u'聚类结果'},axis=1,inplace=True)
#print(result)
#将结果导出到CSV文件中
result.to_csv("Project C.csv",index=False)