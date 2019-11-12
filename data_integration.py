#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 11:18:38 2018

@author: zhaohaibo
"""

import os
#MacOS目录地址 os.chdir("/Users/zhaohaibo/Desktop/15")
#os.chdir('C:\\Users\\zzc93\\Desktop\\BD\\Label_from_Inter\\Label_from_Inter')
import pandas as pd
import numpy as np

# Step1 读取数据
data = pd.read_csv("21_data.csv")
total = len(data)
print("sum of data:%d" % total)
des = data.describe()
fail_data = pd.read_csv("21_failureInfo.csv")
normal_data = pd.read_csv("21_normalInfo.csv")


# 对时间标签做处理，分类出label:
# label = 1: 故障时间区域
# label = 0: 正常时间区域
# label = -1:无效数据


# 转化data时间列为datetime
times = []
for i in range(len(data)):
    dt = pd.to_datetime(data.ix[i][0])
    times.append(dt)
    if(i%10000==0):
        print("complete %d / %d" % (i,len(data)))
times = pd.Series(times)
data.time = times

# 转化normal_data & fail_data时间列为datetime
def to_datetime(obj_pd): 
    Ser1 = obj_pd.iloc[:,0]
    Ser2 = obj_pd.iloc[:,1]
    for i in range(len(Ser1)):
        Ser1[i] = pd.to_datetime(Ser1[i])
        Ser2[i] = pd.to_datetime(Ser2[i])
    obj_pd.iloc[:,0] = Ser1
    obj_pd.iloc[:,1] = Ser2
    return obj_pd
        
normal_data = to_datetime(normal_data)
fail_data = to_datetime(fail_data)

# 根据datetime创建labels列表
labels = []
np.savetxt('label.csv',labels,fmt='%e',delimiter=',')
for i in range(len(times)):
    if(i%10000==0):
        print("complete %d / %d" % (i,len(times)))
    flag = 0
    for j in range(len(normal_data)):
        if((times[i] >= normal_data.startTime[j]) and (times[i] <= normal_data.endTime[j])):
            labels.append(0)
            flag = 1
            break
    for j in range(len(fail_data)):
        if(flag==1):
            break
        elif((times[i] >= fail_data.startTime[j]) and (times[i] <= fail_data.endTime[j])):
            labels.append(1)
            flag = 1
            break
    if(flag == 1):
        continue
    labels.append(-1)
np.savetxt('label.csv',labels,fmt='%e',delimiter=',')
print("complete all")