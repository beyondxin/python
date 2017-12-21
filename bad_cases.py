# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 21:34:20 2017

@author: User
"""

# -*- coding: utf-8 -*-

"""
Created on Thu Dec 14 14:34:29 2017

@author: User
"""
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np #科学计算
from pandas import Series,DataFrame



#data_train = pd.read_csv("C:/ProgramData/Anaconda3/ds/train.csv",\
#                         header = 0)
data_train = pd.read_csv("C:/ProgramData/Anaconda3/ds/bad_cases.csv",\
                         header = 0)
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')# 柱状图 
plt.title("survived") # 标题
plt.ylabel("total number")  

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel("total number")
plt.title("Pclass")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Agefill)
plt.ylabel("Agefill")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y') 
plt.title("CDF of Age")


plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Agefill[data_train.Pclass == 1].plot(kind='kde')   
data_train.Agefill[data_train.Pclass == 2].plot(kind='kde')
data_train.Agefill[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel("Agefill")# plots an axis lable
plt.ylabel("Density") 
plt.title("Distribution of age and Pclaa")
plt.legend(('1', '2','3'),loc='best') # sets our legend for our graph.


plt.subplot2grid((2,3),(1,2))
data_train.Embark.value_counts().plot(kind='bar')
plt.title("Embark")
plt.ylabel("Number")  
plt.show()






fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({'survived':Survived_1, 'not survived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title("survived with Pclass")
plt.xlabel("Pclass") 
plt.ylabel("number") 
plt.show()

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Gender[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Gender[data_train.Survived == 1].value_counts()
df=pd.DataFrame({'survived':Survived_1, 'not survived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title("survived with Sex")
plt.xlabel("Sex") 
plt.ylabel("number") 
plt.show()



 #然后我们再来看看各种舱级别情况下各性别的获救情况
fig=plt.figure()
fig.set(alpha=0.65) # 设置图像透明度，无所谓
plt.title("survived with Pclass and sex")

ax1=fig.add_subplot(221)
data_train.Survived[data_train.Sex == 'female']\
[data_train.Pclass != 3].value_counts().plot(kind='bar',\
label="female highclass", color='#FA2479')
ax1.set_xticklabels(["survived", "unsurvived"], rotation=0)
ax1.legend(["female/high"], loc='best')

ax2=fig.add_subplot(222, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels(["unsurvived", "survived"], rotation=0)
plt.legend(["female/low"], loc='best')

ax3=fig.add_subplot(223, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels(["unsurvived", "survived"], rotation=0)
plt.legend(["male/high"], loc='best')

ax4=fig.add_subplot(224, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels(["unsurvived", "survived"], rotation=0)
plt.legend(["male/low"], loc='best')

plt.show()

