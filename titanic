# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:14:40 2017

@author: User
"""

#导入模块
import pandas as pd
import csv
from pandas import DataFrame 
from sklearn import preprocessing
import pylab as P 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import f_regression
#读取数据

#print('df.dtypes:')
#print(df.dtypes)
#print('df.info:')
#print(df.info())
#print('df.describe:')
#print(df.describe())


#print(temp[1:])


#图表显示
#df['Age'].hist()  
#P.show()

#预处理数据：处理缺失数据 
#clean data

#增加gender列，预设值为4
#df['Gender'] = 4  
#映射SEX的首字母M F到gender

train_test = ['train','test']
for name in train_test:
    df = pd.read_csv('C:/ProgramData/Anaconda3/ds/'+ name +'.csv', header=0)
    df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )
    df['Gender'] = df['Sex'].map({'male':1,'female':0}).astype(int)
    
    median_ages = np.zeros((2,3))
    for i in range(0, 2):  
        for j in range(0, 3):  
            median_ages[i,j] = df[(df['Gender'] == i)&\
                       (df['Pclass'] == j+1)]['Age'].dropna().median()
    df['Agefill']=df['Age']
    
    for i in range(0,2):
        for j in range(0,3):
            df.loc[(df['Age'].isnull()) & (df['Gender']==i) & (df['Pclass']==j+1),\
                   'Agefill'] = median_ages[i,j]
    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    #预处理数据：处理分类变量数字化
    df['Embark'] = df['Embarked'].dropna().map({'S':0,'C':1,'Q':2}).astype(int)
    
    
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    
    #创建其他特征
    df['FamilySize'] = df['SibSp'] + df['Parch']  
    df['Age*Class'] = df.Agefill * df.Pclass
    df['Female*high'] = 0
    for i in range(len(df)):
        if df.iloc[i]['Gender']==0 and df.iloc[i]['Pclass']!=3:
            df['Female*high'][i] = 1
    df['Agechild'] = 0
    for i in range(len(df)):
        if df.iloc[i]['Age'] > 15:
            df['Agechild'][i] = 1
    #df['ExpGender*Pclass'] = (df['Gender'].map({0:math.exp(0),1:math.exp(1)}).astype(float)) * df.Pclass
    
    #预处理数据：删除无效信息
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Age'], axis=1)
    df = df.dropna()
    

    locals()[name] = df
    

  
# normalize the data attributes

# standardize the data attributes
X_train = train.drop(["Survived","PassengerId","Agefill"],axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId",axis=1).copy()

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

print(logreg.score(X_train, Y_train))

random_forest = RandomForestClassifier(n_estimators=300,min_samples_leaf=4,class_weight={0:0.6,1:0.4})

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('C:/ProgramData/Anaconda3/ds/titanic1.csv', index=False)

coeff_df = DataFrame(df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
print(coeff_df)


clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

print (cross_val_score(clf, X_train, Y_train, cv=5))

clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X_train,Y_train)
predictions = clf.predict(X_train)




bad_cases = train.loc[train['PassengerId'].isin(train[predictions != Y_train]['PassengerId'].values)]
bad_cases
bad_cases.to_csv('C:/ProgramData/Anaconda3/ds/bad_cases.csv', index=False)
