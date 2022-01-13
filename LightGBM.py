# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 18:23:51 2021

@author: Administrator
"""

import lightgbm as lgb
from lightgbm import LGBMClassifier
import os 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score # 정확도 함수
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve,f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib
from sklearn.tree import export_graphviz
from IPython.display import Image
from subprocess import call
import pydot
import mglearn
import catboost as cb
import csv 
from bayes_opt import BayesianOptimization
from sklearn.multioutput import MultiOutputClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics

path1 = "C:/Users/Administrator/Desktop/상우형의 사랑/증상정보"
list1 = os.listdir(path1)
fullpath = []
for i in list1:
    fullpath.append(path1 + '/' + i)
fullpath.sort(reverse=True)   

search = pd.read_excel(fullpath[-1])
search.info()

train = pd.read_excel('C:\\Users\\Administrator\\Desktop\\상우형의 사랑/ML/train data.xlsx')
test = pd.read_excel('C:\\Users\\Administrator\\Desktop\\상우형의 사랑/ML/test data.xlsx')
train = train.iloc[:,1:]
test = test.iloc[:,1:]

train = search.join(train.set_index('Name'), on = 'Name')
test = search.join(test.set_index('Name'), on = 'Name')

train[train.isna().any(axis=1)]
train = train.dropna()
len(train) #784

test[test.isna().any(axis=1)]
test = test.dropna()
len(test) #197

mtrain_X = train.iloc[:,81:247]  #MACCS train
mtest_X = test.iloc[:,81:247] #MACCS test

ftrain_X = train.iloc[:,247:]  #PubChem train
ftest_X = test.iloc[:,247:] #PubChem test

train_y2 = train.iloc[:,2:81]
test_y2 = test.iloc[:,2:81]

train_ds = lgb.Dataset(mtrain_X, label = train_y2)
test_ds = lgb.Dataset(mtest_X, label = test_y2)
# params = {learning_rate=0.01,
#           max_depth= 16,
#           boosting= 'gbdt',
#           objective= 'regression',
#           metric= 'mse',
#           is_training_metric= True,
#           num_leaves=144,
#           feature_fraction= 0.9,
#           bagging_fraction= 0.7,
#           bagging_freq= 5,
#           seed=2020}

import lightgbm as lgb
# clf = lgb.LGBMClassifier()
# classifier = MultiOutputClassifier(LGBMClassifier(mlearning_rate=0.01,
#           max_depth= 16,
#           boosting= 'gbdt',
#           objective= 'regression',
#           metric= 'mse',
#           is_training_metric= True,
#           num_leaves=144,
#           feature_fraction= 0.9,
#           bagging_fraction= 0.7,
#           bagging_freq= 5,
#           seed=2020))
# clf = Pipeline([('classify', classifier)])
# clf.fit(ftrain_X, train_y2)
# # print(clf.score(ftrain_X, train_y2))
# y_pred = clf.predict(ftest_X)
# metrics.f1_score(test_y2, y_pred, average='micro')

    
def Score(learning_rate, num_leaves, max_depth, min_child_weight, colsample_bytree, feature_fraction, bagging_fraction, lambda_l1, lambda_l2):
    # params = {'max_depth' : int(max_depth),
    #           'eta' : eta, 
    #           'objective' : 'binary:logistic',
    #           'eval_metric' : 'auc',
    #           'early_stoppings' : 50 }
    model = MultiOutputClassifier(LGBMClassifier(learning_rate=learning_rate,
                                n_estimators = 1000,
                                #boosting = 'dart',
                                num_leaves = int(round(num_leaves)),
                                max_depth = int(round(max_depth)),
                                min_child_weight = int(round(min_child_weight)),
                                colsample_bytree = colsample_bytree,
                                feature_fraction = max(min(feature_fraction, 1), 0),
                                bagging_fraction = max(min(bagging_fraction, 1), 0),
                                lambda_l1 = max(lambda_l1, 0),
                                lambda_l2 = max(lambda_l2, 0)
                               ))
    clf = Pipeline([('classify', model)])
    clf.fit(mtrain_X, train_y2)
    # print(clf.score(mtrain_X, train_y2))
    y_pred = clf.predict(mtest_X)

#     # 각종 metric 계산
    f1=f1_score(test_y2, y_pred, average='micro')
    # print(f1)
    return f1
pbounds = {'learning_rate' : (0.0001, 0.05),
           'num_leaves': (300, 600),
           'max_depth': (2, 25),
           'min_child_weight': (30, 100),
           'colsample_bytree': (0, 0.99),
           'feature_fraction': (0.0001, 0.99),
           'bagging_fraction': (0.0001, 0.99),
           'lambda_l1' : (0, 0.99),
           'lambda_l2' : (0, 0.99),
          }

# Bayesian optimization 객체 생성
# f : 탐색 대상 함수, pbounds : hyperparameter 집합
# verbose = 2 항상 출력, verbose = 1 최댓값일 때 출력, verbose = 0 출력 안함
# random_state : Bayesian Optimization 상의 랜덤성이 존재하는 부분을 통제 
bo=BayesianOptimization(f=Score, pbounds=pbounds, verbose=2, random_state=1 )    
bo.maximize(init_points=30, n_iter=70, acq='ei', xi=0.01)