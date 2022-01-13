# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 17:35:58 2021

@author: Administrator
"""

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
'''몰드레드 데이터 분할
1. 엑셀파일로 변경후 경로 설정
2. 분할(아래 4줄 풀어서 실행 위에 블럭'''
# XY = pd.read_excel('C:\\Users\\Administrator\\Desktop\\상우형의 사랑/ML/train data.xlsx')
# total_Xdatas=XY.iloc[:,:1826]
# Y=XY.iloc[:,1827:]
# train_x, test_x, train_y, test_y = train_test_split(total_Xdatas, train_labels, test_size = 0.1, random_state = 42) 
    

# cb_dtrain = cb.Pool(data = mtrain_X, label = train_y2) # 학습 데이터를 Catboost 모델에 맞게 변환
# # params = {'max_depth' : 10,
# #               'learning_rate': 0.01, # Step Size
# #               'n_estimators' : 100,
# #               'eval_metric' : 'AUC',
# #               'loss_function' : 'MultiClass' }
# # cb_model = cb.train(pool = cb_dtrain, params = params) # 학습 진행
# # preds = np.argmax(cb_model.predict(mtest_X), axis = 1) 
# # accuracy = accuracy_score(test_y2, preds)
# classifier = MultiOutputClassifier(CatBoostClassifier(max_depth=10,
#               learning_rate= 0.01, # Step Size
#               n_estimators= 3,
#               eval_metric='AUC',
#               loss_function='MultiClass'))
# clf = Pipeline([('classify', classifier)])
# ''' 파라미터 튜닝 '''
# # clf.set_params()
# print (clf)
# clf.fit(ftrain_X, train_y2)
# # print(clf.score(ftrain_X, train_y2))
# y_pred = clf.predict(ftest_X)
# metrics.f1_score(test_y2, y_pred[0], average='micro')


def Score(n_estimators, depth, learning_rate):   
    # params = {'max_depth' : int(max_depth),
    #           'eta' : eta, 
    #           'objective' : 'binary:logistic',
    #           'eval_metric' : 'auc',
    #           'early_stoppings' : 50 }
    classifier = MultiOutputClassifier(CatBoostClassifier(max_depth=int(depth),
              learning_rate= learning_rate, # Step Size
              n_estimators= int(n_estimators),
              eval_metric='AUC',
              loss_function='MultiClass'))
    # model = CatBoostClassifier(verbose = 0,
    #                         n_estimators = int(n_estimators),
    #                         learning_rate = learning_rate,
    #                         subsample = subsample, 
    #                         l2_leaf_reg = l2_leaf_reg,
    #                         max_depth = int(depth),
    #                         num_leaves = int(num_leaves),
    #                         random_state = 88,
    #                         grow_policy = "Lossguide",
    #                         max_bin = int(max_bin),  
    #                         use_best_model = True, 
    #                         model_size_reg = model_size_reg,
    #                        )
    
    clf = Pipeline([('classify', classifier)])
    clf.fit(ftrain_X, train_y2)
    # print(clf.score(mtrain_X, train_y2))
    y_pred = clf.predict(ftest_X)
    
#     # 각종 metric 계산
    f1=f1_score(test_y2, y_pred[0], average='micro')
    print(f1)
    return f1
pbounds = {"n_estimators": (150,400),
           "depth": (2,10),
           "learning_rate": (0.01, 0.2)}

  # Bayesian optimization 객체 생성
  # f : 탐색 대상 함수, pbounds : hyperparameter 집합
  # verbose = 2 항상 출력, verbose = 1 최댓값일 때 출력, verbose = 0 출력 안함
  # random_state : Bayesian Optimization 상의 랜덤성이 존재하는 부분을 통제 
bo=BayesianOptimization(f=Score, pbounds=pbounds, verbose=2, random_state=1 )    
bo.maximize(init_points=20, n_iter=35, acq='ei', xi=0.01)




