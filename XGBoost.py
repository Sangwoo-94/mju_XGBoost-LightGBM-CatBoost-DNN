# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:06:51 2021

@author: Administrator
"""
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import pandas as pd
import os
import numpy as np
import xgboost as xgb ## XGBoost 불러오기
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
import sklearn.metrics as metrics
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

path1 = "C:/Users/Administrator/Desktop/상우형의 사랑/증상정보"
list1 = os.listdir(path1)
fullpath = []
for i in list1:
    fullpath.append(path1 + '/' + i)
fullpath.sort(reverse=True)   

search = pd.read_excel(fullpath[-1])
search.info()

# train = pd.read_excel('C:\\Users\\Administrator\\Desktop\\상우형의 사랑/ML/train data.xlsx')
# test = pd.read_excel('C:\\Users\\Administrator\\Desktop\\상우형의 사랑/ML/test data.xlsx')
m_1 = pd.read_excel('C:\\Users\\Administrator\\Desktop\\상우형의 사랑\\mordred_3000_last.xlsx')
# m_Y = pd.read_excel('C:\\Users\\Administrator\\Desktop\\상우형의 사랑\\체크리스트_211130.xlsx')
m_X = m_1.transpose()
m_X = m_X.drop(['Unnamed: 0'],axis=0)
mordred_X = m_X.iloc[:,:1826]
mordred_Y = m_X.iloc[:,1833:]
X_train, X_test, y_train, y_test = train_test_split(mordred_X, mordred_Y, test_size = 0.1, random_state = 42) 

# train = train.iloc[:,1:]
# test = test.iloc[:,1:]

# train = search.join(train.set_index('Name'), on = 'Name')
# test = search.join(test.set_index('Name'), on = 'Name')

# train[train.isna().any(axis=1)]
# train = train.dropna()
# len(train) #784

# test[test.isna().any(axis=1)]
# test = test.dropna()
# len(test) #197

# mtrain_X = train.iloc[:,81:247]  #MACCS train
# ftrain_X = train.iloc[:,247:]  #PubChem train

# mtest_X = test.iloc[:,81:247] #MACCS test
# ftest_X = test.iloc[:,247:] #PubChem test

# train_y2 = train.iloc[:,2:81]
# test_y2 = test.iloc[:,2:81]



# # xtrain, xtest, ytrain, ytest=train_test_split(x, y, train_size=0.8, random_state=88)
# # print(len(xtest)) 
# # dtrain = xgb.DMatrix(data=mtrain_X, label = train_y2)
# # dtest = xgb.DMatrix(data=mtest_X, label=test_y2)
# # wlist = [(dtrain, 'train'), (dtest,'eval')]
# def Score(max_depth,learning_rate,n_estimators,gamma,min_child_weight,subsample,colsample_bytree):
#     # params = {'max_depth' : int(max_depth),
#     #           'eta' : eta, 
#     #           'objective' : 'binary:logistic',
#     #           'eval_metric' : 'auc',
#     #           'early_stoppings' : 50 }
#     classifier = MultiOutputClassifier(XGBClassifier(max_depth=int(max_depth),
#                               learning_rate=learning_rate,
#                               n_estimators=int(n_estimators),
#                               gamma=0.3,
#                               min_child_weight=min_child_weight,
#                               subsample=subsample,
#                               colsample_bytree=colsample_bytree,
#                               nthread=-1))
#     clf = Pipeline([('classify', classifier)])
#     clf.fit(mtrain_X, train_y2)
#     # print(clf.score(mtrain_X, train_y2))
#     y_pred = clf.predict(mtest_X)

# #     # 각종 metric 계산
#     f1=f1_score(test_y2, y_pred, average='micro')
#     # print(f1)
#     return f1
# pbounds = {'max_depth': (3, 7),
#                 'learning_rate': (0.01, 0.2),
#                 'n_estimators': (5000, 10000),
#                 'gamma': (0, 100),
#                 'min_child_weight': (0, 3),
#                 'subsample': (0.5, 1),
#                 'colsample_bytree' :(0.2, 1)
#                 }

#   # Bayesian optimization 객체 생성
#   # f : 탐색 대상 함수, pbounds : hyperparameter 집합
#   # verbose = 2 항상 출력, verbose = 1 최댓값일 때 출력, verbose = 0 출력 안함
#   # random_state : Bayesian Optimization 상의 랜덤성이 존재하는 부분을 통제 
# bo=BayesianOptimization(f=Score, pbounds=pbounds, verbose=2, random_state=1 )    
# bo.maximize(init_points=2, n_iter=10, acq='ei', xi=0.01)
#  # 메소드를 이용해 최대화 과정 수행
#  # init_points :  초기 Random Search 갯수
#  # n_iter : 반복 횟수 (몇개의 입력값-함숫값 점들을 확인할지! 많을 수록 정확한 값을 얻을 수 있다.)
#  # acq : Acquisition Function들 중 Expected Improvement(EI) 를 사용
#  # xi : exploration 강도 (기본값은 0.0)


# # parameters = {'num_rounds': (200, 1000)}













# # classifier = MultiOutputClassifier(XGBClassifier(max_depth=int(3),
# #                               learning_rate=0.2,
# #                               n_estimators=int(500),
# #                               gamma=0.3,
# #                               min_child_weight=0.1,
# #                               subsample=7,
# #                               colsample_bytree=1,
# #                               nthread=-1))
# # clf = Pipeline([('classify', classifier)])
# # ''' 파라미터 튜닝 '''
# # # clf.set_params()
# # print (clf)
# # clf.fit(ftrain_X, train_y2)
# # print(clf.score(ftrain_X, train_y2))
# # y_pred = clf.predict(ftest_X)

# # #     # 각종 metric 계산
# # print(metrics.f1_score(test_y2, y_pred, average='micro'))
# # model = MultiOutputClassifier(estimator=XGBClassifier(max_depth=int(3),
# #                               learning_rate=0.2,
# #                               n_estimators=int(500),
# #                               gamma=0.3,
# #                               min_child_weight=0.1,
# #                               subsample=7,
# #                               colsample_bytree=1,
# #                               nthread=-1))







# # np.array(mtrain_X), np.array(train_y2)
# # dtrain = xgb.DMatrix(data=np.array(mtrain_X), label = np.array(train_y2))
# # dtest = xgb.DMatrix(data=mtest_X, label=test_y2)
# # wlist = [(dtrain, 'train'), (dtest, 'eval')]

# # evals = [(mtrain_X, train_y2)]
# # def XGB_cv(max_depth, learning_rate, n_estimators, gamma
# #            , min_child_weight, subsample
# #            , colsample_bytree, silent=True, nthread=-1):
# #     # 모델 정의
# #     # classifier = MultiOutputClassifier(XGBClassifier())
# #     model = MultiOutputClassifier(estimator=XGBClassifier(max_depth=int(max_depth),
# #                              learning_rate=learning_rate,
# #                              n_estimators=int(n_estimators),
# #                              gamma=gamma,
# #                              min_child_weight=min_child_weight,
# #                              subsample=subsample,
# #                              colsample_bytree=colsample_bytree,
# #                              nthread=nthread))
# #     # model = classifier.set_params(max_depth=int(max_depth),
# #     #                          learning_rate=learning_rate,
# #     #                          n_estimators=int(n_estimators),
# #     #                          gamma=gamma,
# #     #                          min_child_weight=min_child_weight,
# #     #                          subsample=subsample,
# #     #                          colsample_bytree=colsample_bytree,
# #     #                          nthread=nthread
# #     #                          )
# #     # 모델 훈련
# #     model.fit(mtrain_X, train_y2, eval_metric="logloss", eval_set=evals)

# #     # 예측값 출력
# #     y_pred = model.predict(mtest_X)

# #     # 각종 metric 계산
# #     # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# #     # r2 = r2_score(test_y2, y_pred)
# #     # mape = mean_absolute_percentage_error(y_test, y_pred)
# #     f1 = metrics.f1_score(test_y2, y_pred, average='micro')
# #     # 오차 최적화로 사용할 metric 반환
# #     return f1

# # pbounds = {'max_depth': (3, 7),
# #                 'learning_rate': (0.01, 0.2),
# #                 'n_estimators': (5000, 10000),
# #                 'gamma': (0, 100),
# #                 'min_child_weight': (0, 3),
# #                 'subsample': (0.5, 1),
# #                 'colsample_bytree' :(0.2, 1)
# #                 }

# # # Bayesian optimization 객체 생성
# # # f : 탐색 대상 함수, pbounds : hyperparameter 집합
# # # verbose = 2 항상 출력, verbose = 1 최댓값일 때 출력, verbose = 0 출력 안함
# # # random_state : Bayesian Optimization 상의 랜덤성이 존재하는 부분을 통제
# # bo=BayesianOptimization(f=XGB_cv, pbounds=pbounds, verbose=2, random_state=1 )

# # # 메소드를 이용해 최대화 과정 수행
# # # init_points :  초기 Random Search 갯수
# # # n_iter : 반복 횟수 (몇개의 입력값-함숫값 점들을 확인할지! 많을 수록 정확한 값을 얻을 수 있다.)
# # # acq : Acquisition Function들 중 Expected Improvement(EI) 를 사용
# # # xi : exploration 강도 (기본값은 0.0)
# # bo.maximize(init_points=5, n_iter=5, acq='ei', xi=0.01)

# # # ‘iter’는 반복 회차, ‘target’은 목적 함수의 값, 나머지는 입력값을 나타냅니다.
# # # 현재 회차 이전까지 조사된 함숫값들과 비교하여, 현재 회차에 최댓값이 얻어진 경우,
# # # bayesian-optimization 라이브러리는 이를 자동으로 다른 색 글자로 표시하는 것을 확인할 수 있습니다

# # # 찾은 파라미터 값 확인
# # print(bo.max)





