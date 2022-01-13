# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 23:30:29 2021

@author: Administrator
"""

from rdkit import Chem
from mordred import Calculator, descriptors
import pandas as pd 
import numpy as np
from tensorflow import keras
# import numpy as np
# import pandas as pd
import os
from sklearn.metrics import accuracy_score # 정확도 함수
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
import csv
from sklearn.model_selection import train_test_split
import xgboost as xgb
os.chdir("C:/Users/Administrator/Desktop/상우형의 사랑")
# train_loc = "./물질-MACCSKeys.csv"
train_loc_Y = './물질-sm-증상.csv'
# train_loc = "./train.csv"
# test_loc = "./물질-증상.csv"
# arr=np.load("C:/Users/Administrator/Desktop/상우형의 사랑/물질-MACCSKeys.csv")
# trains = pd.readcsv(train_loc, delimiter=',', skip_header = 0)
# trains = pd.read_csv(train_loc)
# trains_name = trains[['Name']]
# trains_name = trains_name.values.tolist()

trains_Y = pd.read_csv(train_loc_Y)
# trains_name_Y = trains_Y[['Name']]
# trains_name_Y = trains_name_Y.values.tolist()  

# for index,i in enumerate(trains_name_Y):
#     if i not in trains_name:
#         trains_Y = trains_Y.drop(index=index, axis=0)
        
        
# for index,i in enumerate(trains_name):
#     trains_name_Y_new = trains_Y[['Name']]
#     trains_name_Y_new = trains_name_Y_new.values.tolist() 
#     if i not in trains_name_Y_new:
#         trains = trains.drop(index=index, axis=0)

        
# trains = trains.drop('CAS No.', axis=1)
# total_Xdatas = trains.drop('Name', axis=1) 
# train_datas = total_Xdatas[:800]

# total_Ydatas = trains_Y.drop('Name', axis=1) 


# create descriptor calculator with all descriptors
calc = Calculator(descriptors, ignore_3D=False)

# Calculator(descriptors, ignore_3D=False).descriptors
# len(calc.descriptors)

# len(Calculator(descriptors, ignore_3D=True, version="1.0.0"))

# n_all = len(Calculator(descriptors, ignore_3D=False).descriptors)
# # calculate single molecule

# mol = Chem.MolFromSmiles('CC(C)C1=C(C=C2C(=C1)CCC3C2(CCCC3(C)C)C)O')

df = pd.DataFrame()
df_final=pd.DataFrame()
# calc(mol)[:1826]
# calculate multiple molecule
# data = pd.read_excel('C:\\Users\\Administrator\\Desktop\\GNN\\자연물_smiles_data.xlsx')
# data = data.drop(['Unnamed: 0'],axis =1)
# data = data.transpose()
# material = data[1]
f = open('mordred_dnn.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f) 
# data = data[0]
data = trains_Y[['canonical_smiles']].values.tolist()
data_1 = trains_Y[['Name']].values.tolist()
re_num=0
re_num_1=0
for index,smi in enumerate(data):
    try:
        print(index,smi[0])
        mol = Chem.MolFromSmiles(smi[0])
        # print(mol)
        
        # j=calc(mol)[0:1826]
        # a=[]
        # a.append(j)
        # df[smi]=a
        df[smi[0]] = calc(mol)[0:1826]
        # print(df)
        # print(mol)
        re_num_1+=1
    except:
        re_num+=1
        # trains_Y=trains_Y.drop(index,axis=0)


trains_Y = trains_Y.drop('Name', axis=1)
trains_Y = trains_Y.drop('CAS No.', axis=1)
trains_Y = trains_Y.drop('canonical_smiles', axis=1)
trains_Y_trans=trains_Y.transpose()
# trains_Y_trans = trains_Y_trans.drop('canonical_smiles', axis=1)
for i in df.columns:
    for j in data:
        if i == j[0]:
            index=data.index([i])
            df_final[i] = trains_Y_trans[index]
            break
        
df_final=df_final.transpose()

total_Xdatas = df.transpose()
train_datas = total_Xdatas[:900]
train_datas = train_datas.apply(pd.to_numeric, errors='coerce').fillna(0)
train_datas = train_datas.drop([1351,1352], axis=1)
# train_datas = train_datas.drop('1352', axis=1)
# trains_Y = df_final.drop('Name', axis=1)
# trains_Y = trains_Y.drop('CAS No.', axis=1)
# trains_Y = trains_Y.drop('canonical_smiles', axis=1)
acc_list=[]
pred_list = {}
for i in trains_Y.columns:
    # train_labels = trains_Y
    trains_Y=df_final
    train_labels = trains_Y[i]
    total_Xdatas = total_Xdatas.apply(pd.to_numeric, errors='coerce').fillna(0)
    # total_Xdatas = total_Xdatas.drop([1351,1352], axis=1)
    train_labels=train_labels.astype('float')
    total_Xdatas=total_Xdatas.astype('float')
    train_x, test_x, train_y, test_y = train_test_split(total_Xdatas, train_labels, test_size = 0.1, random_state = 42) 
    
    
    # test_datas = total_Xdatas[900:]
    # test_labels = trains_Y[i][900:]
    # test_datas = test_datas.apply(pd.to_numeric, errors='coerce').fillna(0)
    # test_datas = test_datas.drop([1351,1352], axis=1)

    def create_model():
        model = keras.Sequential([
            # keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(2, activation='softmax')
        ])
    
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    
        return model
    
    model = create_model()
    
    # # # 훈련단계
    # # for i in range(10):
    
    # # for i in range(20):
    model.fit(train_x, train_y, epochs=100)
    #     # 정확도 평가 단계
    pred_probs = model.predict(test_x)
    preds = [ 1 if x > 0.5 else 0 for x in pred_probs[:,1]]
    
    
    # pred1 = model.predict(test_x)    
    # dtrain = xgb.DMatrix(data=train_x, label = train_y)
    # dtest = xgb.DMatrix(data=test_x, label=test_y)
    # wlist = [(dtrain, 'train'), (dtest,'eval')]
    # params = {'max_depth' : int(6),
    #               'eta' : 0.54, 
    #               'objective' : 'binary:logistic',
    #               'eval_metric' : 'auc',
    #               'early_stoppings' : 50
    #               }
    # # xgb_model
    # num_rounds=400
    # xgb_model = xgb.train(params = params, dtrain=dtrain, num_boost_round=num_rounds, evals=wlist)
    # pred_probs = xgb_model.predict(dtest)
    # preds = [ 1 if x > 0.5 else 0 for x in pred_probs]
    # preds = [ 1 if x > 0.5 else 0 for x in pred1]
    # test_loss, test_acc = model.evaluate(test_x,  pred1, verbose=2)
    # print('\ntest accuracy:', test_acc)
    
    accuracy = accuracy_score(test_y,preds)
    F1 = precision_score(test_y,preds)
    pred_list[i] = F1
    # AUC_last = roc_auc_score(test_labels,preds)
    wr.writerow([i,accuracy,F1,preds])
    # max_acc=0
    # if test_acc > max_acc:
        # max_acc = test_acc
    acc_list.append(F1)
    """평균 정확도"""
f.close()
print(sum(acc_list)/79)















