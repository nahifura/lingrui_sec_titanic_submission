import torch
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from matplotlib import pyplot as plt
import os
import random
import joblib

# 加载模型参数
checkpoint = torch.load('../models/model_para.pth')
W = checkpoint['W']
b = checkpoint['b']
features = ['Sex','Age','SibSp','Parch','Fare','Embarked']

# 复制main中需要用到的函数
def softmax(x):
    X_exp = torch.exp(x)
    X_sum = X_exp.sum(axis=1, keepdim=True)
    return X_exp / X_sum

# 定义交叉熵
def cross_entropy_loss(y_hat, y_true):
    corect = y_hat[torch.arange(len(y_hat)), y_true]
    log_probs = torch.log(corect+1e-12)
    return -log_probs.mean()

# 向前传播
# 模型：计算y_hat
def net(X,W,b):
    return softmax(torch.matmul(X,W) + b)

#======================
# 加载预处理信息
info = joblib.load('../models/ness.pkl')
scaler = info['scaler']
train_age_median = info['train_age_median']
train_fare_median = info['train_fare_median']
train_embarked_mode = info['train_embarked_mode']
train_sibsp_average = info['train_sibsp_average']
train_parch_average = info['train_parch_average']




def softmax(x):
    X_exp = torch.exp(x)
    X_sum = X_exp.sum(axis=1, keepdim=True)
    return X_exp / X_sum

# 定义交叉熵
def cross_entropy_loss(y_hat, y_true):
    corect = y_hat[torch.arange(len(y_hat)), y_true]
    log_probs = torch.log(corect+1e-12)
    return -log_probs.mean()

# 向前传播

# 模型：计算y_hat
def net(X,W,b):
    return softmax(torch.matmul(X,W) + b)

def pre_new(new_data={}):

    # 声明全局变量，把参数弄进来，
    global W,b,train_fare_median,train_embarked_mode,train_sibsp_average,train_parch_average
    X_new = pd.DataFrame([new_data])

    # 补充缺失值
    for feature in features:
        if str(new_data[feature]) == '':
            new_data[feature] = np.nan

    X_new = pd.DataFrame([new_data])

    # 清洗数字
    for feature in features:
        if feature != 'Sex'or 'Embarked':
            X_new[feature] = pd.to_numeric(X_new[feature], errors='coerce')

    # 用训练集的统计量填补缺失值
    X_new['Age'] = X_new['Age'].fillna(train_age_median)
    X_new['Fare'] = X_new['Fare'].fillna(train_fare_median)
    X_new['Embarked'] = X_new['Embarked'].fillna(train_embarked_mode)
    X_new['Sex'] = X_new['Sex'].fillna('male')
    X_new['Parch'] = X_new['Parch'].fillna(train_parch_average)
    X_new['SibSp'] = X_new['SibSp'].fillna(train_sibsp_average)
    print(X_new)


    # 把性别转化为数字
    X_new['Sex'] = X_new['Sex'].map({'male': 0, 'female': 1})
    # 独热编码
    X_new = pd.get_dummies(X_new, columns=['Embarked'], drop_first=True)
    expected_features = scaler.feature_names_in_
    X_new = X_new.reindex(columns=expected_features, fill_value=0)

    # 标准化
    X_new_scaled = scaler.transform(X_new)

    X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

    with torch.no_grad():
        probs = net(X_new_tensor, W, b)  # 使用全局的 W 和 b
        pred = torch.argmax(probs, dim=1).item()
        prob = probs[0][pred].item()

    # 6. 打印结果并返回回
    print(f"[预测结果]: 【{'存活' if pred == 1 else '遇难'}】，置信度: {prob * 100:.2f}%")
    return pred
print('\nok!''lets predict the new data \n')
def pre():
    new={}

    for feature in features:
        new[feature] = input(f'please enter the {feature}: (if the data cannot be uesd,it will give a modest one)')
    print(f'you entered the new data: {new}')
    print(pre_new(new))
while True:
    try:
        pre()
    except:
        print('\n[atention!] youre wrong!!!please enter another feature\n')
        continue