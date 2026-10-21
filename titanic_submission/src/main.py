# 导入一堆可能有用可能没有的模块
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from matplotlib import pyplot as plt
import numpy as np
import os
import random
import pandas as pd
import joblib

# 可复现的随机数种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed()

os.makedirs('../docs', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# 读取文件
train_file = os.path.join('.','train.csv')
tf=pd.read_csv(train_file)
data_dict = {}

# 处理特征，这里舍弃了name，Ticket，Cabin
features = ['Sex','Age','SibSp','Parch','Fare','Embarked']
X = tf[features].copy()
y = tf['Survived'].copy()

# 填补缺失值函数
scaler=StandardScaler()

# 写了个函数来填补缺失值，因为训练集和测试集是分开的，二者填补的数据不能共用，所以用函数两次实现
def tianbu(X,Y):
    X['Age'] = X['Age'].fillna(Y['Age'].median())
    X['Fare'] = X['Fare'].fillna(Y['Fare'].median())
    X['Embarked'] = X['Embarked'].fillna(Y['Embarked'].mode()[0])
    return X

# 转化sex的特征为数字，同上面那个函数，转化sex，embarked时得分开所以写了个函数

def zhuanhua(X):
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
    X = pd.get_dummies(X, columns=['Embarked'], drop_first=True)
    return X

# 划分训练集，测试集
X_train_df, X_test_df, y_train_series, y_test_series = (train_test_split
                                 (X, y, test_size=0.2, random_state=42))
# 填补缺失值

# ========
train_age_median = X_train_df['Age'].median()
train_fare_median = X_train_df['Fare'].median()     #这里是计算出训练集上各个需要用到的中位数，众数，便于在后续预测新成员中补足缺失数据'''
train_embarked_mode = X_train_df['Embarked'].mode()[0]
train_sibsp_average = X_train_df['SibSp'].mean()
train_parch_average = X_train_df['Parch'].mean()
# ======

# 套用上面定义的两个函数直接使用
X_train_df = tianbu(X_train_df,X_train_df)
X_test_df = tianbu(X_test_df,X_train_df)
train_columns = X_train_df.columns.tolist()
X_train_df = zhuanhua(X_train_df)
X_test_df = zhuanhua(X_test_df)

# 标准化
X_train_df = scaler.fit_transform(X_train_df)
X_test_df = scaler.transform(X_test_df)


# 把训练集，测试集变成torch的张量，统一格式，后面才不报错
X_train = torch.tensor(X_train_df, dtype=torch.float32)
y_train = torch.tensor(y_train_series.values, dtype=torch.long)

X_test = torch.tensor(X_test_df, dtype=torch.float32)

y_test = torch.tensor(y_test_series.values, dtype=torch.long)


# 定义softmax函数
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

# 设定超参数
Ir=0.1
batch_size=64
epoches = 64



# 初始化W，b，使用正态分布随机初始化W，b
def init_params(in_features, out_features):
    W = torch.empty(in_features, out_features).normal_(mean=0.0, std=0.01)
    W.requires_grad_(True)
    b = torch.zeros(out_features, requires_grad=True)
    return W, b


W,b = init_params(in_features=7, out_features=2)

# 画图函数：用来画模型在训练集上的损失随训练轮次的变化
def plt_ok(x,y,title=None,xlabel=None,ylabel=None,figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x,y)
    plt.plot(x, y, label=ylabel, color='red', marker='x')
    plt.grid(True)
    plt.show()
# 本来想把两个画图函数合二为一的，但是失败了，这里重写了个
# 绘制模型在训练集上和测试集上随学习轮次推移准确率的对比图
def plt_two2(epoches,train_accuracy_history,test_accuracy_history):
    epochs_list = list(range(1, epoches + 1))

    plt.figure(figsize=(10, 6))

    plt.plot(epochs_list, train_accuracy_history, label='Train Accuracy', color='blue', marker='o')

    plt.plot(epochs_list, test_accuracy_history, label='Test Accuracy', color='red', marker='x')

    plt.title('Model Accuracy on Train vs Test Sets')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# 计算模型的准确率
def accuracy(X, y, b, W):
    with torch.no_grad():
        y_hat = net(X, W, b)
        pred_y = torch.argmax(y_hat, dim=1)
        correct = (pred_y == y).sum()
        return correct.item() / y.shape[0]

# 训练函数
def train(X,y,epoches,batch_size,Ir):
    #声明全局变量W，b
    global W,b

    # 定义小批量要用到的参数
    n_samples = X.shape[0]
    batch_count = 0

    # 把绘图要用的列表提前定义个空列表
    loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []



    #学习的主循环
    for epoch in range(epoches):
        # 小批量量随机切片
        indices = torch.randperm(X.shape[0])
        X_shuffled = X[indices,:]
        y_shuffled = y[indices]

        epoch_loss = 0  #初始损失

        loss = cross_entropy_loss(net(X_shuffled,W,b),y_shuffled)   #用交叉熵算损失

        # 开始训练
        for i in range(0,n_samples,batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # 向前传播算预测值和损失
            y_hat = net(X_batch,W,b)
            loss = cross_entropy_loss(y_hat,y_batch)

            # 反向传播算梯度
            if W.grad is not None:
                W.grad.zero_()
                b.grad.zero_()

            # 求梯度
            loss.backward()

            # 梯度下降的更新公式
            with torch.no_grad():
                W-=Ir*W.grad
                b-=Ir*b.grad

            # 算一轮学习的损失值
            epoch_loss += loss.item()
            loss = loss.item()

            batch_count += 1    # 批次+1

        # 把绘图需要的历史列表全部加上对应的值
        loss_history.append(epoch_loss / batch_count)   # 损失历史
        train_accuracy = accuracy(X_train,y_train,b,W)  # 将要加入训练集准确率历史的计算
        test_accuracy = accuracy(X_test,y_test,b,W)     # 将要加入测试集准确率历史的计算
        train_accuracy_history.append(train_accuracy)   # 加入训练集准确率历史
        test_accuracy_history.append(test_accuracy)     # 加入测试集准确率历史

        if (epoch + 1) % 10 == 0:   #每十轮显示损失
            print(f'Epoch{epoch + 1:2d},loss:{epoch_loss / batch_count:4f}')

     # 绘图（直接用前面的函数）
    plt_ok(list(range(0,len(loss_history))), loss_history,xlabel='epoches_nun',ylabel='train_loss')
    plt_two2(epoches,train_accuracy_history,test_accuracy_history)


# 对新数据的预测函数
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




# 这坨史山终于可以开始启动了
print('[开始训练]')
train(X_train,y_train,epoches,batch_size,Ir)

# 保存 W 和 b
torch.save({'W': W, 'b': b}, '../models/model_para.pth')

# 保存预处理信息
preprocess_info = {
    'scaler': scaler,
    'train_age_median': train_age_median,
    'train_fare_median': train_fare_median,
    'train_embarked_mode': train_embarked_mode,
    'train_sibsp_average': train_sibsp_average,
    'train_parch_average': train_parch_average}
joblib.dump(preprocess_info, '../models/ness.pkl')

print("[保存成功！请查看 models 文件夹]")







new={}
print(f'\n结束训练，[准确率]：{accuracy(X_test,y_test,b,W)*100}%\n')
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




