from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from DeepLeraning.model.model_fun import *
from DeepLeraning.utils.load_data import series_to_supervised,split_data
from keras.callbacks import EarlyStopping
"""
本文是LSTM多元预测
用3个步长的数据预测1个步长的数据
包含：
对数据进行缩放，缩放格式为n行*feature列，因为原数据含新增，故不作做差分
对枚举列（风向）进行数字编码
构造3->1的监督学习数据
构造网络开始预测
将预测结果重新拼接为n行*8列数据
数据逆缩放，求RSME误差
"""
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)

# load dataset
dataset = read_csv('datasets/state1.csv', header=0, index_col=0)
#dataset=dataset.diff()
values = dataset.values

# 标准化/放缩 特征值在（0,1）之间
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
print(values.shape)
# 7天预测1天时间步长，8个特征值
n_hours = 10
n_features = 1
# 构造一个3->1的监督学习型数据
reframed = series_to_supervised(scaled, n_hours, 3)
print(reframed.shape)

# split into train and test sets
values = reframed.values



# 用一年的数据来训练
# n_train_num= int(values.shape[0]*0.6)
# train = values[:n_train_num, :]
# test = values[n_train_num:, :]
# split into input and outputs


train,val,test=split_data(values, split_ratio=(6, 2, 2))

n_obs = n_hours * n_features
# 有(（7+1）*24)列数据，取前(7*24) 列作为X，倒数第8列=(第25列)作为Y
train_X, train_y = train[:, :n_obs], train[:, -n_features:]
val_X, val_y = val[:, :n_obs], val[:, -n_features:]
test_X, test_y = test[:, :n_obs], test[:, -n_features:]

#print(train_X.shape, len(train_X), train_y.shape,train_y)
# 将数据转换为3D输入，timesteps=3，3条数据预测1条 [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
val_X = val_X.reshape((val_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
train_Y = train_y.reshape((train_y.shape[0], 1, n_features))
val_Y = val_y.reshape((val_y.shape[0], 1, n_features))
test_Y = test_y.reshape((test_y.shape[0], 1, n_features))

#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
print(scaler.inverse_transform(test_y))
models=[generate_lstm_model,
        generate_seq2seq_model,
        generate_attention_model,
        generate_seq2seq_attention_model,
        cnn_lstm_attention_model]
model_name=["lstm",
        "seq2seq",
        "alstm",
        "seq2seq_alstm",
        "cnn_alstm"]
# 设计网络

y_predict=[]
for m_fun in models:
    model = m_fun(n_input=train_X.shape[1], n_out=train_Y.shape[2], n_features=n_features)
    #定义提前终止函数，即多少次损失没有明显下降便停止训练，防止出现过拟合
    callbacks = EarlyStopping(monitor='val_loss', patience=5)#提取停止
                 #ModelCheckpoint(filepath='models/covid19_main.model',monitor='val_loss',save_best_only=True)]#保存最优化模型
    # 拟合网络
    history = model.fit(train_X, train_y, epochs=150,callbacks=[callbacks], batch_size=8, validation_data=(val_X, val_y), verbose=2,
                        shuffle=False)
    # 执行预测
    predict_y = model.predict(test_X)  # 所得结果为8组数据

    #plot_model(model, to_file='images/model.jpg')
    # 计算RMSE误差值
    #model.save('models/covid19_main.model')
    from math import sqrt
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score

    feature = -1
    mean_absolute_error=mean_absolute_error(test_y[:, feature], predict_y[:, feature])
    mean_squared_error=mean_squared_error(test_y[:, feature], predict_y[:, feature])
    rmse=sqrt(mean_squared_error)#计算rmse
    r2_score=r2_score(test_y[:, feature], predict_y[:, feature])#计算r平方

    #记录数据 rmse和r平方（评估标准)
    txt_file = open("D:/文本文档.txt", "a", encoding="utf-8")  # 以写的格式打开先打开文件
    txt_file.write(str(m_fun)+"  rmse: "+str(rmse)+"   r2: "+str(r2_score))
    txt_file.write("\n")
    txt_file.close()

    inv_yhat = scaler.inverse_transform(predict_y)
    y_predict.append(inv_yhat)


print("")
test_y = test_y.reshape((test_y.shape[0], n_features))
# 对拼接好的数据进行逆缩放会原来的数据
inv_y = scaler.inverse_transform(test_y)
import matplotlib.pyplot as plt
j=0
#作图，将各个方法的预测值画出
plt.title('COVID_19')
for inv_yhat in y_predict:
    plt.plot(inv_yhat[:, feature], linestyle='--', label=str(model_name[j]))
    j = j + 1
plt.plot(inv_y[:, feature], linestyle='-', label='true_data')
plt.xlabel('Date')
plt.ylabel('Number')
#img_name = "images//" + str(model_name[i]) + ".jpg"
img_name = "images/img_new.jpg"
plt.legend()
plt.savefig(img_name)
plt.close()
