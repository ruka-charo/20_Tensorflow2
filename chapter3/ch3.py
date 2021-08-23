'''第3章 Tensorflowの基本'''
import os, importlib
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/20_Tensorflow2/chapter3')
import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import japanize_matplotlib
plt.style.use('seaborn')
import pandas as pd
pd.set_option('display.max_columns', 50)

from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from function import *
import function
importlib.reload(function)


#%% データの準備(回帰問題)
data = pd.read_csv('sample/chap03/03_02/sales.csv')
train_x = data['cost']
train_y = data['sales']

plt.plot(train_x, train_y, 'o')
plt.show()

# データの準備(分類問題)
data2 = pd.read_csv('sample/chap03/03_04/negaposi.csv')
x = np.array(data2[['x1', 'x2']])
t = np.array(data2['t'])

# データの標準化
ss = StandardScaler()
data_std = ss.fit_transform(data)

train_x_std = data_std[:, 0]
train_y_std = data_std[:, 1]

plt.plot(train_x_std, train_y_std, 'o')
plt.show()


#%% tf.GradientTapeクラスで回帰問題を解く ============================================
# Variableオブジェクトを用意する
a = tf.Variable(0.) # 重み(傾き)保持の変数
b = tf.Variable(0.) # バイアス(切片)保持の変数

#%% 最適化処理の実行
learning_rate = 0.1 # 学習率
epochs = 50 # 学習回数

for i in range(epochs):
    # 自動微分による勾配計算を記録するブロック
    with tf.GradientTape() as tape:
        y_pred = model1(train_x_std, a, b)
        tmp_loss = loss(y_pred, train_y_std)

    # tapeに記録された操作を使用して誤差の勾配を計算
    gradients = tape.gradient(tmp_loss, [a, b])

    # 勾配降下法の更新式を適用してパラメータ値を更新
    a.assign_sub(learning_rate * gradients[0])
    b.assign_sub(learning_rate * gradients[1])

    # 学習5回毎に結果を出力
    if (i + 1) % 5 == 0:
        # 処理回数とa, bの値を出力
        print(f'Step:{i+1} a = {a.numpy()} b = {b.numpy()}')
        #損失を出力
        print(f'Loss = {tmp_loss}')

#%% グラフの可視化
plt.scatter(train_x_std, train_y_std)
y_learned = a*train_x_std + b
plt.plot(train_x_std,  y_learned, 'r')
plt.grid(True)
plt.show()


#%% 多項式の回帰問題を解く =========================================================
# Variableオブジェクトを用意する
b = tf.Variable(0.)
w1 = tf.Variable(0.)
w2 = tf.Variable(0.)
w3 = tf.Variable(0.)
w4 = tf.Variable(0.)

#%% 最適化処理の実行
learning_rate = 0.01 # 学習率を設定
epochs = 200         # 学習回数

for i in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = model2(train_x_std, b, w1, w2, w3, w4)
        tmp_loss = loss(y_pred, train_y_std)
    gradients = tape.gradient(tmp_loss, [b, w1, w2, w3, w4])

    # 勾配降下法の更新式を適用してパラメータ値を更新
    b.assign_sub(learning_rate * gradients[0])
    w1.assign_sub(learning_rate * gradients[1])
    w2.assign_sub(learning_rate * gradients[2])
    w3.assign_sub(learning_rate * gradients[3])
    w4.assign_sub(learning_rate * gradients[4])

    # 学習50回毎に結果を出力
    if (i + 1) % 50 == 0:
        print('Step:{}\n a1 = {} a2 = {}\n a3 = {} a4 = {}\n b = {}'.format(
            i + 1, w1.numpy(), w2.numpy(), w3.numpy(), w4.numpy(), b.numpy()))
        print(' Loss = {}'.format(tmp_loss))


#%% パーセプトロンによる二値分類 ====================================================
# 訓練データで学習を行う
w = learn_weights(x, t)
