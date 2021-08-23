'''第4章 例題で学ぶTensorflowの基本'''
import os, importlib
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/20_Tensorflow2/chapter4')
import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import japanize_matplotlib
plt.style.use('seaborn')
import pandas as pd
pd.set_option('display.max_columns', 50)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf

from function import *
import function
importlib.reload(function)



#%% データの準備
np.random.seed(123)

input_dim = 2
n = 500

# 平均(3,2)の正規分布に従うデータを生成
x1 = np.random.randn(n, input_dim) + np.array([3, 2])
# 平均(7,6)の正規分布に従うデータを生成
x2 = np.random.randn(n, input_dim) + np.array([7, 6])
# x1の正解ラベル0を2階テンソルとして生成
t1 = np.array([[0] for i in range(n)])
# x2の正解ラベル1を2階テンソルとして生成
t2 = np.array([[1] for i in range(n)])


x = np.concatenate((x1, x2), axis=0)
t = np.concatenate((t1, t2), axis=0)

x = x.astype('float32')
t = t.astype('float32')


#%% モデルを使用して学習する
x_train, x_validation, t_train, t_validation = train_test_split(
                                                        x, t, test_size=0.2)

epochs = 50
batch_size = 32
steps = x_train.shape[0] // batch_size
# 隠れ層2ユニット、出力層1ユニットのモデルを構築
model = function.model


# 学習を行う
for epoch in range(epochs):
    epoch_loss = 0. # 1エポック毎の損失を保持する変数
    x_, t_ = shuffle(x_train, t_train, random_state=0) # シャッフル

    # 1ステップにおけるミニバッチを使用した学習
    for step in range(steps):
        start = steps * step # ミニバッチの先頭のインデックス
        end = start + batch_size # ミニバッチの末尾のインデックス

        # ミニバッチでバイアス、重みを更新して誤差を取得
        tmp_loss = train_step(x_[start: end], t_[start: end])

    # 1ステップ終了時の誤差を取得
    epoch_loss = tmp_loss.numpy()

    # 10エポックごとに結果を出力
    if (epoch + 1) % 10 == 0:
        print('epoch({}) loss: {:.3}'.format(epoch+1, epoch_loss))

# モデルの概要
model.summary()
