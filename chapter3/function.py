import numpy as np
np.set_printoptions(precision=3)
import pandas as pd
pd.set_option('display.max_columns', 50)

from sklearn.preprocessing import StandardScaler
import tensorflow as tf


# 回帰モデル
def model1(x, a, b):
    y = a*x + b

    return y

# 多項式回帰モデル
def model2(x, b, w1, w2, w3, w4):
    y = b + w1*x + w2*pow(x, 2) + w3*pow(x, 3) + w4*pow(x, 4)

    return y

# MSE(平均二乗誤差)
def loss(y_pred, y_true):

    return tf.math.reduce_mean(tf.math.square(y_pred - y_true))


# パーセプトロン(分類関数)
def classify(x, w):
    if w @ x >= 0:
        return 1
    else:
        return -1

# 重みの更新
def learn_weights(x, t):
    w = np.random.rand(2) # 重みの初期化
    epochs = 5
    count = 0

    # 指定した回数だけ重みの学習を繰り返す
    for i in range(epochs):
        for element_x, element_t in zip(x, t):
            if classify(element_x, w) != element_t:
                w += element_t * element_x
                print('更新後のw = ', w)
        count += 1
        # ログの出力
        print('[{}回目]: w = {}***'.format(count, w))

    return w
