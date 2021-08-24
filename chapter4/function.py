import numpy as np
np.set_printoptions(precision=3)
import pandas as pd
pd.set_option('display.max_columns', 50)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


def reload_file():
    import os, importlib
    os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/20_Tensorflow2/chapter4')
    import function
    importlib.reload(function)


def preprocessing1(n, input_dim):
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

    # モデルを使用して学習する
    x_train, x_validation, t_train, t_validation = train_test_split(
                                                            x, t, test_size=0.2)

    return x_train, x_validation, t_train, t_validation

def preprocessing2(n, input_dim):
    x1 = np.random.randn(n, input_dim) + np.array([0, 6])
    x2 = np.random.randn(n, input_dim) + np.array([4, 3])
    x3 = np.random.randn(n, input_dim) + np.array([8, 0])

    t1 = np.array([[1, 0, 0] for i in range(n)])
    t2 = np.array([[0, 1, 0] for i in range(n)])
    t3 = np.array([[0, 0, 1] for i in range(n)])

    x = np.concatenate((x1, x2, x3), axis=0)
    t = np.concatenate((t1, t2, t3), axis=0)

    x = x.astype('float32') # 訓練データをfloat64からfloat32に変換
    t = t.astype('float32') # 正解ラベルをfloat64からfloat32に変換

    return x, t


'''ch4-1_tf.py'''
# モデルの定義
class MLP1(tf.keras.Model):
    # 多層パーセプトロン
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        # 隠れ層、活性化関数はシグモイド
        self.l1 = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')
        # 出力層、活性化関数はシグモイド
        self.l2 = tf.keras.layers.Dense(output_dim, activation='sigmoid')

    def call(self, x):
        # MLPのインスタンスからコールバックされる関数
        h = self.l1(x) # 第1層の出力
        y = self.l2(h) # 出力層の出力

        return y

# 損失関数(クロスエントロピー誤差)
bce1 = tf.keras.losses.BinaryCrossentropy()

def loss1(t, y):
    return bce1(t, y)

# バックプロパゲーションによるパラメータの更新
model1 = MLP1(2, 1)
optimizer1 = tf.keras.optimizers.SGD(learning_rate=0.1)

def train_step1(x, y):
    with tf.GradientTape() as tape:
        outputs = model1(x)
        tmp_loss = loss1(y, outputs)

    grads = tape.gradient(tmp_loss, # 現在のステップ誤差
                        model1.trainable_variables) # バイアス、重みリストを取得

    optimizer1.apply_gradients(zip(grads, model1.trainable_variables))

    return tmp_loss


'''ch4-2_tf.py'''
# モデルの作成
class MLP2(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        # 隠れ層
        self.l1 = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')
        self.l2 = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, x):
        h = self.l1(x) # 第1層の出力
        y = self.l2(h) # 出力層の出力

        return y


# 損失関数
cce = tf.keras.losses.CategoricalCrossentropy()

def loss2(t, y):
    return cce(t, y)


# 勾配降下アルゴリズムによるパラメータの更新処理
optimizer2 = tf.keras.optimizers.SGD(learning_rate=0.1)
train_loss = tf.keras.metrics.Mean() # 損失を記録
train_acc = tf.keras.metrics.CategoricalAccuracy() # 精度を記録
model2 = MLP2(2, 3)

def train_step2(x, t):
    with tf.GradientTape() as tape:
        outputs = model2(x) # 順伝播の出力値
        tmp_loss = loss2(t, outputs) # 誤差

    # 誤差の勾配を計算
    grads = tape.gradient(tmp_loss, model2.trainable_variables)
    # バイアス、重みを更新
    optimizer2.apply_gradients(zip(grads, model2.trainable_variables))

    # 損失をMeanオブジェクトに記録
    train_loss(tmp_loss)
    # 精度をCategoricalAccuracyオブジェクトに記録
    train_acc(t, outputs)

    return tmp_loss


'''ch4-3_tf.py'''
def scale_pre(x_train, x_test):
    # (60000, 28, 28)の訓練データを(60000, 784)の2階テンソルに変換
    tr_x = x_train.reshape(-1, 784)
    # 訓練データをfloat32(浮動小数点数)型に、255で割ってスケール変換する
    tr_x = tr_x.astype('float32') / 255

    # (10000, 28, 28)のテストデータを(10000, 784)の2階テンソルに変換
    ts_x = x_test.reshape(-1, 784)
    # テストデータをfloat32(浮動小数点数)型に、255で割ってスケール変換する
    ts_x = ts_x.astype('float32') / 255

    return tr_x, ts_x


# モデルの作成
class MLP3(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()

        # 隠れ層
        self.l1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        # 出力層
        self.l2 = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, x):
        h = self.l1(x) # 第1層の出力
        y = self.l2(h) # 出力層の出力

        return y


# 損失関数
cce3 = tf.keras.losses.CategoricalCrossentropy()

def loss3(t, y):
    return cce3(t, y)


# パラメータの更新処理
optimizer3 = tf.keras.optimizers.SGD(learning_rate=0.1)
train_loss3 = tf.keras.metrics.Mean()
train_acc3 = tf.keras.metrics.CategoricalAccuracy()
model3 = MLP3(256, 10)

def train_step3(x, t):
    with tf.GradientTape() as tape:
        outputs = model3(x) # 順伝播の出力値
        tmp_loss = loss3(t, outputs) # 誤差

    # 誤差の勾配を計算
    grads = tape.gradient(tmp_loss, model3.trainable_variables)
    # バイアス、重みを更新
    optimizer3.apply_gradients(zip(grads, model3.trainable_variables))

    # 損失をMeanオブジェクトに記録
    train_loss3(tmp_loss)
    # 精度をCategoricalAccuracyオブジェクトに記録
    train_acc3(t, outputs)

    return tmp_loss
