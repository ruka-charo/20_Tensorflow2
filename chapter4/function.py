import numpy as np
np.set_printoptions(precision=3)
import pandas as pd
pd.set_option('display.max_columns', 50)

from sklearn.preprocessing import StandardScaler
import tensorflow as tf


# ステップ関数
def step_func(x):

    return (x > 0).astype(np.int)

# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# モデルの定義
class MLP(tf.keras.Model):
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
bce = tf.keras.losses.BinaryCrossentropy()
def loss(t, y):

    return bce(t, y)

# バックプロパゲーションによるパラメータの更新
model = MLP(2, 1)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
def train_step(x, y):
    with tf.GradientTape() as tape:
        outputs = model(x)
        tmp_loss = loss(y, outputs)

    grads = tape.gradient(tmp_loss, # 現在のステップ誤差
                        model.trainable_variables) # バイアス、重みリストを取得

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return tmp_loss
