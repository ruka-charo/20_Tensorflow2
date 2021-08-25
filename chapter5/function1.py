import numpy as np
np.set_printoptions(precision=3)
import pandas as pd
pd.set_option('display.max_columns', 50)

from tensorflow import keras


def reload_function1():
    import os, importlib
    os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/20_Tensorflow2/chapter5')
    import function1
    importlib.reload(function1)


'''ch5-1_tf.py'''
def preprocessing(x_train, x_test):
    # 訓練データを正規化
    x_train = x_train.astype('float32') / 255
    # テストデータを正規化
    x_test =  x_test.astype('float32') / 255

    # (60000, 28, 28)の3階テンソルを(60000, 28, 28, 1)の4階テンソルに変換
    x_train = x_train.reshape(-1, 28, 28, 1)
    # (10000, 28, 28)の3階テンソルを(10000, 28, 28, 1)の4階テンソルに変換
    x_test = x_test.reshape(-1, 28, 28, 1)

    return x_train, x_test


# モデルの生成
class CNN(keras.Model):
    def __init__(self, output_dim, x_train):
        super().__init__()
        # 正則化の係数
        weight_decay = 1e-4

        # (第1層) 畳み込み層1
        self.c1 = keras.layers.Conv2D(
                filters=64, #フィルターの数
                kernel_size=(3, 3), # フィルターのサイズ
                padding='same', # ゼロパディング
                input_shape=x_train[0].shape, # 入力のサイズ
                kernel_regularizer=keras.regularizers.l2(weight_decay), # 正則化
                activation='relu')

        # (第2層) 畳み込み層2
        self.c2 = keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                padding='same',
                kernel_regularizer=keras.regularizers.l2(weight_decay),
                activation='relu')

        # (第3層) プーリング層1
        self.p1 = keras.layers.MaxPooling2D(pool_size=(2, 2))

        # (第4層) 畳み込み層3
        self.c3 = keras.layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                padding='same',
                kernel_regularizer=keras.regularizers.l2(weight_decay),
                activation='relu')

        # (第5層) プーリング層2
        self.p2 = keras.layers.MaxPooling2D(pool_size=(2, 2))

        # ドロップアウト40%
        self.d1 = keras.layers.Dropout(0.4)

        # Flatten(フラット化)
        self.f1 = keras.layers.Flatten()

        # (第6層) 全結合層
        self.l1 = keras.layers.Dense(128, activation='relu',)

        # (第7層) 出力層
        self.l2 = keras.layers.Dense(output_dim, activation='softmax')


        # 全ての層をリストにする
        self.ls = [self.c1, self.c2, self.p1, self.c3, self.p2,
                    self.d1, self.f1, self.l1, self.l2]


    def call(self, x):
        for layer in self.ls:
            x = layer(x)

        return x


# 損失関数の定義
cce = keras.losses.SparseCategoricalCrossentropy()
def loss(t, y):
    return cce(t, y)


# オプティマイザ
#model = CNN(10)
optimizer = keras.optimizers.SGD(learning_rate=0.1)

train_loss = keras.metrics.Mean()
train_acc = keras.metrics.SparseCategoricalAccuracy()
val_loss = keras.metrics.Mean()
val_acc = keras.metrics.SparseCategoricalAccuracy()

def train_step(x, t):
    with tf.GradientTape() as tape:
        outputs = model(x)
        tmp_loss = loss(t, outputs)

    grads = tape.gradient(tmp_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(tmp_loss)
    train_acc(t, outputs)

def val_step(x, t):
    preds = model(x) # 予測値
    tmp_loss = loss(t, preds) # 誤差
    val_loss(tmp_loss)
    val_acc(t, preds)


def test_step(x, t):
    # テストデータの予測値を取得
    preds = model(x)
    # 出力値と正解ラベルの誤差
    tmp_loss = loss(t, preds)
    # 損失をMeanオブジェクトに記録
    test_loss(tmp_loss)
    # 精度をSpase_CategoricalAccuracyオブジェクトに記録
    test_acc(t, preds)
