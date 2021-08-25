'''CNNの実装'''
import os
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/20_Tensorflow2/chapter5')

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras

reload_function1()
from function1 import *


#%% Fashion-MNISTデータセットの読み込み
(x_train, t_train), (x_test, t_test) = keras.datasets.fashion_mnist.load_data()

# 前処理
x_train, x_test = preprocessing(x_train, x_test)
x_train, x_val, t_train, t_val = \
    train_test_split(x_train, t_train, test_size=0.2)


#%% モデルを生成して学習する
epochs = 100
batch_size = 64
steps = x_train.shape[0] // batch_size
steps_val = x_val.shape[0] // batch_size

train_loss = function1.train_loss
train_acc = function1.train_acc
val_loss = function1.val_loss
val_acc = function1.val_acc

model = function1.model


#%% 学習を行う
for epoch in range(epochs):
    # 訓練データと正解ラベルをシャッフル
    x_, t_ = shuffle(x_train, t_train, random_state=1)

    # 1ステップにおけるミニバッチを使用した学習
    for step in range(steps):
        start = step * batch_size # ミニバッチの先頭インデックス
        end = start + batch_size  # ミニバッチの末尾のインデックス
        # ミニバッチでバイアス、重みを更新
        train_step(x_[start:end], t_[start:end])

    # 検証データによるモデルの評価
    for step_val in range(steps_val):
        start = step_val * batch_size # ミニバッチの先頭インデックス
        end = start + batch_size      # ミニバッチの末尾のインデックス
        # 検証データのミニバッチで損失と精度を測定
        val_step(x_[start:end], t_[start:end])

    # 1エポックごとに結果を出力
    print('epoch({}) train_loss: {:.4} train_acc: {:.4}'
          'val_loss: {:.4} val_acc: {:.4}'.format(
              epoch+1,
              train_loss.result(), # 訓練データの損失を出力
              train_acc.result(),  # 訓練データの精度を出力
              val_loss.result(),   # 検証データの損失を出力
              val_acc.result()     # 検証データの精度を出力
              ))


#%% テストデータによるモデルの評価
# 損失を記録するオブジェクトを生成
test_loss = keras.metrics.Mean()
# 精度を記録するオブジェクトを生成
test_acc = keras.metrics.SparseCategoricalAccuracy()

# テストデータで予測して損失と精度を取得
test_step(x_test, t_test)

print('test_loss: {:.4f}, test_acc: {:.4f}'.format(
    test_loss.result(),
    test_acc.result()
))
