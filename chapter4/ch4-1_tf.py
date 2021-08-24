'''第4章 例題で学ぶTensorflowの基本'''
import os, importlib
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/20_Tensorflow2/chapter4')
import numpy as np
np.set_printoptions(precision=3)

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

x_train, x_validation, t_train, t_validation = preprocessing1(n, input_dim)


#%% モデルを使用して学習する
epochs = 50
batch_size = 32
steps = x_train.shape[0] // batch_size
# 隠れ層2ユニット、出力層1ユニットのモデルを構築
model1 = function.model1


# 学習を行う
for epoch in range(epochs):
    epoch_loss = 0. # 1エポック毎の損失を保持する変数
    x_, t_ = shuffle(x_train, t_train, random_state=0) # シャッフル

    # 1ステップにおけるミニバッチを使用した学習
    for step in range(steps):
        start = steps * step # ミニバッチの先頭のインデックス
        end = start + batch_size # ミニバッチの末尾のインデックス

        # ミニバッチでバイアス、重みを更新して誤差を取得
        tmp_loss = train_step1(x_[start: end], t_[start: end])

    # 1ステップ終了時の誤差を取得
    epoch_loss = tmp_loss.numpy()

    # 10エポックごとに結果を出力
    if (epoch + 1) % 10 == 0:
        print('epoch({}) loss: {:.3}'.format(epoch+1, epoch_loss))

# モデルの概要
model1.summary()


#%% モデルの評価
t_preds = model1(x_validation) # 検証データの予測値

# バイナリデータの精度を取得するオブジェクトを生成(閾値はデフォルトの0.5)
bn_acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
bn_acc.update_state(t_validation, t_preds) # 測定データの設定

# 検証データの精度を取得
validation_acc = bn_acc.result().numpy()
# 検証データの損失を取得
validation_loss = loss1(t_validation, t_preds)
print(f'acc:{validation_acc}\nloss:{validation_loss}')
