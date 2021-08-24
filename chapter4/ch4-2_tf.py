import os, importlib
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/20_Tensorflow2/chapter4')
import numpy as np
np.set_printoptions(precision=3)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf

from function import *
#import function
#importlib.reload(function)


#%% データの準備
np.random.seed(123)
input_dim = 2
n = 500

x_train, x_validation, t_train, t_validation = preprocessing2(n, input_dim)


#%% モデルを生成して学習する
epochs = 200
batch_size = 32
steps = x_train.shape[0] // batch_size
# 隠れ層2ユニット、出力層3ユニットのモデル
model2 = function.model2

# 記録用
train_loss = function.train_loss
train_acc = function.train_acc

# 学習を行う
for epoch in range(epochs):
    x_, t_ = shuffle(x_train, t_train, random_state=1)

    # 1ステップの学習
    for step in range(steps):
        start = steps * step     # ミニバッチの先頭インデックス
        end = start + batch_size # ミニバッチの末尾のインデックス
        # ミニバッチでバイアス、重みを更新して誤差を取得
        tmp_loss = train_step2(x_[start:end], t_[start:end])

    # 20エポック毎に結果を出力
    if (epoch + 1) % 20 == 0:
        print(f'epoch:{epoch+1} loss:{train_loss.result()} acc:{train_acc.result()}')

model2.summary()


#%% モデルの評価
val_preds = model2(x_validation)

# 精度オブジェクトの準備
categorical_acc = tf.keras.metrics.CategoricalAccuracy()
categorical_acc.update_state(t_validation, val_preds)

# 精度を計算
validation_acc = categorical_acc.result().numpy()
validation_loss = loss2(t_validation, val_preds)

print(f'loss: {validation_loss}\nacc: {validation_acc}')
