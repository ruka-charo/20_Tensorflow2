import os, importlib
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/20_Tensorflow2/chapter4')

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras

reload_file()
from function import *


#%% Fashion-MNISTデータセットの読み込み
(x_train, t_train), (x_test, t_test) = keras.datasets.fashion_mnist.load_data()

#%% データ前処理
# スケール処理
tr_x, ts_x = scale_pre(x_train, x_test)

# 正解ラベルのOneHot化
class_num = 10

tr_t = keras.utils.to_categorical(t_train, class_num)
ts_t = keras.utils.to_categorical(t_test, class_num)

# 検証用データの作成
x_train, x_validation, t_train, t_validation = train_test_split(
                                                    tr_x, tr_t, test_size=0.2)

#%% モデルを生成して学習
epochs = 100
batch_size = 64
steps = x_train.shape[0] // batch_size
model = function.model3

# 学習
for epoch in range(epochs):
    x_, t_ = shuffle(x_train, t_train, random_state=1)

    # 1ステップの学習
    for step in range(steps):
        start = steps * step     # ミニバッチの先頭インデックス
        end = start + batch_size # ミニバッチの末尾のインデックス
        # ミニバッチでバイアス、重みを更新して誤差を取得
        tmp_loss = train_step3(x_[start:end], t_[start:end])

    # 20エポック毎に結果を出力
    if (epoch + 1) % 10 == 0:
        print(f'epoch:{epoch+1} loss:{train_loss3.result()} acc:{train_acc3.result()}')

model3.summary()

#%% 検証データによるモデルの評価
val_preds = model3(x_validation)

categorical_acc = tf.keras.metrics.CategoricalAccuracy()
categorical_acc.update_state(t_validation, val_preds)

validation_acc = categorical_acc.result().numpy()
validation_loss = loss3(t_validation, val_preds)

print(f'loss: {validation_loss}\nacc: {validation_acc}')


#%% テストデータによるモデルの評価
test_preds = model3(ts_x)

categorical_acc = tf.keras.metrics.CategoricalAccuracy()
categorical_acc.update_state(ts_t, test_preds)

test_acc = categorical_acc.result().numpy()
test_loss = loss3(ts_t, test_preds)

print(f'loss: {test_loss}\nacc: {test_acc}')
