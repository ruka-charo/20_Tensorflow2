'''CNNの実装'''
import os
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/20_Tensorflow2/chapter5')

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import datasets, models, layers, optimizers, regularizers

from function1 import preprocessing


#%% Fashion-MNISTデータセットの読み込み
(x_train, t_train), (x_test, t_test) = datasets.fashion_mnist.load_data()
x_train, x_test = preprocessing(x_train, x_test)


#%% CNNの構築
weight_decay = 1e-4
learning_rate = 0.1

model = models.Sequential()

# (第1層)畳み込み層1
model.add(layers.Conv2D(
        filters=64, #フィルターの数
        kernel_size=(3, 3), # フィルターのサイズ
        padding='same', # ゼロパディング
        input_shape=x_train[0].shape, # 入力のサイズ
        kernel_regularizer=keras.regularizers.l2(weight_decay), # 正則化
        activation='relu'))

# (第2層) 畳み込み層2
model.add(layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        activation='relu'))

# (第3層) プーリング層1
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# (第4層) 畳み込み層3
model.add(layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        padding='same',
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        activation='relu'))

# (第5層) プーリング層2
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# ドロップアウト40%
model.add(layers.Dropout(0.4))

# Flatten(フラット化)
model.add(layers.Flatten())

# (第6層) 全結合層
model.add(layers.Dense(128, activation='relu'))

# (第7層) 出力層
model.add(layers.Dense(10, activation='softmax'))


model.summary()


#%% モデルの学習
epoch = 100
batch_size = 64

history = model.fit(x_train,               # 訓練データ
                    t_train,               # 正解ラベル
                    batch_size=batch_size, # ミニバッチのサイズを設定
                    epochs=epoch,          # エポック数を設定
                    verbose=1,             # 進捗状況を出力する
                    validation_split=0.2,  # 20パーセントのデータを検証に使用
                    shuffle=True)


#%% テストデータによるモデルの評価
test_loss, test_acc = model.evaluate(x_test, t_test, verbose=0)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
