import os, importlib
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/20_Tensorflow2/chapter4')

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras

#reload_file()
from function import *


#%% Fashion-MNISTデータセットの読み込み
(x_train, t_train), (x_test, t_test) = keras.datasets.fashion_mnist.load_data()

# データ前処理
# スケール処理
x_train = x_train / 255
x_test = x_test / 255


#%% モデルの生成
learning_rate = 0.1

model = keras.Sequential([
            # 入力するテンソルの形状をフラットにする
            keras.layers.Flatten(input_shape=(28, 28)),
            # 隠れ層(ユニット=256)
            keras.layers.Dense(256, activation='relu'),
            # 出力層(ニューロン=10)
            keras.layers.Dense(10, activation='softmax')
            ])

model.compile(loss='sparse_categorical_crossentropy',
            optimizer=keras.optimizers.SGD(lr=learning_rate),
            metrics=['accuracy'])

model.summary()


#%% モデルの学習
epoch = 100
batch_size = 64

history = model.fit(x_train, t_train,
                    epochs=epoch,
                    batch_size=batch_size,
                    verbose=1,
                    validation_split=0.2)

#%% テストデータによるモデルの評価
test_loss, test_acc = model.evaluate(x_test, t_test, verbose=0)
print('loss:', test_loss)
print('acc:', test_acc)

# epoch毎のログ
history.history


#%% モデルの別の作り方
model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
            optimizer=keras.optimizers.SGD(lr=learning_rate),
            metrics=['accuracy'])

model.summary()
