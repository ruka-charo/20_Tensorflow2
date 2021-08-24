import os, importlib
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/20_Tensorflow2/chapter4')
import numpy as np
np.set_printoptions(precision=3)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

from function import *
#import function
#importlib.reload(function)


#%% データの準備
np.random.seed(123)
input_dim = 2
n = 500

x_train, x_validation, t_train, t_validation = preprocessing1(n, input_dim)


#%% モデルの構築
# モデルオブジェクトの生成
model = Sequential()

model.add(Dense(2, activation='sigmoid')) # 第1層を追加
model.add(Dense(1, activation='sigmoid')) # 第2層を追加

# オプティマイザーはSGD、学習率はデフォルト(0.1)
optimizer = optimizers.SGD(learning_rate=0.1)

# 評価関数に精度を求める関数を使用
model.compile(optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy'])

#%% モデルの学習
epoch = 50

history = model.fit(x_train, t_train,
                    epochs=epoch,
                    batch_size=32,
                    verbose=1, # 進捗状況を出力
                    validation_data=(x_validation, t_validation))

model.summary()
