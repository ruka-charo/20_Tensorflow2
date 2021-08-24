import os, importlib
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/20_Tensorflow2/chapter4')
import numpy as np
np.set_printoptions(precision=3)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

from function import *
import function
importlib.reload(function)


#%% データの準備
np.random.seed(123)
input_dim = 2
n = 500

x, t = preprocessing2(n, input_dim)


#%% モデルの構築
# モデルオブジェクトの生成
model = Sequential()

model.add(Dense(2, activation='sigmoid')) # 第1層を追加
model.add(Dense(3, activation='softmax')) # 第2層を追加

# 評価関数
optimizer = optimizers.SGD(learning_rate=0.1)
model.compile(optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])


#%% モデルの学習
epoch = 200
batch_size = 32

history = model.fit(x, t,
                    epochs=epoch,
                    batch_size=batch_size,
                    verbose=1,
                    validation_split=0.2) # 検証用データの用意
