'''転移学習とファインチューニング'''
from tensorflow.keras import models, layers, optimizers, regularizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import LearningRateScheduler
import math


#%% VGG16モデルを学習済みの重みとともに読み込む
pre_trained_model = VGG16(include_top=False, # 前結合層は読み込まない
                        weights='imagenet', # ImageNetで学習した重みを利用
                        input_shape=(64, 64, 3)) # 入力データの形状


# 第1〜15層までの重みを凍結
for layer in pre_trained_model.layers[:15]:
    layer.trainable = False

# 第16層以降の重みを更新可能にする
for layer in pre_trained_model.layers[15:]:
    layer.trainable = True


#%% モデルの構築
model = models.Sequential()

# VGG16モデルを追加
model.add(pre_trained_model)

model.add(layers.GlobalMaxPooling2D())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation='softmax'))

# モデルのコンパイル
model.compile(loss='sparse_categorical_crossentropy',
            optimizer=optimizers.Adam(lr=0.001),
            metrics=['accuracy'])

model.summary()


#%% 転移学習を行なう
batch_size = 64

# データ拡張
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2)

# 訓練データ用のジェネレーターを生成
training_generator = datagen.flow(x_train, t_train,
                                batch_size=batch_size,
                                subset='training')
# 検証データ用のジェネレーターを生成
validation_generator = datagen.flow(x_train, t_train,
                                    batch_size=batch_size,
                                    subset='validation')

# 学習率をスケジューリングする
def step_decay(epoch):
    initial_lrate = 0.001 # 学習率の初期値
    drop = 0.5 # 減衰率は50%
    epochs_drop = 10.0 # 10エポック毎に減衰する
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))

    return lrate

# 学習率のコールバック
lrate = LearningRateScheduler(step_decay)

# ファインチューニングモデルで学習する
epochs = 50
history = model.fit(
    # 拡張データをミニバッチの数だけ生成
    training_generator,
    epochs=epochs,
    verbose=1,
    validation_data=validation_generator,
    # エポック終了後にlrateをコールバック
    callbacks=[lrate])
