import os
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/20_Tensorflow2/chapter5')

from tensorflow.keras import datasets
from tensorflow.keras import models, layers, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# データの準備
(x_train, t_train), (x_test, t_test) = datasets.cifar10.load_data()


#%% モデルの構築
weight_decay = 1e-4
learning_rate = 0.1

# CNNを構築
model = models.Sequential()

# (第1層)畳み込み層1 正則化を行う
model.add(
    layers.Conv2D(
        filters=32,                    # フィルターの数は32
        kernel_size=(3,3),             # 3×3のフィルターを使用
        input_shape=x_train.shape[1:], # 入力データの形状
        padding='same',                # ゼロパディングを行う
        kernel_regularizer=regularizers.l2(weight_decay),
        activation='relu'              # 活性化関数はReLU
        ))

# (第2層)プーリング層1：ウィンドウサイズは2×2
model.add(
    layers.MaxPooling2D(pool_size=(2,2)))

# (第3層)畳み込み層2 正則化を行う
model.add(
    layers.Conv2D(
        filters=128,                # フィルターの数は128
        kernel_size=(3,3),          # 3×3のフィルターを使用
        padding='same',             # ゼロパディングを行う
        kernel_regularizer=regularizers.l2(weight_decay),
        activation='relu'           # 活性化関数はReLU
        ))

# (第4層)プーリング層2：ウィンドウサイズは2×2
model.add(
    layers.MaxPooling2D(pool_size=(2,2)))

# (第5層)畳み込み層3 正則化を行う
model.add(
    layers.Conv2D(
        filters=256,                # フィルターの数は256
        kernel_size=(3,3),          # 3×3のフィルターを使用
        padding='same',             # ゼロパディングを行う
        kernel_regularizer=regularizers.l2(weight_decay),
        activation='relu'           # 活性化関数はReLU
        ))

# (第6層)プーリング層2：ウィンドウサイズは2×2
model.add(
    layers.MaxPooling2D(pool_size=(2,2)))

# Flatten
model.add(layers.Flatten())

# ドロップアウト：ドロップアウトは40％
model.add(layers.Dropout(0.4))

# （第7層）全結合層
model.add(
    layers.Dense(
        512,                   # ニューロン数は512
        activation='relu'))    # 活性化関数はReLU


# （第8層）出力層
model.add(
    layers.Dense(
        10,                    # 出力層のニューロン数は10
        activation='softmax')) # 活性化関数はソフトマックス


# Sequentialオブジェクトのコンパイル
model.compile(
    # 損失関数はスパースラベル対応クロスエントロピー誤差
    loss='sparse_categorical_crossentropy',
    # オプティマイザーはSGD
    optimizer=optimizers.SGD(lr=learning_rate),
    # 学習評価として正解率を指定
    metrics=['accuracy'])

# モデルのサマリを表示
model.summary()


#%% 学習
batch_size = 64
epochs = 100

# データ拡張
datagen = ImageDataGenerator(
    rescale=1.0/255.0,      # ピクセル値を255で割って正規化する
    validation_split=0.2,   # 20パーセントのデータを検証用にする
    rotation_range=15,      # 15度の範囲でランダムに回転させる
    width_shift_range=0.1,  # 横サイズの0.1の割合でランダムに水平移動
    height_shift_range=0.1, # 縦サイズの0.1の割合でランダムに垂直移動
    horizontal_flip=True,  # 水平方向にランダムに反転、左右の入れ替え
    zoom_range=0.2,         # ランダムに拡大
)

# 訓練データ用のジェネレーターを生成
training_generator = datagen.flow(x_train, t_train,
                                  batch_size=batch_size,
                                  subset='training')      # 訓練用のデータを生成
# 検証データ用のジェネレーターを生成
validation_generator = datagen.flow(x_train, t_train,
                                    batch_size=batch_size,
                                    subset='validation') # 検証用のデータを生成

# 学習を行う
history = model.fit(
    # 拡張データをミニバッチの数だけ生成
    training_generator,
    epochs=epochs,         # エポック数
    verbose=1,             # 学習の進捗状況を出力する
    validation_data=validation_generator,  # 20パーセントのデータを検証に使用
)


#%% テストデータによるモデルの評価
# テストデータを正規化
x_test = x_test / 255

# 学習済みのモデルにテストデータを入力して損失と精度を取得
test_loss, test_acc = model.evaluate(x_test, t_test, verbose=0)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
