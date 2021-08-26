import os
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/20_Tensorflow2/chapter5')
from tensorflow import keras

# CIFAR-10データセットの読み込み
(x_train, t_train), (x_test, t_test) = keras.datasets.cifar10.load_data()


#%% モデルを生成して学習する
epochs = 100
batch_size = 64
train_steps = len(x_train)*0.8 // batch_size
val_steps = len(x_train)*0.2 // batch_size

# 出力層10ニューロンのモデルを生成
model = CNN(10)

# ImageDataGeneratorを生成
datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0 # ピクセル値を255で割って正規化する
    validation_split=0.2 # 20%のデータを検証用にする
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2
)

# 訓練用データのジェネレーターを生成
training_generator = datagen.flow(x_train, t_train,
                                batch_size=batch_size,
                                subset='training')

# 検証用データのジェネレーターを生成
validation_generator = datagen.flow(x_train, t_train,
                                batch_size=batch_size,
                                subset='validation')


#%% 学習を行う
for epoch in range(epochs):
    # 訓練時のステップカウンター
    step_counter = 0
    # 1ステップ毎にミニバッチで学習する
    for x_batch, t_batch in training_generator:
        # ミニバッチでバイアス、重みを更新
        train_step(x_batch, t_batch)
        step_counter += 1
        # すべてのステップが終了したらbreak
        if step_counter >= train_steps:
            break

    # 検証時のステップカウンター
    v_step_counter = 0
    # 検証データによるモデルの評価
    for x_val_batch, t_val_batch  in validation_generator:
        # 検証データのミニバッチで損失と精度を測定
        val_step(x_val_batch, t_val_batch)
        v_step_counter += 1
        # すべてのステップが終了したらbreak
        if v_step_counter >= val_steps:
            break

    # 1エポックごとに結果を出力
    print('epoch({}) train_loss: {:.4} train_acc: {:.4} '
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

# テストデータを正規化
x_test =  x_test.astype('float32') / 255
# テストデータで予測して損失と精度を取得
test_step(x_test, t_test)

print('test_loss: {:.4f}, test_acc: {:.4f}'.format(
    test_loss.result(),
    test_acc.result()
))
