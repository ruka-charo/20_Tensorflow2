'''画像データの拡張処理'''
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# CIFAR-10データセットをロード
(x_train, t_train), (x_test, t_test) = datasets.cifar10.load_data()

#%% 画像を描画
num_classes = 10 # 分類先のクラスの数
pos = 1          # 画像の描画位置を保持する変数

# クラスの数だけ繰り返す
for target_class in range(num_classes):
    # 各クラスに分類される画像のインデックスを保持するリスト
    target_idx = []

    # クラスiが正解の場合の正解ラベルのインデックスを取得する
    for i in range(len(t_train)):
        # i行、0列の正解ラベルがtarget_classと一致するか
        if t_train[i][0] == target_class:
            # クラスiが正解であれば正解ラベルのインデックスをtargetIdxに追加
            target_idx.append(i)

    np.random.shuffle(target_idx) # クラスiの画像のインデックスをシャッフル
    plt.figure(figsize=(20, 20))  # 描画エリアを横25インチ、縦3インチにする

    # シャフルした最初の10枚の画像を描画
    for idx in target_idx[:10]:
        plt.subplot(10, 10, pos)  # 10行、10列の描画領域のpos番目の位置を指定
        plt.imshow(x_train[idx])  # Matplotlibのimshow()で画像を描画
        pos += 1

plt.show()


'''ImageDataGeneratorによる色々な処理'''
#%% 回転処理(最大90度)
datagen = ImageDataGenerator(rotation_range=90)

# バッチサイズの数だけ拡張データを作成
g = datagen.flow(x_train, t_train, batch_size, shuffle=True)
# 拡張データをリストに格納
x_batch, t_batch = g.next()
draw(x_batch)


#%% 色々な処理
datagen = ImageDataGenerator(width_shift_range=0.5) # 並行移動 最大0.5
datagen = ImageDataGenerator(height_shift_range=0.5) # 垂直移動 最大0.5
datagen = ImageDataGenerator(zoom_range=0.8) # ランダムに拡大 最大0.5
datagen = ImageDataGenerator(horizontal_flip=True) # 左右をランダムに反転
datagen = ImageDataGenerator(vertical_flip=True) # 上下をランダムに反転
datagen = ImageDataGenerator(channel_shift_range=0.7) # 画像のチャンネルをランダムに移動 最大0.7
