from tensorflow import keras


class CNN(keras.Model):

    def __init__(self, output_dim):
        super().__init__()
        # 正則化の係数
        weight_decay = 1e-4

        # (第1層)畳み込み層1 正則化を行う
        self.c1 = keras.layers.Conv2D(
            filters=32,                   # フィルター数32
            kernel_size=(3, 3),           # 3×3のフィルター
            padding='same',               # ゼロパディング
            input_shape=x_train[0].shape, # 入力データの形状
            kernel_regularizer=keras.regularizers.l2(
                weight_decay), # 正則化
            activation='relu'             # 活性化関数はReLU
            )

        # (第2層)プーリング層1：ウィンドウサイズは2×2
        self.p1 = keras.layers.MaxPooling2D(
            pool_size=(2, 2))             # 縮小対象の領域は2×2

        # (第3層)畳み込み層2　正則化を行う
        self.c2 = keras.layers.Conv2D(
            filters=128,                  # フィルターの数は128
            kernel_size=(3, 3),           # 3×3のフィルターを使用
            padding='same',               # ゼロパディングを行う
            kernel_regularizer=keras.regularizers.l2(
                weight_decay),           # 正則化
            activation='relu'             # 活性化関数はReLU
            )

        # (第4層)プーリング層2：ウィンドウサイズは2×2
        self.p2 = keras.layers.MaxPooling2D(
            pool_size=(2, 2))             # 縮小対象の領域は2×2

        # (第5層)畳み込み層3　正則化を行う
        self.c3 = keras.layers.Conv2D(
            filters=256,                  # フィルターの数は256
            kernel_size=(3, 3),           # 3×3のフィルターを使用
            padding='same',               # ゼロパディングを行う
            kernel_regularizer=keras.regularizers.l2(weight_decay), # 正則化
            activation='relu'             # 活性化関数はReLU
            )

        # (第6層)プーリング層3：ウィンドウサイズは2×2
        self.p3 = keras.layers.MaxPooling2D(
            pool_size=(2, 2))             # 縮小対象の領域は2×2

        # Flaten
        # ニューロン数＝4×4×256=4,096
        # (4, 4, 256)を(,4096)にフラット化
        self.f1 = keras.layers.Flatten()

        # ドロップアウト：ドロップアウトは40％
        self.d1 = keras.layers.Dropout(0.4)

        # （第7層）全結合層
        self.l1 =  keras.layers.Dense(
            512,                          # 出力層のニューロン数は512
            activation='relu')            # 活性化関数はReLU

        # （第8層）出力層
        self.l2 =  keras.layers.Dense(
            10,                           # 出力層のニューロン数は10
            activation='softmax')         # 活性化関数はソフトマックス

        # すべての層をリストにする
        self.ls = [self.c1, self.p1, self.c2, self.p2, self.c3, self.p3,
                   self.f1, self.d1, self.l1, self.l2]

    def call(self, x):
        for layer in self.ls:
            x = layer(x)

        return x


# 損失関数
cce = keras.losses.SparseCategoricalCrossentropy()

def loss(t, y):
    return cce(t, y)


# 勾配降下アルゴリズムを使用するオプティマイザーを生成
optimizer = keras.optimizers.SGD(learning_rate=0.1)

# 損失を記録するオブジェクトを生成
train_loss = keras.metrics.Mean()
# 精度を記録するオブジェクトを生成
train_acc = keras.metrics.SparseCategoricalAccuracy()
# 検証時の損失を記録するオブジェクトを生成
val_loss = keras.metrics.Mean()
# 検証時の精度を記録するオブジェクトを生成
val_acc = keras.metrics.SparseCategoricalAccuracy()


# パラメータ更新処理
def train_step(x, t):
    # 自動微分による勾配計算を記録するブロック
    with tf.GradientTape() as tape:
        # モデルに入力して順伝播の出力値を取得
        outputs = model(x)
        # 出力値と正解ラベルの誤差
        tmp_loss = loss(t, outputs)

    # tapeに記録された操作を使用して誤差の勾配を計算
    grads = tape.gradient(
        # 現在のステップの誤差
        tmp_loss,
        # バイアス、重みのリストを取得
        model.trainable_variables)
    # 勾配降下法の更新式を適用してバイアス、重みを更新
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # 損失をMeanオブジェクトに記録
    train_loss(tmp_loss)
    # 精度をCategoricalAccuracyオブジェクトに記録
    train_acc(t, outputs)


# 検証データでの評価
def val_step(x, t):
    # 検証データの予測値を取得
    preds = model(x)
    # 出力値と正解ラベルの誤差
    tmp_loss = loss(t, preds)
    # 損失をMeanオブジェクトに記録
    val_loss(tmp_loss)
    # 精度をSpase_CategoricalAccuracyオブジェクトに記録
    val_acc(t, preds)


def test_step(x, t):
    # テストデータの予測値を取得
    preds = model(x)
    # 出力値と正解ラベルの誤差
    tmp_loss = loss(t, preds)
    # 損失をMeanオブジェクトに記録
    test_loss(tmp_loss)
    # 精度をSpaseCategoricalAccuracyオブジェクトに記録
    test_acc(t, preds)
