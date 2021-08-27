'''LSTMの実装'''
from tensorflow.keras import models, layers, optimizers, regularizers


# RNNの構築
weight_decay = 1e-4
model = models.Sequential()

# 入力層
model.add(layers.InputLayer(input_shape=(28, 28)))

# 中間層(LSTMブロック)
model.add(layers.LSTM(units=128, dropout=0.25, return_sequences=True))
model.add(layers.LSTM(units=128, dropout=0.25, return_sequences=True))
model.add(layers.LSTM(units=128, dropout=0.5, return_sequences=False,
                    kernel_regularizer=regularizers.l2(weight_decay))) # 正則化

# 出力層
model.add(layers.Dense(units=10, activation='softmax'))


# オブジェクトをコンパイル
model.compile(loss='sparse_categorical_crossentropy',
            optimizer=optimizers.Adam(),
            metrics=['accuracy'])

model.summary()
