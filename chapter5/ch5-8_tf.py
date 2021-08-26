'''モデルの保存、呼び出し'''
from tensorflow import keras


#%% 学習済みの重みを保存する
model.save_weights('model_weights', save_format='tf')

#%% モデルを再生成してコンパイル、保存した重みを読み込む
loaded_model = MLP(256, 10)

# モデルをコンパイル
loaded_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

# 保存した重みを読み込む
loaded_model.load_weights('model_weights')
