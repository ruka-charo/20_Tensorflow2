'''モデルの保存、読み込み'''

#%% モデルを保存
with open('model.json', 'w') as json_file:
    json_file.write(model.to_json())

# パラメータを保存
model.save_weights('weights.h5')


#%% モデルの読み込み
model_r = model_from_json(open('model.json', 'r').read())

# 復元したモデルのコンパイル
model_r.compile(loss='sparse_categorical_crossentropy',
                optimizer = keras.optimizers.Adam,
                metrics=['accuracy'])

# 重みの読み込み
model_r.load_weights('weights.h5')
