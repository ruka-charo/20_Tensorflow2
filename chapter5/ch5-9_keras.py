from tensorflow.keras import models, layers, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau


'''EarlyStopping'''
# 早期終了を行うEarlyStoppingを生成
early_stopping = EarlyStopping(monitor='val_loss', # 監視対象
                                patience=20, # 監視する回数
                                verbose=1) # ログの出力

history = model.fit(training_generator,
                    epochs=epochs,
                    verbose=1,
                    validation_data=validation_generator,
                    callbacks=[early_stopping]) # コールバックはリストで指定


'''ReduceLROnPlateau'''
# 5エポックの間にval_accuracyが改善されなかったら学習率を0.5倍にする
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', # 監視対象
                            factor=0.5, # 学習率を減衰させる割合
                            patience=5, # 監視対象のエポック数
                            verbose=1,
                            mode='max', # 値の増加が停止したらlrを更新
                            min_lr=0.0001) # 学習率の下限

history = model.fit(training_generator,
                    epochs=epochs,
                    verbose=1,
                    validation_data=validation_generator,
                    callbacks=[reduce_lr])

# historyにlrの推移も保存されている
history.history['lr']
