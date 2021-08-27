'''学習の早期終了、学習率減衰'''
import tensorflow as tf


# TensorflowAPIにはEarlyStoppingが用意されていないので自分で作る必要あり
class EarlyStopping:
    def __init__(self, patience=10, verbose=0):
        '''
        parameters
            patience(int): 監視するエポック数
            verbose(int) : 早期終了の出力フラグ
        '''
        # インスタンス変数の初期化
        # 監視中のエポック数のカウンターを初期化
        self.epoch = 0
        # 比較対象の損失を無限大'inf'で初期化
        self.pre_loss = float('inf')
        # 監視対象のエポック数をパラメーターで初期化
        self.patience = patience
        # 早期終了メッセージの出力フラグをパラメーターで初期化
        self.verbose = verbose

    def __call__(self, current_loss):
        '''
        Parameters:
            current_loss(float): 1エポック終了後の検証データの損失
        Return:
            True:監視回数の上限までに前エポックの損失を超えた場合
            False:監視回数の上限までに前エポックの損失を超えない場合
        '''
        # 前エポックの損失より大きくなった場合
        if self.pre_loss < current_loss:
            # カウンターを1増やす
            self.epoch += 1
            # 監視回数の上限に達した場合
            if self.epoch > self.patience:
                # 早期終了のフラグが1の場合
                if self.verbose:
                    # メッセージを出力
                    print('early stopping')
                # 学習を終了するTrueを返す
                return True
        # 前エポックの損失以下の場合
        else:
            # カウンターを0に戻す
            self.epoch = 0
            # 損失の値を更新する
            self.pre_loss = current_loss

        # 監視回数の上限までに前エポックの損失を超えなければ
        # Falseを返して学習を続行する
        # 前エポックの損失を上回るが監視回数の範囲内であれば
        # Falseを返す必要があるので、return文の位置はここであることに注意
        return False


#%% モデルに反映させる方法
# オブジェクトを生成(ImageDataGeneratorの生成と同じタイミングで)
ers = EarlyStopping(patience=20, verbose=1)

# エポック毎のfor文中に引数に損失をセット
if ers(val_loss.result()):
    break
