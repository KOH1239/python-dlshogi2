import os
import sys
sys.path.append('..')

import numpy as np
import datetime
from tqdm import tqdm
import pickle

from collections import deque

import torch
import torch.nn.functional as F

from timm.scheduler import CosineLRScheduler

from pydlshogi2.dataloader import HcpeDataLoader

from pydlshogi2.network.dlshogi_encoder import CNNEncoder
from pydlshogi2.network.CaptioningTransformer_decoder import CaptioningTransformer
from pydlshogi2.captioning_utils.token_collate import collate_func


class ConfigTrain(object):
    '''
    ハイパーパラメータ、システム共通変数の設定
    '''
    def __init__(self):

        # ハイパーパラメータ
        self.dim_embedding = 300   # 埋め込み層の次元
        self.dim_feedforward = 128 # FNNの中間特徴量次元
        self.num_heads = 4         # マルチヘッドアテンションのヘッド数
        self.num_layers = 8        # Transformerデコーダ層の層数
        self.lr = 0.001            # 学習率
        self.dropout = 0.3         # dropout確率
        self.batch_size = 32       # ミニバッチ数
        self.num_epochs = 100       # エポック数→Colab無料版でテストする際は10未満に修正を推奨

        # パスの設定
        self.train_data = "/workspace/train_AtoR.hcpe"
        self.test_data = "/workspace/test_AtoR.hcpe"
        self.dlshogi_model_path = "/workspace/model/model_resnet10_swish-072_for_caption"
        self.dlshogi_network = "kifcaption"

        self.comment_file = "/workspace/kif_caption/comments_AtoR.db"
        self.word_to_id_file = '/workspace/kif_caption/word_to_id_AtoR.pkl'
        self.save_directory = '/workspace/kif_caption/model'

        # 検証に使う学習セット内のデータの割合
        self.val_ratio = 0.3

        # データローダーに使うCPUプロセスの数
        self.num_workers = 4

        # 学習に使うデバイス
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 移動平均で計算する損失の値の数
        self.moving_avg = 100

def train():
    print("start train")
    config = ConfigTrain()

    # 辞書（単語→単語ID）の読み込み
    with open(config.word_to_id_file, 'rb') as f:
        word_to_id = pickle.load(f)

    # 辞書サイズを保存
    vocab_size = len(word_to_id)
        
    # モデル出力用のディレクトリを作成
    os.makedirs(config.save_directory, exist_ok=True)

    # train dataloader
    train_dataloader = HcpeDataLoader(config.train_data, config.batch_size, config.device, shuffle=True)

    # test data loader
    test_dataloader = HcpeDataLoader(config.test_data, config.batch_size, config.device)

    # モデルの定義
    encoder = CNNEncoder(config.dim_embedding, config.dlshogi_network, config.dlshogi_model_path)
    decoder = CaptioningTransformer(
        config.dim_embedding, config.dim_feedforward,
        config.num_heads, config.num_layers, vocab_size,
        word_to_id['<null>'], config.dropout)
    encoder.to(config.device)
    decoder.to(config.device)

    # 損失関数の定義
    loss_func = lambda x, y: F.cross_entropy(
        x, y, ignore_index=word_to_id.get('<null>', None))

    # 最適化手法の定義
    params = list(decoder.parameters()) + \
    list(encoder.linear.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr)

    # WarmupとCosine Decayを行うスケジューラを利用
    scheduler = CosineLRScheduler(
        optimizer, t_initial=config.num_epochs, lr_min=1e-4,
        warmup_t=20, warmup_lr_init=5e-5, warmup_prefix=True)

    # 学習経過の書き込み
    now = datetime.datetime.now()
    train_loss_file = '{}/6-5_train_loss_{}.csv'\
        .format(config.save_directory, now.strftime('%Y%m%d_%H%M%S'))
    val_loss_file = '{}/6-5_val_loss_{}.csv'\
        .format(config.save_directory, now.strftime('%Y%m%d_%H%M%S'))
    
    total_train_samples = len(train_dataloader)
    total_train_step = total_train_samples // config.batch_size

    total_test_samples = len(test_dataloader)
    total_test_step = total_test_samples // config.batch_size

    # 学習
    val_loss_best = float('inf')
    for epoch in range(config.num_epochs):
        with tqdm(train_dataloader, total=total_train_step) as pbar:
            pbar.set_description(f'[エポック {epoch + 1}]')

            # 学習モードに設定
            encoder.train()
            decoder.train()

            train_losses = deque()
            for x1, x2, move_label, result, index in pbar:
                x1, x2, move_label, result, captions, lengths = collate_func(x1, x2, move_label, result, index, word_to_id, config.comment_file)
                # ミニバッチを設定
                captions = captions.to(config.device)

                optimizer.zero_grad()

                # エンコーダ-デコーダモデル
                features = encoder(x1, x2)
                # 最後の単語から次を予測する必要はないため最後の単語を除外
                captions_in = captions[:, :-1]
                outputs = decoder(features, captions_in)

                # 損失の計算
                # <start>の予測は行わないため除外
                targets = captions[:, 1:]
                # 単語軸が第1軸である必要があるため、転置
                outputs = outputs.transpose(1, 2)
                loss = loss_func(outputs, targets)

                # 誤差逆伝播
                loss.backward()

                optimizer.step()

                # 学習時の損失をログに書き込み
                train_losses.append(loss.item())
                if len(train_losses) > config.moving_avg:
                    train_losses.popleft()
                pbar.set_postfix({
                    'loss': torch.Tensor(train_losses).mean().item()})
                with open(train_loss_file, 'a') as f:
                    print(f'{epoch}, {loss.item()}', file=f)

        # 学習率を表示
        print(f'学習率: {scheduler._get_lr(epoch)}')

        # 検証
        with tqdm(test_dataloader, total=total_test_step) as pbar:
            pbar.set_description(f'[検証]')

            # 評価モード
            encoder.eval()
            decoder.eval()

            val_losses = []
            for x1, x2, move_label, result, index in pbar:
                x1, x2, move_label, result, captions, lengths = collate_func(x1, x2, move_label, result, index, word_to_id, config.comment_file)

                # ミニバッチを設定
                captions = captions.to(config.device)

                # エンコーダ-デコーダモデル
                features = encoder(x1, x2)
                # 最後の単語から次を予測する必要はないため最後の単語を除外
                captions_in = captions[:, :-1]
                outputs = decoder(features, captions_in)

                # 損失の計算
                # <start>の予測は行わないため除外
                targets = captions[:, 1:]
                # 単語軸が第1軸である必要があるため、転置
                outputs = outputs.transpose(1, 2)
                loss = loss_func(outputs, targets)
                val_losses.append(loss.item())

                # Validation Lossをログに書き込み
                with open(val_loss_file, 'a') as f:
                    print(f'{epoch}, {loss.item()}', file=f)

        # Loss 表示
        val_loss = np.mean(val_losses)
        print(f'Validation loss: {val_loss}')

        # より良い検証結果が得られた場合、モデルを保存
        if val_loss < val_loss_best:
            val_loss_best = val_loss

            # エンコーダモデルを保存
            torch.save(
                encoder.state_dict(),
                f'{config.save_directory}/transformer_encoder_best.pth')

            # デコーダモデルを保存
            torch.save(
                decoder.state_dict(),
                f'{config.save_directory}/transformer_decoder_best.pth')

if __name__ == "__main__":
    train()