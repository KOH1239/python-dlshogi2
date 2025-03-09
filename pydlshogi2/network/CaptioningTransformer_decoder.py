import torch
from torch import nn

# 位置エンコーディング
class PositionalEncoding(nn.Module):
    '''
    位置エンコーディング （Positional encoding）
    dim_embedding: 埋込み次元
    dropout      : ドロップアウト確率
    max_len      : 入力の最大系列長
    temperature  : 温度定数
    '''
    def __init__(self, dim_embedding: int, dropout: float=0.1,
                 max_len: int=5000, temperature=10000):
        super().__init__()

        assert dim_embedding % 2 == 0

        self.dropout = nn.Dropout(p=dropout)

        dim_t = torch.arange(0, dim_embedding, 2)
        dim_t = dim_t / dim_embedding
        dim_t = temperature ** dim_t

        x_encoding = torch.arange(max_len).unsqueeze(1)
        x_encoding = x_encoding / dim_t

        # 位置情報を保持するテンソル
        pe = torch.zeros(max_len, dim_embedding)
        pe[:, ::2] = x_encoding.sin()
        pe[:, 1::2] = x_encoding.cos()

        # PEをメモリに保存
        self.register_buffer('pe', pe)

    '''
    位置エンコーディングの順伝播
    x: 位置エンコーディングを埋め込む対象のテンソル,
       [バッチサイズ, 系列長, 埋め込み次元]
    '''
    def forward(self, x: torch.Tensor):
        seq = x.shape[1]
        x = x + self.pe[:seq]
        x = self.dropout(x)

        return x

# TransformerDecoderの実装
class TransformerDecoderLayer(nn.Module):
    '''
    Transformerデコーダ層
    dim_hidden     : 特徴量次元
    num_heads      : アテンションヘッドの数
    dim_feedforward: FNNの中間特徴量次元
    dropout        : ドロップアウト確率
    '''
    def __init__(self, dim_hidden: int, num_heads: int,
                 dim_feedforward: int=2048, dropout: int=0.1):
        super().__init__()

        # 単語ベクトルに対するマルチヘッド自己アテンション
        self.self_attention = nn.MultiheadAttention(
            dim_hidden, num_heads, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim_hidden)
        self.dropout1 = nn.Dropout(dropout)

        # エンコーダとデコーダ中間出力に対するマルチヘッドアテンション
        self.cross_attention = nn.MultiheadAttention(
            dim_hidden, num_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim_hidden)
        self.dropout2 = nn.Dropout(dropout)

        # FNN＆レイヤー正規化
        self.fnn = nn.Sequential(
            nn.Linear(dim_hidden, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_hidden)
        )
        self.norm3 = nn.LayerNorm(dim_hidden)
        self.dropout3 = nn.Dropout(dropout)

    '''
    Transformerデコーダ層の順伝播
    tgt     : デコーダへの入力系列,
              [バッチサイズ, 系列長, 埋め込み次元]
    memory  : エンコーダ層の特徴量, [バッチサイズ, 1, 埋め込み次元]
    tgt_mask: 入力系列のマスク, [系列長, 系列長]
    '''
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: torch.Tensor=None):

        # デコーダ入力に対するマスク付きマルチヘッド自己アテンション
        tgt2 = self.self_attention(
            query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)  # 残差接続
        tgt = self.norm1(tgt)            # レイヤー正規化

        # エンコーダとデコーダ中間出力に対するマルチヘッド交差アテンション
        tgt2 = self.cross_attention(
            query=tgt, key=memory, value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)  # 残差接続
        tgt = self.norm2(tgt)            # レイヤー正規化

        # FNN&レイヤー正規化
        tgt2 = self.fnn(tgt)
        tgt = tgt + self.dropout3(tgt2)  # 残差接続
        tgt = self.norm3(tgt)            # レイヤー正規化

        return tgt

# CaptioningTransformerの実装
class CaptioningTransformer(nn.Module):
    '''
    CaptioningTransformerのコンストラクタ
    dim_embedding  : 埋め込み次元
    dim_feedforward: FNNの中間特徴次元
    num_heads      : マルチヘッドアテンションのヘッド数
    num_layers     : Transformerデコーダ層の数
    vocab_size     : 辞書の次元
    null_index     : NULLのID
    dropout        : ドロップアウト確率
    '''
    def __init__(self, dim_embedding: int, dim_feedforward: int,
                 num_heads: int, num_layers: int, vocab_size: int,
                 null_index: int, dropout: float=0.5):
        super().__init__()

        # 単語埋め込み
        self.embed = nn.Embedding(
            vocab_size, dim_embedding, padding_idx=null_index)

        # 位置エンコーディング
        self.positional_encoding = PositionalEncoding(dim_embedding)

        # Transformerデコーダ
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                dim_embedding, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # 単語出力分布計算
        self.linear = nn.Linear(dim_embedding, vocab_size)

        # パラメータ初期化
        self._reset_parameters()

    '''
    パラメータの初期化関数
    '''
    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    ''' CaptioningTransformerの順伝播処理
    features: 画像特徴量 [バッチサイズ, 埋め込み次元]
    captions: 正解キャプション [バッチサイズ, 系列長]
    '''
    def forward(self, features: torch.Tensor,
                captions: torch.Tensor):
        seq = captions.shape[1]

        # 単語埋め込み [バッチサイズ, 系列長]
        # -> [バッチサイズ, 系列長, 埋め込み次元]
        embeddings = self.embed(captions)

        # 位置エンコーディング
        embeddings = self.positional_encoding(embeddings)

        features = features.unsqueeze(1)

        # 未来のキャプションを参照しないようにマスク行列を生成
        tgt_mask = torch.tril(features.new_ones((seq, seq)))
        tgt_mask = tgt_mask == 0

        # Transformerデコーダでキャプション生成
        # 画像の特徴も入力する
        for layer in self.decoder_layers:
            embeddings = layer(embeddings, features, tgt_mask)

        # [バッチサイズ, 系列長, 埋め込み次元]
        # -> [バッチサイズ, 系列長, 辞書の次元]
        preds = self.linear(embeddings)

        return preds

    '''
    CaptioningTransformerのサンプリング処理
    features  : 画像特徴ベクトル [バッチサイズ, 埋め込み次元]
    word_to_id: 単語->単語ID辞書
    max_length: 最大キャプション長
    '''
    @torch.no_grad()
    def sample(self, features: torch.Tensor, word_to_id: list,
               max_length: int=30):
        bs = features.shape[0]

        # <start> トークンで出力キャプションを初期化
        captions = features.new_full(
            (bs, 1), word_to_id['<start>'], dtype=torch.int64)

        # 単語を逐次予測
        for _ in range(max_length):
            preds = self.forward(features, captions)
            preds = preds[:, -1]
            preds = preds.softmax(dim=1)
            words = preds.argmax(dim=1, keepdim=True)

            captions = torch.cat((captions, words), dim=1)

        return captions