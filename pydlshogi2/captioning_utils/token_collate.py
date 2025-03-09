import torch
from torch import nn
import sqlite3
from typing import Sequence, Dict, Tuple, Union
from pydlshogi2.captioning_utils.token_move_tool import words_with_shogi_move
'''
batch     : features1, features2, move_label, result, コメントインデックスをまとめたもの
word_to_id: 単語->単語ID辞書
'''
def collate_func(x1, x2, move_label, result, index, word_to_id: Dict[str, int], db_path):

    # SQLiteデータベースに接続
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # index のすべての要素に対して一度に SQL クエリを実行
    index_values = [idx.item() for idx in index]
    placeholders = ', '.join(['?'] * len(index_values))  # SQLクエリ用のプレースホルダ
    cursor.execute(f'SELECT * FROM comments WHERE comment_index IN ({placeholders})', tuple(index_values))
    rows = cursor.fetchall()

    # 取得したデータを使ってトークナイズ
    captions = []
    for row in rows:
        caption = row[1]  # キャプションが row[1] にあると仮定
        captions.append(tokenize_caption(caption, word_to_id))

    # 接続を閉じる
    conn.close()
    
    # キャプションの長さが降順になるように並び替え
    batch = zip(x1, x2, move_label, result, captions)
    batch = sorted(batch, key=lambda x: len(x[4]), reverse=True)
    x1, x2, move_label, result, captions = zip(*batch)
    x1 = torch.stack(x1)
    x2 = torch.stack(x2)
    move_label = torch.stack(move_label)
    result = torch.stack(result)

    lengths = [cap.shape[0] for cap in captions]
    targets = torch.full((len(captions), max(lengths)), 
                         word_to_id['<null>'], dtype=torch.int64)
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    return x1, x2, move_label, result, targets, lengths


'''
トークナイザ - 文章(caption)を単語IDのリスト(tokens_id)に変換
caption   : 画像キャプション
word_to_id: 単語->単語ID辞書
'''
def tokenize_caption(caption: str, word_to_id: Dict[str, int]):
    tokens = words_with_shogi_move(caption)
    
    tokens_temp = []    
    # 単語についたピリオド、カンマを削除
    for token in tokens:
        if token in {'。', '、', '.', ',', '！', '？', '!', '?'}: 
            continue
        
        tokens_temp.append(token)
    
    tokens = tokens_temp        
        
    # 文章(caption)を単語IDのリスト(tokens_id)に変換
    tokens_ext = ['<start>'] + tokens + ['<end>']
    tokens_id = []
    for k in tokens_ext:
        if k in word_to_id:
            tokens_id.append(word_to_id[k])
        else:
            tokens_id.append(word_to_id['<unk>'])
    
    return torch.Tensor(tokens_id)