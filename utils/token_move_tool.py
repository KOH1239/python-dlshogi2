import re
import MeCab

def words_with_shogi_move(comment):
    # 棋譜表現の正規表現
    shogi_move_pattern = re.compile(r"""
        (▲|△)?  # 先手または後手（省略可能）
        ([1-9１-９][一二三四五六七八九]|同)  # 盤上の座標（半角・全角対応）or "同"
        (王|玉|飛|龍|竜|角|馬|金|銀|成銀|桂|成桂|香|成香|歩|と)  # 駒の種類
        (左|右)?  # 左または右
        (上|直|引|寄)?  # 動きの方向
        (成|不成|打)?  # 成、不成、または打
    """, re.VERBOSE)

    # MeCabの準備
    mecab = MeCab.Tagger("-Owakati")  # 分かち書き用

    # 棋譜表記を抽出（リストに格納）
    matches = [match.group() for match in shogi_move_pattern.finditer(comment)]

    # 棋譜表記のプレースホルダー置換
    modified_sentence = comment
    for i, move in enumerate(matches):
        modified_sentence = modified_sentence.replace(move, f"▼{i}▼", 1)  # 1回ずつ置き換え

    # MeCabで分かち書き
    split_sentence = mecab.parse(modified_sentence).strip().split()

    # プレースホルダーを元の棋譜表記に戻す
    result = []
    found_star = False  # "▼" が現れたことを確認するフラグ

    for word in split_sentence:
        if word == "▼":  # "▼"が来た場合
            if not found_star:  # 最初の"▼"のみインデックスとして扱う
                found_star = True  # "▼" が見つかったのでフラグを立てる
            else:
                found_star = False   # 2回目以降の"▼"は無視してそのまま追加
        elif found_star:  # "▼" の後が数字の場合
            try:
                index = int(word)  # 次のwordはインデックス
                result.append(matches[index])  # 対応する棋譜表記を追加
            except (IndexError, ValueError) as e:
                print(f"Error: {e} '{word}' in the comment: {comment}")
                pass  # 数字でない場合はスキップ
            found_star = True  # フラグをリセット
        else:
            result.append(word)
    return result