import argparse
from cshogi import *
from cshogi import Board, BLACK, move16
from cshogi import KIF
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from dlshogi.utils.remove_comment import remove_no_need_comment
import logging
import csv

dtypeHcp = np.dtype((np.uint8, 32))
dtypeEval = np.dtype(np.int16)
dtypeMove16 = np.dtype(np.int16)
dtypeGameResult = np.dtype(np.int8)

HuffmanCodedPosAndEvalComment = np.dtype(
    [('hcp', dtypeHcp),
     ('eval', dtypeEval),
     ('bestMove16', dtypeMove16),
     ('gameResult', dtypeGameResult),
     ('dummy', np.uint8),
     ('comment_index', np.int32),
	])

parser = argparse.ArgumentParser()
parser.add_argument('kif_dir')
parser.add_argument('hcpe_train')
parser.add_argument('hcpe_test')
parser.add_argument('--filter_moves', type=int, default=50)
parser.add_argument('--test_ratio', type=float, default=0.1)
args = parser.parse_args()

kif_file_list = glob.glob(os.path.join(args.kif_dir, '**', '*.kif'), recursive=True)

# ファイルリストをシャッフル
file_list_train, file_list_test = train_test_split(kif_file_list, test_size=args.test_ratio)

hcpes = np.zeros(1024, HuffmanCodedPosAndEvalComment)

f_train = open(args.hcpe_train, 'wb')
f_test = open(args.hcpe_test, 'wb')
comment_index_map = {}  # インデックスとコメントの対応を保存する辞書
current_index = 0  # コメントに割り当てるインデックス

board = Board()
with open("./comments_202505_all_clip.csv", "w", encoding="utf-8") as comment_file:
    csv_writer = csv.writer(comment_file)
    csv_writer.writerow(['index', 'comment'])  # ヘッダーを書き込む

    for file_list, f in zip([file_list_train, file_list_test], [f_train, f_test]):
        kif_num = 0
        position_num = 0
        for filepath in file_list:
            kif = KIF.Parser.parse_file(filepath)
            # 投了、千日手、宣言勝ちで終了した棋譜以外を除外
            if kif.endgame not in ('%TORYO', '%SENNICHITE', '%KACHI'):
                continue
            # 手数が少ない棋譜を除外
            if len(kif.moves) < args.filter_moves:
                continue

            # 開始局面を設定
            board.set_sfen(kif.sfen)
            p = 0
            try:
                for i, (move, comment) in enumerate(zip(kif.moves, kif.comments)):
                    # 不正な指し手のある棋譜を除外
                    if not board.is_legal(move):
                        raise Exception()
                    # (改良後のコードの一部)
                    comment = remove_no_need_comment(comment, kif.names)
                    if comment:
                        # 1. コメントを句点「。」で文に分割する
                        #    分割後、各文の前後の空白を除去し、空の文は無視する
                        sentences = [s.strip() for s in comment.split('。') if s.strip()]

                        # 2. 分割された各文に対してループ処理を行う
                        for sentence in sentences:
                            # hcpeバッファに空きがあるか確認
                            if p >= len(hcpes):
                                # バッファが一杯の場合の処理 (例: 警告を出してループを抜ける)
                                # 注: この部分はあなたのプログラム全体のバッファ管理方法に依存します。
                                #     必要に応じて、ここでバッファをファイルに書き出すなどの処理を追加してください。
                                logging.warning("HCPE buffer is full. Skipping remaining sentences for this move.")
                                break
                            
                            hcpe = hcpes[p]
                            
                            # --- 以下の処理は元のコードと同じだが、対象が sentence になる ---
                            
                            # 局面はhcpに変換 (同じ局面情報が、分割された全ての文に対して複製される)
                            board.to_hcp(hcpe['hcp'])
                            # 16bitに収まるようにクリッピングする
                            eval = 0
                            # 手番側の評価値にする
                            hcpe['eval'] = eval if board.turn == BLACK else -eval
                            # 指し手の32bit数値を16bitに切り捨てる
                            hcpe['bestMove16'] = move16(move)
                            # 勝敗結果
                            hcpe['gameResult'] = kif.win
                            
                            # "文 (sentence)" にインデックスを割り当て
                            if sentence not in comment_index_map:
                                comment_index_map[sentence] = current_index
                                # インデックスと「文」をファイルに書き込む
                                csv_writer.writerow([current_index, sentence])
                                current_index += 1
                            hcpe['comment_index'] = comment_index_map[sentence]

                            # 次のhcpeレコードのためにインデックスをインクリメント
                            p += 1
                    
                    # この局面に対する全ての文の処理が終わったら、次の手に進む
                    board.push(move)
            except Exception as e:
                print(f'skip {filepath}')
                print(f"Exception: {e}")
                continue

            if p == 0:
                continue

            hcpes[:p].tofile(f)

            kif_num += 1
            position_num += p

        print('kif_num', kif_num)
        print('position_num', position_num)