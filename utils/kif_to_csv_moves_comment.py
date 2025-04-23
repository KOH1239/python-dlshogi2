import argparse
from cshogi import *
from cshogi import Board
from cshogi import KIF
import os
import glob
from dlshogi.utils.remove_comment import remove_no_need_comment
from cshogi.KIF import move_to_kif
import csv
import re

parser = argparse.ArgumentParser()
parser.add_argument('kif_dir')
parser.add_argument('--filter_moves', type=int, default=50)
args = parser.parse_args()

kif_file_list = glob.glob(os.path.join(args.kif_dir, '**', '*.kif'), recursive=True)
# print(kif_file_list)
comment_index_map = {}  # インデックスとコメントの対応を保存する辞書
current_index = 0  # コメントに割り当てるインデックス

moves_sequence = []

board = Board()
with open("./comments_A_to_R.csv", "w", newline="", encoding="utf-8") as comment_file:
    writer = csv.writer(comment_file)
    writer.writerow(["Moves", "Comment"])
    
    for file_path in kif_file_list:
        kif = KIF.Parser.parse_file(file_path)
        # 投了、千日手、宣言勝ちで終了した棋譜以外を除外
        if kif.endgame not in ('%TORYO', '%SENNICHITE', '%KACHI'):
            continue
        # 手数が少ない棋譜を除外
        if len(kif.moves) < args.filter_moves:
            continue

        try:
            for i, (move, comment) in enumerate(zip(kif.moves, kif.comments)):
                # 不正な指し手のある棋譜を除外
                if not board.is_legal(move):
                    raise Exception()
                filter_comment = remove_no_need_comment(comment, kif.names)

                if i == 0:
                    ja_move = move_to_kif(move, for_caption=True)
                else:
                    prev_move = kif.moves[i-1]
                    ja_move = move_to_kif(move, prev_move, for_caption=True)
                
                turn = "▲" if i % 2 == 0 else "△"
                formatted_move = turn + ja_move

                # ()とその中の数字を削除
                clean_move = re.sub(r"\(\d+\)", "", formatted_move)
                # "同 " を "同" に置換
                clean_move = re.sub(r"同\s+", "同", clean_move)

                # 「全」→「成銀」、「圭」→「成桂」、「杏」→「成香」に変換
                clean_move = clean_move.replace("全", "成銀").replace("圭", "成桂").replace("杏", "成香")

                moves_sequence.append(clean_move)

                if filter_comment is not None:
                    # 現在のコメントが出るまでの全指し手を保存
                    moves_str = "".join(moves_sequence)
                    writer.writerows((moves_str, filter_comment))
        except Exception as e:
            print(f'skip {file_path}')
            print(f"Exception: {e}")
            continue


