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
args = parser.parse_args()

# ディレクトリパスからファイル名部分を取得
dir_name = os.path.basename(os.path.normpath(args.kif_dir))
output_filename = dir_name.split('_', 1)[1]  # 1回だけ分割して右側を取得

kif_file_list = glob.glob(os.path.join(args.kif_dir, '**', '*.kif'), recursive=True)
# print(kif_file_list)
comment_index_map = {}  # インデックスとコメントの対応を保存する辞書
current_index = 0  # コメントに割り当てるインデックス

moves_sequence = []

board = Board()
with open(f"./kif_comment_csv/{output_filename}.csv", "w", newline="", encoding="utf-8") as comment_file:
    writer = csv.writer(comment_file)
    writer.writerow(["sfen", "Comment"])
    
    for file_path in kif_file_list:
        kif = KIF.Parser.parse_file(file_path)
        # 開始局面を設定
        board.set_sfen(kif.sfen)
        try:
            for i, (move, comment) in enumerate(zip(kif.moves, kif.comments)):
                # 不正な指し手のある棋譜を除外
                if not board.is_legal(move):
                    raise Exception()
                filter_comment = remove_no_need_comment(comment, kif.names)

                board.push(move)
                sfen = board.sfen()
                
                if filter_comment is not None:
                    writer.writerow([sfen, filter_comment])
        except Exception as e:
            print(f'skip {file_path}')
            print(f"Exception: {e}")
            continue