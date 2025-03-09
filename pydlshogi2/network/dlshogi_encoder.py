import torch
from torch import nn

from dlshogi.network.policy_value_network import policy_value_network
from dlshogi import serializers
from dlshogi.common import MAX_MOVE_LABEL_NUM

class CNNEncoder(nn.Module):
    '''
    Show and tellのエンコーダ
    dim_embedding: 埋め込み次元
    '''
    def __init__(self, dim_embedding: int, arg_network: str, arg_model: str):
        super().__init__()

        # dlshogiのpolicy networkとvalue networkの出力層前までのネットワークをバックボーンネットワークとする
        model = policy_value_network(arg_network)
        serializers.load_npz(arg_model, model)
        self.backbone = model

        # デコーダへの出力
        self.linear = nn.Linear(2*MAX_MOVE_LABEL_NUM*9*9, dim_embedding)

    '''
    エンコーダの順伝播
    features1: 入力1, [バッチサイズ, チャネル数(62), 高さ(9), 幅(9)]
    features2: 入力2, [バッチサイズ, チャネル数(57), 高さ(9), 幅(9)]
    '''
    def forward(self, features1: torch.Tensor, features2: torch.Tensor):
        # 特徴抽出 -> [バッチサイズ, 2*27(MAX_MOVE_LABEL)*9*9]
        # 今回はバックボーンネットワークは学習させない
        with torch.no_grad():
            policy, value, policy_features, value_features, resnet_features = self.backbone(features1, features2)
            policy_features = torch.flatten(policy_features, 1)
            value_features = torch.flatten(value_features, 1)
            features = torch.cat([policy_features, value_features], dim=-1)

        # 全結合
        features = self.linear(features)

        return features
