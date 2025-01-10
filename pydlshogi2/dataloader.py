import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging
import torch

from cshogi import Board
from pydlshogi2.features import FEATURES_NUM, make_input_features, make_move_label, make_result

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

class HcpeDataLoader:
    def __init__(self, files, batch_size, device, shuffle=False):
        self.load(files)
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

        self.torch_features = torch.empty((batch_size, FEATURES_NUM, 9, 9), dtype=torch.float32, pin_memory=True)
        self.torch_move_label = torch.empty((batch_size), dtype=torch.int64, pin_memory=True)
        self.torch_result = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)
        self.torch_comment_index = torch.empty((batch_size), dtype=torch.int32, pin_memory=True)

        self.features = self.torch_features.numpy()
        self.move_label = self.torch_move_label.numpy()
        self.result = self.torch_result.numpy().reshape(-1)
        self.comment_index = self.torch_comment_index.numpy()

        self.i = 0
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.board = Board()

    def load(self, files):
        data = []
        if type(files) not in [list, tuple]:
            files = [files]
        for path in files:
            if os.path.exists(path):
                logging.info(path)
                data.append(np.fromfile(path, dtype=HuffmanCodedPosAndEvalComment))
            else:
                logging.warn('{} not found, skipping'.format(path))
        self.data = np.concatenate(data)

    def mini_batch(self, hcpevec):
        self.features.fill(0)
        for i, hcpe in enumerate(hcpevec):
            self.board.set_hcp(hcpe['hcp'])
            make_input_features(self.board, self.features[i])
            self.move_label[i] = make_move_label(
                hcpe['bestMove16'], self.board.turn)
            self.result[i] = make_result(hcpe['gameResult'], self.board.turn)
            self.comment_index[i] = hcpe["comment_index"]

        if self.device.type == 'cpu':
            return (self.torch_features.clone(),
                    self.torch_move_label.clone(),
                    self.torch_result.clone(),
                    self.torch_comment_index.clone(),
                    )
        else:
            return (self.torch_features.to(self.device),
                    self.torch_move_label.to(self.device),
                    self.torch_result.to(self.device),
                    self.torch_comment_index.to(self.device),
                    )

    def sample(self):
        return self.mini_batch(np.random.choice(self.data, self.batch_size, replace=False))

    def pre_fetch(self):
        hcpevec = self.data[self.i:self.i+self.batch_size]
        self.i += self.batch_size
        if len(hcpevec) < self.batch_size:
            return

        self.f = self.executor.submit(self.mini_batch, hcpevec)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            np.random.shuffle(self.data)
        self.pre_fetch()
        return self

    def __next__(self):
        if self.i > len(self.data):
            raise StopIteration()

        result = self.f.result()
        self.pre_fetch()

        return result
