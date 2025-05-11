import torch
import numpy as np
from collections import defaultdict
from typing import Optional
import random
from src.datasets import AmazonDataset


class BERT4RecDataSet(AmazonDataset):
    def __init__(self, user_train, usernum, itemnum, maxlen, mask_prob, **kwargs):
        self.user_train = user_train
        self.maxlen = maxlen
        self.usernum = usernum
        self.itemnum = itemnum
        self.mask_prob = mask_prob

        self.valid_users = [u for u in range(1, usernum + 1) if len(user_train[u]) > 1]
        self._all_items = set([i for i in range(1, self.itemnum + 1)])

    def __getitem__(self, idx): 
        user = self.valid_users[idx]
        user_seq = list(self.user_train[user])
        tokens = []
        labels = []
        for s in user_seq[-self.maxlen:]:
            prob = np.random.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    # masking
                    tokens.append(self.itemnum + 1)  # mask_index: itemnum + 1, 0: pad, 1~itemnum: item index
                elif prob < 0.9:
                    # noise
                    tokens.extend(self.random_neg_sampling(rated_item = user_seq, itemnum_sample = 1))  # item random sampling
                else:
                    tokens.append(s)
                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        mask_len = self.maxlen - len(tokens)
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return {
            'user': torch.LongTensor([user]),
            'seq': torch.LongTensor(tokens),
            'labels': torch.LongTensor(labels),
            'len_seq': torch.LongTensor([len(self.user_train[user])]),
        }

    def random_neg_sampling(self, rated_item : list, itemnum_sample : int):
        nge_samples = random.sample(list(self._all_items - set(rated_item)), itemnum_sample)
        return nge_samples
