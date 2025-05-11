import torch
import numpy as np
from collections import defaultdict
from typing import Optional


class AmazonDataset():
    def __init__(self, user_train, usernum, itemnum, maxlen):
        '''
        For SASRec
        '''
        self.user_train = user_train
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen
        
        # Filter valid users (having more than 1 interaction)
        self.valid_users = [u for u in range(1, usernum + 1) if len(user_train[u]) > 1]

    def __len__(self):
        return len(self.valid_users)

    def __getitem__(self, idx):
        # [a, b, c, d] -> [a, b, c] - hist, [d] - trg
        user = self.valid_users[idx]
        
        nxt = self.user_train[user][-1]
        idx = self.maxlen - 1
        num_negs = 1
        ts = set(self.user_train[user])
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen, num_negs], dtype=np.int32)
        
        for i in reversed(self.user_train[user][:-1]):
            # fill the seq with all interacted except the last item
            seq[idx] = i
            pos[idx] = nxt
            # negative sampling
            if nxt != 0:
                neg[idx] = self.random_neq(1, self.itemnum + 1, ts, num_negs)
            nxt = i
            idx -= 1
            if idx == -1:
                break
                
        # Convert to torch tensors
        return {
            'user': torch.LongTensor([user]),
            'seq': torch.LongTensor(seq),
            'pos': torch.LongTensor(pos),
            'neg': torch.LongTensor(neg),
            'len_seq': torch.LongTensor([len(self.user_train[user])]),
        }
    
    def random_neq(self, l, r, s, size):
        # Helper function to sample negative items
        neg_items = []
        for _ in range(size):
            t = np.random.randint(l, r)
            while t in s:
                t = np.random.randint(l, r)
            neg_items.append(t)
        return neg_items