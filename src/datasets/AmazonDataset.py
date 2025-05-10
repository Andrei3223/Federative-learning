import os
import torch
import numpy as np
import gzip
import json
from collections import defaultdict
from datetime import datetime
from typing import Optional


class DataProcessor():
    def __init__(
            self,
            data_path: str,
            meta_path: Optional[str] = None,
            min_hist_len=2,

            User=None,
            usermap=None,
            itemmap=None,
            usernum=None,
            itemnum=None,

            use_file=True,
        ):
        self.data_path = data_path
        self.meta_path = meta_path
        self.min_hist_len = min_hist_len

        self.User = User
        self.user_map = usermap
        self.item_map = itemmap
        self.user_num = usernum
        self.item_num = itemnum

        self.use_file = use_file

    def parse(self, path: str):
        # g = gzip.open(path, 'r')
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                yield json.loads(line)

    def preprocess_dataset(self, out_dir: str, out_name: str, user_idxs: Optional[set[int]] = None):
        '''
        parse dataset, get users and item min_hist_len history, put in the file
        '''
        if self.use_file:
            if os.path.exists(f'{out_dir}/{out_name}.txt'):
                print(f"The file {out_dir}/{out_name}.txt exists!")
                return self.read_and_split_data(f'{out_dir}/{out_name}.txt')
            print(f"Creating file {out_dir}/{out_name}.txt")

        # Read data from json, preprocess it, create f'{out_dir}/{out_name}.txt' file
        # return: [user_train, user_valid, user_test, usernum, itemnum]
        countU = defaultdict(lambda: 0)
        countP = defaultdict(lambda: 0)
        line = 0
        
        os.makedirs(out_dir, exist_ok=True)
        reviews_data = []
        # First pass - count occurrences and write reviews to file
        # with open(f'{out_dir}/reviews_{out_name}.txt' , 'w') as f:
        for l in self.parse(self.data_path):
            line += 1
            reviews_data.append(l)
            # f.write(" ".join([l['reviewerID'], l['asin'], str(l['overall']), str(l['unixReviewTime'])]) + ' \n')
            asin = l['asin']
            rev = l['reviewerID']
            time = l['unixReviewTime']
            countU[rev] += 1
            countP[asin] += 1
        
        # Second pass - create mappings and filter data
        usermap, itemmap = dict(), dict()
        usernum, itemnum = 0, 0

        User = dict()
        line = 0
        
        print(len(countU))
        # for l in self.parse(self.data_path):
        for l in reviews_data:
            line += 1
            asin = l['asin']
            rev = l['reviewerID']
            time = l['unixReviewTime']
            
            if countU[rev] < self.min_hist_len or countP[asin] < self.min_hist_len:
                continue

            if user_idxs and rev not in user_idxs:  # example: use to train only on common users 
                continue

            if rev in usermap:
                userid = usermap[rev]
            else:
                usernum += 1
                userid = usernum
                usermap[rev] = userid
                User[userid] = []
                
            if asin in itemmap:
                itemid = itemmap[asin]
            else:
                itemnum += 1
                itemid = itemnum
                itemmap[asin] = itemid
                
            User[userid].append([time, itemid])
        
        # Sort reviews in User according to time
        for userid in User.keys():
            # assert len(User[userid]) >= self.min_hist_len, f"{len(User[userid])}"
            User[userid].sort(key=lambda x: x[0])
        
        print(f"{usernum=}, {itemnum=}")
        
        if self.use_file:
            with open(f'{out_dir}/{out_name}.txt', 'w') as f:
                for user in User.keys():
                    for i in User[user]:
                        f.write('%d %d\n' % (user, i[1]))
        
        self.User = User
        self.user_map = usermap
        self.item_map = itemmap
        self.user_num = usernum
        self.item_num = itemnum

        user_train = {}
        user_valid = {}
        user_test = {}

        for user in User:
            nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = [item[1] for item in User[user]]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = [item[1] for item in User[user][:-2]]
                user_valid[user] = [User[user][-2][1]] 
                user_test[user] = [User[user][-1][1]]
        print('End of data preprocessing.')
        return user_train, user_valid, user_test, usernum, itemnum
    
    def read_and_split_data(self, file_path):
        # Reads data from file_path and splits it into train, val, test
        user_items = defaultdict(list)
        usernum, itemnum = 0, 0
        
        with open(file_path, 'r') as f:
            for line in f:
                u, i = line.rstrip().split(' ')
                u, i = int(u), int(i)
                
                usernum = max(usernum, u)
                itemnum = max(itemnum, i)
                
                user_items[u].append(i)
        
        user_train, user_valid, user_test = {}, {}, {}
        
        for user in user_items:
            nfeedback = len(user_items[user])
            if nfeedback < 3:
                user_train[user] = user_items[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = user_items[user][:-2]
                user_valid[user] = [user_items[user][-2]]
                user_test[user] = [user_items[user][-1]]
        
        print(f"Dataset loaded: {usernum} users, {itemnum} items")
        print(f"Train set: {sum(len(items) for items in user_train.values())} interactions")
        print(f"Valid set: {sum(len(items) for items in user_valid.values())} interactions")
        print(f"Test set: {sum(len(items) for items in user_test.values())} interactions")
    
        return user_train, user_valid, user_test, usernum, itemnum


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

    
