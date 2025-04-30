import os
import torch
import numpy as np
import gzip
import json
from collections import defaultdict
from datetime import datetime
from typing import Optional


class FederativeDataset():
    def __init__(self, dict1, dict2, user_map1, user_map2, maxlen_A, maxlen_B):
        self.common_list, self.num_common_ids = self.merge_with_common_ids_comprehensive(dict1, dict2, user_map1, user_map2)
        self.maxlen_A, self.maxlen_B = maxlen_A, maxlen_B
    
    def __len__(self):
        return self.num_common_ids
    
    def __getitem__(self, idx):
        # [a, b, c, d] -> [a, b, c] - hist, [d] - trg
        # id1, id2 = self.common_list[idx][0], self.common_list[idx][2]
        list_A, list_B = self.common_list[idx][1], self.common_list[idx][3],
        
       
        seq_A = np.zeros([self.maxlen_A], dtype=np.int32)
        seq_B = np.zeros([self.maxlen_B], dtype=np.int32)
        
        idx = self.maxlen_A - 1
        for i in reversed(list_A):
            seq_A[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        idx = self.maxlen_B - 1
        for i in reversed(list_B):
            seq_B[idx] = i
            idx -= 1
            if idx == -1:
                break
                
        # Convert to torch tensors
        return {
            'seq_A': torch.LongTensor(seq_A),
            'seq_B': torch.LongTensor(seq_B),
        }

    def merge_with_common_ids_comprehensive(self, dict1, dict2, user_map1, user_map2):
        '''
        merge two datasets into one
        id1: list[items1], id2: list[items2] -> real_id: (id1, list[items1], id2, list[items2])
        '''
        # Find common IDs
        common_ids = set(user_map1.keys()) & set(user_map2.keys())
        num_common_ids = len(common_ids)
        
        # Create the merged result
        result = []

        for common_id in common_ids:
            id1, id2 = user_map1[common_id], user_map2[common_id]
            result.append((id1, dict1[id1], id2, dict2[id2]))
        
        return result, num_common_ids