import numpy as np
import torch
from torch import nn
import math
from src.models.BERT4Rec_modules import *


class BERT(nn.Module):
    def __init__(self, bert_max_len, item_num, bert_num_blocks, bert_num_heads,
                 bert_hidden_units, bert_dropout, device, **kwargs):
        super().__init__()

        # fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        max_len = bert_max_len
        item_num = item_num
        n_layers = bert_num_blocks
        heads = bert_num_heads
        vocab_size = item_num + 2
        hidden = bert_hidden_units
        self.hidden = hidden
        dropout = bert_dropout
        self.device = device

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])
        self.out = nn.Linear(hidden, item_num + 1)
        
    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        x = self.out(x)
        return x

    def init_weights(self):
        pass
    
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


# net = BERT(10, 10, 6, 8, 8, 0.4)
# print(net)
