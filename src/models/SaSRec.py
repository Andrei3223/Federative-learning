import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                         args.num_heads,
                                                         args.dropout_rate)

            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs, get_prev_layer_ouput=False):
        # get the item embedding
        # seqs.shape = [batch_size, 1]
        # print(log_seqs)

        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        # seqs.shape = [batch_size, embeddings]
        # scale the values
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])

        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        # position embeddings
        seqs = self.emb_dropout(seqs)
        # dropout
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim
        # mask the elements, ignore the padding tokens (items with id 0) in the sequences.
        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        # casual masking
        # input the sequence to attention layers
        num_layers = len(self.attention_layers) if not get_prev_layer_ouput else len(self.attention_layers) - 1
        for i in range(num_layers):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)
            # key_padding_mask=timeline_mask
            # need_weights=False this arg do not work?
            # multi-head attention outputs
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)
        # (batch_size, sequence_length, hidden_units)
        return log_feats

    ## SASRec forward
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        # (batch_size, sequence_length, hidden_units)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        # (batch_size, sequence_length, hidden_units)
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        # (batch_size, sequence_length, hidden_units)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        # score for positive engagement = (batch_size, sequence_length)
        neg_logits = (log_feats.unsqueeze(2) * neg_embs).sum(dim=-1)
        # score for negative engagement = (batch_size, sequence_length)
        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)
        # print(item_embs.shape)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)

    def predict_other_model_items(self, log_seqs, item_embs):
        # to evaluate cold users' metrics
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]  # only use last QKV classifier
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits  # preds # (U, I)
    
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


class SASRecSampledLoss(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRecSampledLoss, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        initial_temperature = 1.0  # you can choose any initial value depending on your problem
        self.temperature = torch.nn.Parameter(torch.tensor([initial_temperature], device=self.dev))
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                         args.num_heads,
                                                         args.dropout_rate)

            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        # get the item embedding
        # seqs.shape = [batch_size, 1]
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        # seqs.shape = [batch_size, embeddings]
        # scale the values
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])

        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        # position embeddings
        seqs = self.emb_dropout(seqs)
        # dropout
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim
        # mask the elements, ignore the padding tokens (items with id 0) in the sequences.
        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        # casual masking
        # input the sequence to attention layers
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)
            # key_padding_mask=timeline_mask
            # need_weights=False this arg do not work?
            # multi-head attention outputs
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)
        # (batch_size, sequence_length, hidden_units)
        return log_feats

    ## SASRec forward
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet
        # (batch_size, sequence_length, hidden_units)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        # (batch_size, sequence_length, hidden_units)
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        # (batch_size, sequence_length, num_negs, hidden_units)
        pos_logits = (log_feats * pos_embs).sum(dim=-1) / self.temperature
        # score for positive engagement = (batch_size, sequence_length)
        neg_logits = (log_feats.unsqueeze(2) * neg_embs).sum(dim=-1) / self.temperature
        # score for negative engagement = (batch_size, sequence_length, num_negs)
        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)
        # print(item_embs.shape)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)

