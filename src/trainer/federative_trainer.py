from src.trainer import BaseTrainer
from src.loss import calculate_2_wasserstein_dist

from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import os
import random
import copy


class FederativeTrainer(BaseTrainer):
    """
    Class for federative training of two models: A and B.
    """

    def __init__(
        self,
        model_A,
        optimizer_A,
        lr_scheduler_A,
        dataset_A,  #  = [train, valid, test, usernum, itemnum]

        model_B,
        optimizer_B,
        lr_scheduler_B,
        dataset_B,  #  = [train, valid, test, usernum, itemnum]

        config,
        criterion,
        dataloaders,

        optimizer_frob,
        lr_scheduler_frob,

        device,
        writer,

        idxs_common_A=None,  # user idxs from domain A which active on domain B
        idxs_common_B=None,
        **kwargs,
    ):
        self.config = config
        self.config_trainer = self.config["trainer"]
        self.cfg_trainer_A = self.config["trainer"]["domain_A"]
        self.cfg_trainer_B = self.config["trainer"]["domain_B"]
        self.max_len_A = self.cfg_trainer_A["max_len"]
        self.max_len_B = self.cfg_trainer_B["max_len"]

        self.device = device
        self.writer = writer

        self.model_A = model_A
        self.optimizer_A = optimizer_A
        self.lr_scheduler_A = lr_scheduler_A

        self.model_B = model_B
        self.optimizer_B = optimizer_B
        self.lr_scheduler_B = lr_scheduler_B

        self.criterion = criterion

        self.train_dataloader_A = dataloaders["train_A"]
        self.train_dataloader_B = dataloaders["train_B"]
        self.common_loader = dataloaders["common"]
        self.dataset_A = dataset_A
        self.dataset_B = dataset_B

        self.optimizer_frob = optimizer_frob
        self.lr_scheduler_frob = lr_scheduler_frob

        self.idxs_common_A, self.idxs_common_B = idxs_common_A, idxs_common_B
        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.config_trainer.n_epochs

        self.num_neg =  self.config_trainer.get("num_neg", 100)  # validation negitives

        # define checkpoint dir and init everything if required
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "models"))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
    
    def train_epoch(self, train_config, model, train_dataloader, optimizer, lr_scheduler=None, name="A"):
        model.train()
        for batch in train_dataloader:
            u, seq, pos, neg = batch["user"], batch["seq"], batch["pos"], batch["neg"]

            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(neg_logits.shape, device=self.device)

            optimizer.zero_grad()

            indices = np.where(pos != 0)
            loss = self.criterion(pos_logits[indices], pos_labels[indices])
            loss += self.criterion(neg_logits[indices], neg_labels[indices])

            l2_coef = train_config["l2_emb"]
            for param in model.parameters():
                if param.requires_grad:
                    loss += l2_coef * torch.norm(param) 
            loss.backward()
            # self._clip_grad_norm()
            optimizer.step()
            
            self.writer.log({f"train_loss_{name}":  loss.item()})
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        self.writer.log({f"learning_rate_{name}": lr})
        # self.writer.log({}, commit=True)

        optimizer.zero_grad()
    
    def approximate_epoch(self, emb_type="last_embed"):
        self.optimizer_A.zero_grad()
        self.optimizer_B.zero_grad()
    
        self.model_A.train()
        self.model_B.train()
        loss_function = self.config_trainer.approx.get("loss_function", "Frobenius")
        for batch in self.common_loader:
            self.optimizer_frob.zero_grad()
            
            seq_a, seq_b = batch["seq_A"], batch["seq_B"]
            input_len_a, input_len_b = batch["input_len_a"], batch["input_len_b"]
            
            if emb_type == "last_embed":
                # for sasrec take only last embed from sequence
                emb_a = self.model_A.log2feats(seq_a)[..., -1]
                emb_b = self.model_B.log2feats(seq_b)[..., -1]
            elif emb_type == "out_avg":
                # avg of k output embeddings, for input of size k
                out_a = self.model_A.log2feats(seq_a)
                out_b = self.model_B.log2feats(seq_b)
                emb_a = self.get_out_embed(out_a, input_len_a)
                emb_b = self.get_out_embed(out_b, input_len_b)
            elif emb_type == "input_avg":
                # avg of k input embeddings, for input of size k
                inp_emb_a = self.model_A.item_emb(torch.LongTensor(seq_a).to(self.device))
                inp_emb_b = self.model_B.item_emb(torch.LongTensor(seq_b).to(self.device))
                emb_a = self.get_out_embed(inp_emb_a, input_len_a)
                emb_b = self.get_out_embed(inp_emb_b, input_len_b)
            elif emb_type == "pre_last_layer_last_embed":
                # take only last embed from num_layer-1 output sequence 
                out_a = self.model_A.log2feats(seq_a, get_prev_layer_ouput=True)
                out_b = self.model_B.log2feats(seq_b, get_prev_layer_ouput=True)
                emb_a = self.get_out_embed(out_a, input_len_a)
                emb_b = self.get_out_embed(out_b, input_len_b)

            if loss_function == "Frobenius":
                approx_loss = torch.norm(emb_a - emb_b, p='fro')
            elif loss_function == "Wasserstein":
                approx_loss = calculate_2_wasserstein_dist(emb_a, emb_b, device=self.device)
            else:
                raise NameError(f"Loss should be Frobenius or Wasserstein")
            approx_loss.backward()
            self.optimizer_frob.step()

            self.writer.log({f"loss_{loss_function}_{emb_type}":  approx_loss.item()})
        
        if self.lr_scheduler_frob is not None:
            self.lr_scheduler_frob.step()
        lr = self.optimizer_frob.param_groups[0]['lr']
        self.writer.log({"learning_rate_approx": lr})
        # self.writer.log({}, commit=True)

        self.optimizer_frob.zero_grad()

    def get_out_embed(self, out_seq, input_len):
        '''
        Sum k output embeddings, if input size was k
        out_seq: tensor[bs, seq_len, vec_len]
        input_len: tensor[bs]
        return: tensor[vec_len]
        '''
        bs, seq_len, vec_len = out_seq.shape

        # Create mask
        idxs = torch.arange(seq_len).unsqueeze(0)
        mask = (idxs >= (seq_len - input_len)).int()  # shape (B, seq_len)
        mask = mask.unsqueeze(-1).to(out_seq.device)  # shape (B, seq_len, 1)

        # Apply mask
        masked_data = out_seq * mask

        # Avg
        input_len[input_len == 0] = 1
        divisor = torch.ones((bs, seq_len, 1), dtype=out_seq.dtype, device=out_seq.device)
        divisor = divisor * mask * input_len.view(bs, 1, 1).to(out_seq.device)
        divisor[divisor == 0] = 1
        # print(f"{divisor.shape=}, {masked_data.shape=}, {mask.shape=}")

        return (masked_data / divisor).sum(axis=1)

    def train(self):
        name_A, name_B = self.cfg_trainer_A.dataset["name"], self.cfg_trainer_B.dataset["name"] 
        for epoch in tqdm(range(self.start_epoch, self.epochs)):
            # if epoch % self.config_trainer.get("embed_step_freq", 1) == 0:
            if self.config_trainer.approx.get("multistep", False):
                self.approximate_epoch(emb_type="input_avg")
                self.approximate_epoch(emb_type="pre_last_layer_last_embed")
                self.approximate_epoch(emb_type="out_avg")
            else:
                self.approximate_epoch(emb_type="out_avg")
            self.train_epoch(
                self.cfg_trainer_A, self.model_A,
                self.train_dataloader_A, self.optimizer_A,
                self.lr_scheduler_A, name_A,
            )
            self.train_epoch(
                self.cfg_trainer_B, self.model_B,
                self.train_dataloader_B, self.optimizer_B,
                self.lr_scheduler_B, name_B,
            )
            

            if epoch % self.config_trainer.get("val_freq", 10) == 0:
                self.model_A.eval()
                self.model_B.eval()
                NDCG, HT = self.evaluate_valid(self.model_A, self.dataset_A, self.max_len_A)
                self.writer.log({f"NDCG_{name_A}": NDCG, f"HT_{name_A}": HT}, commit=False)
                NDCG_B, HT_B = self.evaluate_valid(self.model_B, self.dataset_B, self.max_len_B)
                self.writer.log({f"NDCG_{name_B}": NDCG_B, f"HT_{name_B}": HT_B}, commit=False)

                NDCG, HT = self.evaluate_valid(self.model_A, self.dataset_A, self.max_len_A, self.idxs_common_A)
                self.writer.log({f"NDCG_{name_A}_common_users": NDCG, f"HT_{name_A}_common_users": HT}, commit=False)
                NDCG_B, HT_B = self.evaluate_valid(self.model_B, self.dataset_B, self.max_len_B, self.idxs_common_B)
                self.writer.log({f"NDCG_{name_B}_common_users": NDCG_B, f"HT_{name_B}_common_users": HT_B}, commit=False)

                print(f"validation on {epoch=}: {NDCG=}, {HT=}, {NDCG_B=}, {HT_B=}")

            if epoch % self.config_trainer.get("save_freq", 20) == 0:
                self._save_checkpoint(epoch=epoch, name=name_A, model=self.model_A, optimizer=self.optimizer_A, lr_scheduler=self.lr_scheduler_A)
                self._save_checkpoint(epoch=epoch, name=name_B, model=self.model_B, optimizer=self.optimizer_B, lr_scheduler=self.lr_scheduler_B)
            
            self.writer.log({}, commit=True)
    
    def inference(self, dataset_common=None):
        name_A, name_B = self.cfg_trainer_A.dataset["name"], self.cfg_trainer_B.dataset["name"]
        self.model_A.eval()
        self.model_B.eval()
        result = {}
        # NDCG, HT = self.evaluate_valid(self.model_A, self.dataset_A, self.max_len_A, num_neg=self.num_neg)
        # result |= {f"NDCG_{name_A}": NDCG, f"HT_{name_A}": HT}
        # NDCG_B, HT_B = self.evaluate_valid(self.model_B, self.dataset_B, self.max_len_B, num_neg=self.num_neg)
        # result |= {f"NDCG_{name_B}": NDCG_B, f"HT_{name_B}": HT_B}

        # NDCG, HT = self.evaluate_valid(self.model_A, self.dataset_A, self.max_len_A, self.idxs_common_A, num_neg=self.num_neg)
        # result |= {f"NDCG_{name_A}_common_users": NDCG, f"HT_{name_A}_common_users": HT}
        # NDCG_B, HT_B = self.evaluate_valid(self.model_B, self.dataset_B, self.max_len_B, self.idxs_common_B, num_neg=self.num_neg)
        # result |= {f"NDCG_{name_B}_common_users": NDCG_B, f"HT_{name_B}_common_users": HT_B}

        if dataset_common:
            NDCG_A2B_cold, HT_A2B_cold = self.evaluate_cold(
                model_initial=self.model_A,
                model_cold_domain=self.model_B,
                dataset_common=dataset_common,
                dataset_other_domain=self.dataset_B,
                maxlen=self.max_len_A,
                num_neg=self.num_neg,
                cold_domain_first=0,
                test_param=False,
            )
            NDCG_B2A_cold, HT_B2A_cold = self.evaluate_cold(
                model_initial=self.model_B,
                model_cold_domain=self.model_A,
                dataset_common=dataset_common,
                dataset_other_domain=self.dataset_A,
                maxlen=self.max_len_B,
                num_neg=self.num_neg,
                cold_domain_first=1,
                test_param=False,
            )
            result |= {f"NDCG_{name_B}_cold": NDCG_A2B_cold, f"HT_{name_B}_cold": HT_A2B_cold}
            result |= {f"NDCG_{name_A}_cold": NDCG_B2A_cold, f"HT_{name_A}_cold": HT_B2A_cold}

        for key, value in result.items():
            print(f"{key}: {value:.5f}")
        return result


    def evaluate_cold(
            self,
            model_initial,
            model_cold_domain,
            dataset_common,
            dataset_other_domain,
            maxlen,
            num_neg=100,
            cold_domain_first=0,  # 0 or 1
            test_param=False,
        ):
        """
        Args:
            model: model from domain with interactions
            dataset_common: Federarive dataset 
            dataset_other_domain: Dataset containing train, valid, test sets and user/item counts
            maxlen: maxlen from domain with interactions
            num_neg: number of negative in sampled metrixs
            cold_domain_first: 1 if cold domain is second in dataset_common, 0 - first
            test_param: True - test, False - validation
        
        Returns:
            NDCG@10 and HR@10 metrics on cold users
        """
        model_initial.eval()
        model_cold_domain.eval()

        [train_other, valid_other, test_other, usernum_other, itemnum_other] = copy.deepcopy(dataset_other_domain)
        if test_param:
            valid_other = test_other

        NDCG = 0.0
        valid_user = 0.0
        HT = 0.0

        for u, sample in enumerate(dataset_common.common_list):
            train_initial = sample[1 + 2 * cold_domain_first]
            idx_in_cold_dataset = sample[2 - 2 * cold_domain_first]
            if len(train_initial) < 1 or len(valid_other[idx_in_cold_dataset]) < 1: 
                continue
            # Create sequence from user history
            seq = np.zeros(maxlen, dtype=np.int64)
            idx = maxlen - 1
            for i in reversed(train_initial):
                seq[idx] = i
                idx -= 1
                if idx == -1: 
                    break

            # Create set of items user has already interacted with
            rated = set(train_initial)
            rated.add(0)
            
            # Add positive item (from validation) with 100 negative samples
            item_idx = [valid_other[idx_in_cold_dataset][0]]
            for _ in range(num_neg):
                t = np.random.randint(1, itemnum_other + 1)
                while t in rated: 
                    t = np.random.randint(1, itemnum_other + 1)
                item_idx.append(t)

            
            # Get model predictions
            with torch.no_grad():
                item_embs = model_cold_domain.item_emb(torch.LongTensor(item_idx).to(self.device))  # get item embeds from cold domen
                predictions = -model_initial.predict_other_model_items(np.array([seq]), item_embs)
                # print(predictions.shape)
                predictions = predictions[0]

                rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

            # Calculate metrics
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

        # Compute final metrics
        if valid_user > 0:
            NDCG = NDCG / valid_user
            HT = HT / valid_user
        else:
            NDCG = 0
            HT = 0
        return NDCG, HT
