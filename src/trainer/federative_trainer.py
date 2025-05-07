from src.trainer import BaseTrainer
from src.loss import calculate_2_wasserstein_dist

from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import os



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
    
    def approximate_epoch(self):
        self.optimizer_A.zero_grad()
        self.optimizer_B.zero_grad()
    
        self.model_A.train()
        self.model_B.train()
        loss_function = self.config_trainer.approx.get("loss_function", "Frobenius")
        for batch in self.common_loader:
            self.optimizer_frob.zero_grad()
            
            seq_a, seq_b = batch["seq_A"], batch["seq_B"]
            
            # emb_a = self.model_A.log2feats(seq_a)[..., -1]  # for sasrec take only last embed
            # emb_b = self.model_B.log2feats(seq_b)[..., -1]
            input_len_a, input_len_b = batch["input_len_a"], batch["input_len_b"]
            out_a = self.model_A.log2feats(seq_a)
            out_b = self.model_B.log2feats(seq_b)
            
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

            self.writer.log({f"loss_{loss_function}":  approx_loss.item()})
        
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
            self.approximate_epoch()

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
    
    def inference(self):
        name_A, name_B = self.cfg_trainer_A.dataset["name"], self.cfg_trainer_B.dataset["name"]
        self.model_A.eval()
        self.model_B.eval()
        result = {}
        NDCG, HT = self.evaluate_valid(self.model_A, self.dataset_A, self.max_len_A, num_neg=self.num_neg)
        result |= {f"NDCG_{name_A}": NDCG, f"HT_{name_A}": HT}
        NDCG_B, HT_B = self.evaluate_valid(self.model_B, self.dataset_B, self.max_len_B, num_neg=self.num_neg)
        result |= {f"NDCG_{name_B}": NDCG_B, f"HT_{name_B}": HT_B}

        NDCG, HT = self.evaluate_valid(self.model_A, self.dataset_A, self.max_len_A, self.idxs_common_A, num_neg=self.num_neg)
        result |= {f"NDCG_{name_A}_common_users": NDCG, f"HT_{name_A}_common_users": HT}
        NDCG_B, HT_B = self.evaluate_valid(self.model_B, self.dataset_B, self.max_len_B, self.idxs_common_B, num_neg=self.num_neg)
        result |= {f"NDCG_{name_B}_common_users": NDCG_B, f"HT_{name_B}_common_users": HT_B}

        for key, value in result.items():
            print(f"{key}: {value:.5f}")
        return result
