from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.hug_bert_large import *
from src.models.aloe_module_descriptive import AloeModule as PreTrainedAloe
import torch.nn.functional as F
import transformers

import logging

torch.autograd.set_detect_anomaly(True)


class descriptive(nn.Module):
    def __init__(self, hidden_size, max_labels):
        super(descriptive, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, max_labels)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h


class multichoice(nn.Module):
    def __init__(self, hidden_size, max_labels):
        super(multichoice, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, max_labels)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h


class AloeModule(LightningModule):
    def __init__(
        self,
        model_args: dict,
        lr: float = 0.000005,
        weight_decay: float = 0.0,
        num_warmup_steps: int = 1000,
        descriptive_ckpt_path: str = "",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = BertModel(BertConfig()) # BertModel(BertConfig(**model_args["huggingface"]))
        self.model_dc = descriptive(model_args["huggingface"]["trasnformer"]["hidden"], model_args['max_labels'])
        self.model_mc = multichoice(model_args["huggingface"]["trasnformer"]["hidden"], max_labels=2)

        if descriptive_ckpt_path!="":
            logging.info("Loading the pre-trained model checkpoint")
            model2del = PreTrainedAloe(model_args=model_args).load_from_checkpoint(descriptive_ckpt_path)
            self.model.load_state_dict(model2del.model.state_dict())
            self.model_dc.load_state_dict(model2del.model_dc.state_dict())
            del model2del
        else:
            logging.warn("Models are initialized randomly. This might affect the performance.")

        
        # pre-trained BERT model
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")
        
        ## TODO: Play with finetuning bert
        for name, param in self.bert.named_parameters():
            param.requires_grad = False

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc_des, self.train_acc_cnt = Accuracy(), Accuracy()
        self.val_acc_des, self.val_acc_cnt = Accuracy(), Accuracy()
        self.test_acc_des, self.test_acc_cnt = Accuracy(), Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, text_embd: torch.Tensor, visuals_emdb: torch.Tensor, optimizer_idx: int = 0):
        bert_out = self.model(inputs_embeds=text_embd, attention_mask=None, visual_embd=visuals_emdb)
        if optimizer_idx==1:
            outputs = self.model_mc(bert_out.last_hidden_state[:, 0])
        else:
            outputs = self.model_dc(bert_out.last_hidden_state[:, 0])
        return outputs

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        visuals, question, target1 = batch["descriptive"]
        with torch.no_grad():
            text_embd = self.bert(question["ids"], question["mask"]).last_hidden_state
        logits1 = self.forward(text_embd=text_embd, visuals_emdb=visuals, optimizer_idx=0)
        
        loss1 = self.criterion(logits1, target1)
        preds1 = torch.argmax(logits1, dim=1)


        visuals, question, target2 = batch["counterfactual"]
        with torch.no_grad():
            text_embd = self.bert(question["ids"], question["mask"]).last_hidden_state
        logits2 = self.forward(text_embd=text_embd, visuals_emdb=visuals, optimizer_idx=1)
        
        loss2 = self.criterion(logits2, target2)
        preds2 = torch.argmax(logits2, dim=1)

        return loss1+loss2, (preds1, preds2), (target1, target2)

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc_des = self.train_acc_des(preds[0], targets[0])
        acc_cnt = self.train_acc_cnt(preds[1], targets[1])
        
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc_des", acc_des, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc_cnt", acc_cnt, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc_des = self.val_acc_des(preds[0], targets[0])
        acc_cnt = self.val_acc_cnt(preds[1], targets[1])
        
        self.log("val/acc_des", acc_des, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_cnt", acc_cnt, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc_des = self.val_acc_des.compute()  # get val accuracy from current epoch
        acc_cnt = self.val_acc_cnt.compute()  # get val accuracy from current epoch
        
        self.val_acc_best.update(acc_des+acc_cnt)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc_des = self.test_acc_des(preds[0], targets[0])
        acc_cnt = self.test_acc_cnt(preds[1], targets[1])
        
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc_des", acc_des, on_step=False, on_epoch=True)
        self.log("test/acc_cnt", acc_cnt, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        for acc in [self.train_acc_des, self.test_acc_des, self.val_acc_des, self.train_acc_cnt, self.test_acc_cnt, self.val_acc_cnt]:
            acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        logging.info(f"Total estimated stepping batches are: {self.trainer.estimated_stepping_batches}")
        opt = torch.optim.RAdam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        # steps_per_epoch = (len(self.train_dataloader())//self.batch_size)//self.trainer.accumulate_grad_batches
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.hparams.lr, steps_per_epoch=self.trainer.estimated_stepping_batches, epochs=self.trainer.max_epochs)
        
        return [opt], [{"scheduler": scheduler, "interval": "step"}]
