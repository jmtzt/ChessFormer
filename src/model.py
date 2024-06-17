import lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from network import GPT, GPTConfig
from utils import filter_config, get_metadata


class GPTChessLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GPT(GPTConfig(**self.get_model_config()))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters(
            {**self.config, "vocab_size": self.get_model_config()["vocab_size"]}
        )

    def get_model_config(self):
        gpt_config = filter_config(self.config, GPTConfig)
        meta = get_metadata(self.config["meta_path"])
        gpt_config["vocab_size"] = meta["vocab_size"]
        return gpt_config

    def forward(self, x, y):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log(
            "train_loss",
            loss if loss else -1.0,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "lr", lr, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log(
            "val_loss",
            loss if loss else -1.0,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            betas=(self.hparams["beta1"], self.hparams["beta2"]),
            weight_decay=self.hparams["weight_decay"],
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams["lr_decay_iters"],
            eta_min=self.hparams["min_lr"],
        )

        return [optimizer], [scheduler]
