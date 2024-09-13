from dataclasses import asdict

import lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.config import ModelConfig
from src.network import GPT, GPTConfig
from src.utils import filter_config, get_learning_rate, get_metadata


class GPTChessLightning(pl.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = GPT(GPTConfig(**self.get_model_config()))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters(
            {
                **asdict(self.config),
                "vocab_size": self.get_model_config()["vocab_size"],
            }
        )

    def get_model_config(self):
        gpt_config = filter_config(asdict(self.config), GPTConfig)
        meta = get_metadata(self.config.meta_path)
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

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: get_learning_rate(
                self.hparams["learning_rate"],
                self.hparams["warmup_iters"],
                self.hparams["lr_decay_iters"],
                self.hparams["min_lr"],
                step,
            ),
        )
        return [optimizer], [scheduler]
