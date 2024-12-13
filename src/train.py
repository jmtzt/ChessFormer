from dataclasses import asdict

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config import DataConfig, ModelConfig, TrainConfig
from src.datamodules import GPTChessDataModule
from src.model import GPTChessLightning


def main():
    model_config = ModelConfig(
        init_from="scratch",
        n_layer=16,
        n_head=8,
        n_embd=512,
        block_size=1024,
        bias=False,
        dropout=0.0,
        learning_rate=0.00036,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.1,
        warmup_iters=0,
        lr_decay_iters=50,
        meta_path="data/stockfish/meta.pkl",
    )

    data_config = DataConfig(
        dataset_path="data/stockfish",
        file_path="stockfish_dataset_blocks.zip",
        batch_size=64,
        num_workers=24,
    )

    train_config = TrainConfig(
        max_epochs=50,
        val_check_interval=0.01,
        log_every_n_steps=10,
        overfit_batches=1,
        checkpoint_path="checkpoints/",
        checkpoint_interval=10000,
        wandb_project="chessgpt",
        wandb_tags=["runpod_stockfish_run"],
        gradient_accumulation_steps=10,
    )
    torch.manual_seed(1337)
    # torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    # torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch.set_float32_matmul_precision("high")

    model = GPTChessLightning(model_config)
    dm = GPTChessDataModule(
        **asdict(data_config), block_size=model_config.block_size
    )

    ckpt_callback = ModelCheckpoint(
        save_last=True,
        dirpath=train_config.checkpoint_path,
        every_n_train_steps=train_config.checkpoint_interval,
        verbose=True,
    )

    wandb_logger = WandbLogger(
        project=train_config.wandb_project,
        tags=train_config.wandb_tags,
        # resume="must",
        # id="exfy7bt4",
    )

    trainer = pl.Trainer(
        max_epochs=train_config.max_epochs,
        log_every_n_steps=train_config.log_every_n_steps,
        val_check_interval=train_config.val_check_interval,
        overfit_batches=train_config.overfit_batches,
        accumulate_grad_batches=train_config.gradient_accumulation_steps,
        logger=wandb_logger,
        callbacks=[ckpt_callback],
        precision="bf16-mixed",
    )
    trainer.fit(
        model,
        datamodule=dm,
        # ckpt_path="/opt/joao/repos/xp/chessgpt/checkpoints/last-v4.ckpt",
    )


if __name__ == "__main__":
    main()
