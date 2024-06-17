import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from datamodules import GPTChessDataModule
from model import GPTChessLightning

model_config = {
    "init_from": "scratch",
    "n_layer": 8,
    "n_head": 16,
    "n_embd": 512,
    "block_size": 1024,
    "bias": False,
    "dropout": 0.0,
    "learning_rate": 6e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "weight_decay": 1e-1,
    "lr_decay_iters": 600000,
    "min_lr": 3e-5,
    "meta_path": "data/lichess/meta.pkl",
}

data_config = {
    "dataset_path": "data/lichess",
    "file_path": "lichess_6gb_blocks.zip",
    "batch_size": 128,
    "num_workers": 8,
}

train_config = {
    "max_epochs": 5,
    "val_check_interval": 0.2,
    "log_every_n_steps": 1,
    "overfit_batches": 0,
    "checkpoint_path": "checkpoints/",
    "checkpoint_interval": 10000,
    "wandb_project": "chessgpt",
    "wandb_tags": ["initial_run"],
    "gradient_accumulation_steps": 10,
}


def main():
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    model = GPTChessLightning(model_config)
    dm = GPTChessDataModule(**data_config)

    ckpt_callback = ModelCheckpoint(
        save_last=True,
        dirpath=train_config["checkpoint_path"],
        every_n_train_steps=train_config["checkpoint_interval"],
        verbose=True,
    )

    wandb_logger = WandbLogger(
        project=train_config["wandb_project"], tags=train_config["wandb_tags"]
    )

    trainer = pl.Trainer(
        max_epochs=train_config["max_epochs"],
        log_every_n_steps=train_config["log_every_n_steps"],
        val_check_interval=train_config["val_check_interval"],
        overfit_batches=train_config["overfit_batches"],
        logger=wandb_logger,
        callbacks=[ckpt_callback],
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
