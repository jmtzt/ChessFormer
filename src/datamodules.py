import pickle
from pathlib import Path

import lightning as pl
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class GPTChessDataset(Dataset):
    """PyTorch Dataset for loading chess data."""

    def __init__(self, data_path, block_size):
        super().__init__()
        self.data = np.memmap(data_path, dtype=np.uint8, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) // (self.block_size + 1)

    def __getitem__(self, idx):
        start_idx = idx * (self.block_size + 1)
        end_idx = start_idx + self.block_size
        x = torch.tensor(self.data[start_idx:end_idx], dtype=torch.long)
        y = torch.tensor(
            self.data[start_idx + 1 : end_idx + 1], dtype=torch.long
        )
        return x, y


class GPTChessDataModule(pl.LightningDataModule):
    """Data Module for Chess data handling."""

    def __init__(
        self,
        dataset_path: str,
        file_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.file_path = file_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.meta_path = self.dataset_path / "meta.pkl"
        self.train_path = self.dataset_path / "train.bin"
        self.val_path = self.dataset_path / "val.bin"
        self.column_name = "transcript"
        self.dtype = np.uint8

        if not self.meta_path.exists():
            raise FileNotFoundError(f"Meta file not found at {self.meta_path}.")

        with open(self.meta_path, "rb") as f:
            meta = pickle.load(f)
        self.stoi = meta["stoi"]
        self.itos = meta["itos"]

    def setup(self, stage=None):
        dataset = load_dataset(
            path=str(self.dataset_path), data_files=self.file_path
        )
        print(f"Loaded dataset: {dataset}")
        split_dataset = dataset["train"].train_test_split(
            test_size=0.01, seed=1337, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")
        self.dataset = split_dataset.map(
            self.process,
            remove_columns=[self.column_name],
            num_proc=self.num_workers,
            desc="tokenizing",
        )

    def process(self, example):
        ids = np.array(
            [self.stoi[c] for c in example[self.column_name]], dtype=self.dtype
        )
        return {"ids": ids, "len": len(ids)}

    def save_preprocessed_data(self):
        # This fn assumes that the dataset is already processed and tokenized
        for split in ["train", "val"]:
            dset = self.dataset[split]
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            print(f"{split} has {arr_len} tokens")
            filename = self.train_path if split == "train" else self.val_path
            arr = np.memmap(
                filename, dtype=self.dtype, mode="w+", shape=(arr_len,)
            )
            print(arr.shape)
            total_batches = 1024
            idx = 0

            for batch_idx in range(total_batches):
                # Shard the dataset manually
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
            print(f"Data for {split} saved to {filename}")

    def train_dataloader(self):
        train_dataset = GPTChessDataset(self.train_path, self.batch_size)
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        val_dataset = GPTChessDataset(self.val_path, self.batch_size)
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    DATASET_PATH = "data/lichess"
    dm = ChessDataModule(
        dataset_path=DATASET_PATH, file_path="lichess_6gb_blocks.zip"
    )
    dm.setup()
    dm.save_preprocessed_data()
