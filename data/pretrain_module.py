import random
from typing import Optional

import pytorch_lightning as pl
import torch.distributed as dist
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DistributedSampler
from tqdm import tqdm

from utils.common.enums import Term

from .loader import DataLoader
from .pretrain_dataset import PretrainDataset


class PretrainDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for preparing pretraining datasets and
    dataloaders.

    See here for more details on DataModules in PyTorch Lightning:
    https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    """

    def __init__(
        self,
        data_cfg: DictConfig,
        dataset_cfg: DictConfig,
        loader_cfg: DictConfig,
        batch_size: int = 256,
        **kwargs,
    ):
        """
        Initializes the DataModule with the given configurations.

        Args:
            data_cfg (DictConfig): Configuration for which dataset names
                and term to load.
            dataset_cfg (DictConfig): Configuration for arguments to pass to
                each `PretrainDataset`.
            loader_cfg (DictConfig): Configuration for arguments to pass to
                the training and validation dataloaders.
            batch_size (int): Initial batch size for training. Keep this in the
                constructor to allow the tuner to set it later if batch size
                tuning is performed.
            **kwargs: Additional keyword arguments, if any.
        """
        super().__init__()
        self.data_cfg = data_cfg
        self.names = data_cfg.names
        self.sampling_multipliers = data_cfg.get("sampling_multipliers", None)
        self.term = Term(data_cfg.term)

        self.dataset_cfg = dataset_cfg
        self.verbose = dataset_cfg.verbose

        self.loader_cfg = loader_cfg
        self.train_loader_cfg = loader_cfg.train
        self.val_loader_cfg = loader_cfg.val

        # Convert to list for compatability with setup
        if self.sampling_multipliers is None:
            self.sampling_multipliers = [None] * len(self.names)
        else:
            assert len(self.sampling_multipliers) == len(self.names), (
                f"Sampling multipliers must match number of datasets. "
                f"Got {len(self.sampling_multipliers)} multipliers for {len(self.names)} datasets."
            )

        # Ensure number of val data loaders doesn't exceed number of datasets
        self.num_val_loaders = min(
            loader_cfg.num_val_loaders,
            len(self.names),
        )

        # Initialize samplers in case we need to create distributed samplers
        self.train_sampler = None
        self.val_samplers = [None] * self.num_val_loaders
        self.batch_size = loader_cfg.get("batch_size", batch_size)
        self.seed = loader_cfg.seed

        # Save batch size to hparams in case we tune it later
        self.save_hyperparameters("batch_size")

        # Handle additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    def setup(self, stage: Optional[str] = None):
        """
        Setups the datasets for each stage. Should only be used for "fit".
        """
        # Create combined training dataset
        kwargs = {
            "iterable": zip(self.names, self.sampling_multipliers),
            "desc": "Loading training datasets",
            "total": len(self.names),
            "unit": "dataset",
            "disable": False,
        }

        self.train_datasets = [
            PretrainDataset(
                name=name,
                term=self.term,
                sampling_multiplier=multiplier,
                mode="training",
                **self.dataset_cfg,
            )
            for name, multiplier in tqdm(**kwargs)
        ]

        self.combined_train_dataset = ConcatDataset(self.train_datasets)

        # Load `num_val_loaders` val datasets for val data loaders
        self.val_names = random.sample(self.names, k=self.num_val_loaders)

        kwargs = {
            "desc": "Loading validation datasets",
            "total": len(self.val_names),
            "unit": "dataset",
            "disable": not self.verbose,
        }

        self.val_datasets = [
            PretrainDataset(
                name=name,
                term=self.term,
                mode="validation",
                **self.dataset_cfg,
            )
            for name in tqdm(self.val_names, **kwargs)
        ]

        # Initialize distributed samplers if using distributed setup
        if dist.is_initialized() and dist.get_world_size():
            self.train_sampler = DistributedSampler(
                dataset=self.combined_train_dataset,
                shuffle=self.train_loader_cfg.shuffle,
                seed=self.seed,
                drop_last=self.train_loader_cfg.drop_last,
            )

            self.val_samplers = [
                DistributedSampler(
                    dataset=dataset,
                    shuffle=self.val_loader_cfg.shuffle,
                    seed=self.seed,
                    drop_last=self.val_loader_cfg.drop_last,
                )
                for dataset in self.val_datasets
            ]

    def train_dataloader(self):
        """
        Returns the training dataloader, which combines all training datasets
        into a single dataloader.
        """
        # Update training batch if we tuned it
        self.train_loader_cfg["batch_size"] = (
            self.hparams.batch_size
            if self.hparams.batch_size is not None
            else self.train_loader_cfg["batch_size"]
        )
        return DataLoader(
            dataset=self.combined_train_dataset,
            sampler=self.train_sampler,
            **self.train_loader_cfg,
        )

    def val_dataloader(self):
        """
        Returns a list of dataloaders for each validation dataset.
        """
        return [
            DataLoader(
                dataset=dataset,
                sampler=sampler,
                **self.val_loader_cfg,
            )
            for dataset, sampler in zip(self.val_datasets, self.val_samplers)
        ]
