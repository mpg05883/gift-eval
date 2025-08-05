from typing import Optional

import pytorch_lightning as pl
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.torch.batchify import batchify
from gluonts.transform import Transformation
from omegaconf import DictConfig

from .train_test_dataset import TrainTestDataset


class TrainTestDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for preparing train-test datasets and
    dataloaders.

    See here for more details on DataModules in PyTorch Lightning:
    https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    """

    def __init__(
        self,
        dataset: TrainTestDataset,
        loader_cfg: DictConfig,
        train_transform: Optional[Transformation] = None,
        val_transform: Optional[Transformation] = None,
        batch_size: int = 256,
        **kwargs,
    ):
        """
        Initializes the DataModule with the given configurations.

        Args:
            dataset (TrainTestDataset): The dataset to use for training and
                validation. Contains the dataset's training and validation
                splits.
            loader_cfg (DictConfig): Configuration for the dataloaders,
                including training and validation settings.
            train_transform (Transformation, optional): Transformation to apply
                to the training dataset. Not necessary when only performing
                validation. Defaults to None.
            val_transform (Transformation): Transformation to apply to the
                validation dataset. Not necessary when only performing
                training. Defaults to None.
            batch_size (int): Initial batch size for training. Keep this in the
                constructor to allow the tuner to set it later if batch size
                tuning is performed.
            **kwargs: Additional keyword arguments, if any.
        """
        super().__init__()
        self.dataset = dataset
        self.loader_cfg = loader_cfg
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = loader_cfg.get("batch_size", batch_size)

        # Handle additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Save batch size to hparams in case we tune it later
        self.save_hyperparameters("batch_size")

    def setup(self, stage: str = None):
        self.training_dataset = self.dataset.training_dataset
        self.validation_dataset = self.dataset.validation_dataset
        self.train_loader_cfg = self.loader_cfg.train
        self.val_loader_cfg = self.loader_cfg.val

    def train_dataloader(self):
        """
        Returns the training set's dataloader.
        """
        return TrainDataLoader(
            self.training_dataset,
            stack_fn=batchify,
            transform=self.train_transform,
            **self.train_loader_cfg,
        )

    def val_dataloader(self):
        """
        Returns the validation set's dataloader.
        """
        return ValidationDataLoader(
            self.validation_dataset,
            stack_fn=batchify,
            transform=self.val_transform,
            **self.val_loader_cfg,
        )
