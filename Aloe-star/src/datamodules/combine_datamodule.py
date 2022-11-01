from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from pytorch_lightning.trainer.supporters import CombinedLoader

from src.datamodules.counterfactual_datamodule import CounterfactualDataModule
from src.datamodules.descriptive_datamodule import DescriptiveDataModule
import pandas as pd
import pickle
import os

import logging

class CombineDataModule(LightningDataModule):
    def __init__(
        self,
        descriptive_args: dict,
        counterfactual_args: dict,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.descriptive_module = DescriptiveDataModule(**descriptive_args)
        self.counterfactual_module = CounterfactualDataModule(**counterfactual_args)

        self.data_train: Optional[DataLoader] = None
        self.data_val: Optional[DataLoader] = None
        self.data_test: Optional[DataLoader] = None
        self.check_answer_map()

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self):
        """Check whether answer_map is available or not.
        """
        self.descriptive_module.prepare_data()
        self.counterfactual_module.prepare_data()
    
    def check_answer_map(self):
        self.descriptive_module.check_answer_map()
        self.counterfactual_module.check_answer_map()

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.descriptive_module.setup()
            self.counterfactual_module.setup()

            self.data_train = {
                "descriptive": self.descriptive_module.train_dataloader(), 
                "counterfactual": self.counterfactual_module.train_dataloader()
            }
            self.data_val = {
                "descriptive": self.descriptive_module.val_dataloader(), 
                "counterfactual": self.counterfactual_module.val_dataloader()
            }
            self.data_test = {
                "descriptive": self.descriptive_module.test_dataloader(), 
                "counterfactual": self.counterfactual_module.test_dataloader()
            }

    def train_dataloader(self):
        return CombinedLoader(self.data_train, 'max_size_cycle')

    def val_dataloader(self):
        return CombinedLoader(self.data_val, 'max_size_cycle')

    def test_dataloader(self):
        return CombinedLoader(self.data_test, 'max_size_cycle')
